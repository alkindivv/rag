"""Explicit legal reference parser for Indonesian statutes.

Parses queries like:
- "UU 4/2009 Pasal 149 ayat (2) huruf b"
- "UU No. 1 Tahun 2023 Pasal 14A ayat (3)"
- Variants with/without symbols and with suffixes (A, B, bis, ter)

Outputs a structured dict and helpful ltree lquery patterns to drive explicit search.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Optional


ROMAN = r"(?P<bab_roman>M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))"
# Pasal number with optional suffix; prefer multi-letter latin suffixes over single-letter
PASAL_NUM = r"(?P<pasal>\d+)(?:(?P<pasal_suffix>bis|ter|quater|[A-Za-z]))?"
AYAT_NUM = r"(?P<ayat>\d+)"
HURUF_CHAR = r"(?P<huruf>[a-z])"
ANGKA_NUM = r"(?P<angka>\d+)"

# UU 4/2009, UU No. 4 Tahun 2009, etc.
UU_PREFIX = r"(?:(UU|Undang[- ]Undang))\s*(No\.|Nomor)?\s*"
UU_NUM_YEAR = r"(?P<uu_num>\d+)\s*(?:/|tahun|Tahun|thn)\s*(?P<uu_year>\d{4})"

# Main compiled patterns
PATTERNS = [
    # UU X/Y ... Pasal N (suffix) ayat (M) huruf a angka 1 (allow leading text)
    re.compile(
        rf".*?{UU_PREFIX}?{UU_NUM_YEAR}.*?(?:Pasal)\s*{PASAL_NUM}"
        rf"(?:\s*ayat\s*[\(\[]?{AYAT_NUM}[\)\]]?)?"
        rf"(?:\s*huruf\s*{HURUF_CHAR})?"
        rf"(?:\s*angka\s*{ANGKA_NUM})?",
        re.IGNORECASE,
    ),
    # Bab ROMAN Pasal N (suffix) ... (allow leading text)
    re.compile(
        rf".*?(?:Bab\s*{ROMAN}).*?(?:Pasal)\s*{PASAL_NUM}"
        rf"(?:\s*ayat\s*[\(\[]?{AYAT_NUM}[\)\]]?)?"
        rf"(?:\s*huruf\s*{HURUF_CHAR})?",
        re.IGNORECASE,
    ),
    # Pasal-only pattern (without UU), common for quick references (allow leading text)
    re.compile(
        rf".*?(?:Pasal)\s*{PASAL_NUM}"
        rf"(?:\s*ayat\s*[\(\[]?{AYAT_NUM}[\)\]]?)?"
        rf"(?:\s*huruf\s*{HURUF_CHAR})?",
        re.IGNORECASE,
    ),
]


@dataclass
class ExplicitParse:
    uu_num: Optional[str] = None
    uu_year: Optional[str] = None
    pasal: Optional[str] = None
    pasal_suffix: Optional[str] = None
    ayat: Optional[str] = None
    huruf: Optional[str] = None
    angka: Optional[str] = None
    bab_roman: Optional[str] = None
    has_cross_ref: bool = False  # e.g., "sebagaimana dimaksud"

    def to_lquery(self) -> Optional[str]:
        """Construct a conservative lquery pattern for unit_path.

        unit_path convention (example):
        UU-2009-4.BAB-... .pasal-149.ayat-2.huruf-b
        We'll pattern using wildcards while pinning known segments.
        """
        if not (self.pasal):
            return None
        # Build base anchored to UU when present, else wildcard UU
        uu_part = f"UU-{self.uu_year}-{self.uu_num}" if (self.uu_num and self.uu_year) else "UU-*"
        pasal_part = (self.pasal + (self.pasal_suffix.upper() if self.pasal_suffix else "")).lower()
        base = f"{uu_part}.*.pasal-{pasal_part}"
        if self.ayat:
            base += f".*.ayat-{self.ayat}"
        if self.huruf:
            base += f".*.huruf-{self.huruf.lower()}"
        if self.angka:
            base += f".*.angka-{self.angka}"
        return base

    def to_exact_ltree(self) -> Optional[str]:
        if not (self.pasal):
            return None
        # exact up to the deepest level provided
        parts = []
        if self.uu_num and self.uu_year:
            parts.append(f"UU-{self.uu_year}-{self.uu_num}")
        # include BAB when available for better specificity (normalized upper)
        if self.bab_roman:
            parts.append(f"BAB-{self.bab_roman.upper()}")
        pasal_seg = (self.pasal + (self.pasal_suffix.upper() if self.pasal_suffix else "")).lower()
        parts.append(f"pasal-{pasal_seg}")
        if self.ayat:
            parts.append(f"ayat-{self.ayat}")
        if self.huruf:
            parts.append(f"huruf-{self.huruf.lower()}")
        if self.angka:
            parts.append(f"angka-{self.angka}")
        return ".".join(parts)


def parse(query: str) -> Optional[ExplicitParse]:
    q = query.strip()
    # cross-reference detection
    has_cross = bool(re.search(r"sebagaimana\s+dimaksud", q, flags=re.IGNORECASE))
    for pat in PATTERNS:
        m = pat.search(q)
        if not m:
            continue
        d = m.groupdict()
        # Normalize roman if present
        bab_roman = d.get("bab_roman")
        return ExplicitParse(
            uu_num=d.get("uu_num"),
            uu_year=d.get("uu_year"),
            pasal=d.get("pasal"),
            pasal_suffix=(d.get("pasal_suffix") or None),
            ayat=d.get("ayat"),
            huruf=d.get("huruf"),
            angka=d.get("angka"),
            bab_roman=bab_roman,
            has_cross_ref=has_cross,
        )
    return None


def build_explicit_filters(query: str) -> Dict[str, Optional[str]]:
    """Helper to produce ltree filters from raw query.

    Returns:
      { 'lquery': str|None, 'ltree_exact': str|None }
    """
    p = parse(query)
    if not p:
        return {"lquery": None, "ltree_exact": None}

    # Prefer exact when fully specified to huruf/angka; else start with lquery
    exact = p.to_exact_ltree()
    if p.ayat or p.huruf or p.angka or p.pasal_suffix:
        return {"lquery": p.to_lquery(), "ltree_exact": exact}
    return {"lquery": p.to_lquery(), "ltree_exact": None}
