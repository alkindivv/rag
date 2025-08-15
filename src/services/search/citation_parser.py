"""Citation parser for Indonesian legal references.

Parses queries like:
- "UU 4/2009 Pasal 149 ayat (2) huruf b"
- "pasal 12 ayat (3)"
- "Bab I Bagian Kedua Pasal 3A ayat (1) huruf c"

Returns a dict with normalized fields when detected.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

_ROMAN_RE = r"(?P<bab_roman>i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)"

# Precompiled patterns for performance
_PATTERNS = [
    # UU number with year variations: UU 4/2009, UU No. 4 Tahun 2009
    re.compile(r"uu\s*(?:no\.?\s*)?(?P<uu_number>\d+)\s*(?:/|tahun)\s*(?P<uu_year>\d{4})", re.IGNORECASE),
    # Undang-Undang No. 13 Tahun 2008
    re.compile(r"undang[-\s]*undang\s*(?:no\.?\s*)?(?P<uu_number>\d+)\s*tahun\s*(?P<uu_year>\d{4})", re.IGNORECASE),
    # Pasal with optional suffix (A/B) and optional 'bis'/'ter'
    re.compile(r"pasal\s*(?P<pasal_num>\d+)(?P<pasal_suffix>[a-zA-Z])?\s*(?P<pasal_ext>bis|ter)?", re.IGNORECASE),
    # Ayat (2)
    re.compile(r"ayat\s*\(\s*(?P<ayat_num>\d+)\s*\)", re.IGNORECASE),
    # Huruf b
    re.compile(r"huruf\s*(?P<huruf>[a-z])", re.IGNORECASE),
    # BAB roman or numeric
    re.compile(rf"bab\s*(?:(?P<bab_num>\d+)|{_ROMAN_RE})", re.IGNORECASE),
    # Bagian Kesatu/Kedua/...
    re.compile(r"bagian\s*(?P<bagian>(kesatu|kedua|ketiga|keempat|kelima|keenam|ketujuh|kedelapan|kesembilan|kesepuluh|pertama|kedua))", re.IGNORECASE),
    # Paragraf 3
    re.compile(r"paragraf\s*(?P<paragraf>\d+)", re.IGNORECASE),
]

_BAGIAN_MAP = {
    # Canonicalize variants
    "kesatu": "kesatu",
    "kedua": "kedua",
    "ketiga": "ketiga",
    "keempat": "keempat",
    "kelima": "kelima",
    "keenam": "keenam",
    "ketujuh": "ketujuh",
    "kedelapan": "kedelapan",
    "kesembilan": "kesembilan",
    "kesepuluh": "kesepuluh",
    "pertama": "kesatu",
    "kedua2": "kedua",
}


def parse_citation(query: str) -> Optional[Dict[str, str]]:
    """Parse an explicit legal citation from the query.

    Returns a dict of detected keys or None if nothing explicit is found.
    Keys: uu_number, uu_year, pasal_num, pasal_suffix, pasal_ext, ayat_num, huruf,
          bab_num|bab_roman, bagian, paragraf
    """
    if not query or not query.strip():
        return None

    text = query.strip().lower()

    data: Dict[str, str] = {}

    for pat in _PATTERNS:
        for m in pat.finditer(text):
            groups = {k: v for k, v in m.groupdict().items() if v}
            if not groups:
                continue
            data.update(groups)

    if not data:
        return None

    # Normalize bagian
    if "bagian" in data:
        data["bagian"] = _BAGIAN_MAP.get(data["bagian"].lower(), data["bagian"].lower())

    # Normalize roman BAB to upper
    if "bab_roman" in data:
        data["bab_roman"] = data["bab_roman"].upper()

    # Normalize pasal suffix and extension
    if "pasal_suffix" in data:
        data["pasal_suffix"] = data["pasal_suffix"].upper()
    if "pasal_ext" in data:
        data["pasal_ext"] = data["pasal_ext"].lower()

    return data


def build_unit_path(c: Dict[str, str]) -> Optional[str]:
    """Construct ltree-like unit_path from parsed citation dict.

    Example: uu.4.2009.pasal.149.ayat.2.huruf.b
    """
    if not c:
        return None

    parts = []
    if c.get("uu_number") and c.get("uu_year"):
        parts += ["uu", c["uu_number"], c["uu_year"]]

    if c.get("bab_num") or c.get("bab_roman"):
        parts += ["bab", (c.get("bab_num") or c.get("bab_roman"))]

    if c.get("bagian"):
        parts += ["bagian", c["bagian"]]

    if c.get("paragraf"):
        parts += ["paragraf", c["paragraf"]]

    if c.get("pasal_num"):
        pasal = c["pasal_num"]
        if c.get("pasal_suffix"):
            pasal += c["pasal_suffix"].upper()
        if c.get("pasal_ext"):
            pasal += c["pasal_ext"].lower()
        parts += ["pasal", pasal]

    if c.get("ayat_num"):
        parts += ["ayat", c["ayat_num"]]

    if c.get("huruf"):
        parts += ["huruf", c["huruf"].lower()]

    return ".".join(parts) if parts else None
