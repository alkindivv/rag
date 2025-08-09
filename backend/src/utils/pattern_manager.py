from __future__ import annotations
"""Regex hierarchy for legal document parsing."""

import re
from typing import Dict, Tuple


def get_hierarchy_patterns() -> Dict[str, Tuple[re.Pattern[str], float]]:
    """Return regex patterns for each hierarchy level."""

    flags = re.IGNORECASE | re.MULTILINE
    return {
        "buku": (re.compile(r"^\s*BUKU\s+([IVXLC]+)\b", flags), 1),
        "bab": (re.compile(r"^\s*BAB\s+([IVX]+[A-Z]?)\b", flags), 2),
        "bagian": (re.compile(r"^\s*BAGIAN\s+(?:KE\s*)?(\w+)\b", flags), 3),
        "paragraf": (re.compile(r"^\s*PARAGRAF\s+(?:KE\s*)?(\w+)\b", flags), 4),
        "pasal": (re.compile(r"^\s*Pasal\s+(\d+[A-Z]?)\b", flags), 5),
        "ayat": (re.compile(r"^\s*\(\s*(\d+)\s*\)\s*(.*)", flags), 6),
        "angka": (re.compile(r"^\s*(\d{1,2})\.\s*(.*)", flags), 7.4),
        "huruf": (re.compile(r"^\s*([a-z])\.\s*(.*)", flags), 7.5),
    }


def is_amendment_line(line: str) -> bool:
    """Check if line indicates amendment."""

    return bool(re.search(r"\b(disisipkan|diubah|dicabut|ditambahkan|ketentuan)\b", line, re.IGNORECASE))
