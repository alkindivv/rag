"""Utilities for building citation strings and paths."""

from __future__ import annotations

import re
from typing import List, Dict


def build_citation_string(path: List[Dict[str, str]], doc_title: str | None = None) -> str:
    """Build citation string from path components."""
    if not path:
        return doc_title or ""
    if doc_title is None:
        doc_title = path[0].get("label", "")
    parts: List[str] = []
    for item in path[1:]:
        t = item.get("type")
        label = item.get("label", "")
        if t == "pasal":
            parts.append(label)
        elif t == "ayat":
            num = re.sub(r"[^0-9]", "", label)
            parts.append(f"ayat ({num})")
        elif t == "huruf":
            letter = re.sub(r"[^a-zA-Z]", "", label)
            if letter:
                parts.append(f"huruf {letter.lower()}")
        elif t == "angka":
            num = re.sub(r"[^0-9]", "", label)
            parts.append(f"angka {num}")
    if parts:
        return f"{doc_title}, {' '.join(parts)}"
    return doc_title
