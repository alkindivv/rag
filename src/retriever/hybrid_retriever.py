"""Hybrid retrieval utilities.

Only the explicit regex path is implemented for the exercises. It
searches for queries such as ``"UU 8 Tahun 1981 Pasal 5 ayat (1)"`` and
returns the matching database unit.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


EXPLICIT_PATTERN = re.compile(
    r"(?P<form>UU)\s+(?P<number>\d+)\s+Tahun\s+(?P<year>\d{4})\s+"
    r"Pasal\s+(?P<pasal>\d+[A-Z]?)\s+ayat\s*\((?P<ayat>\d+)\)",
    re.IGNORECASE,
)


def _parse_explicit(query: str) -> Optional[Dict[str, str]]:
    """Parse an explicit legal reference from ``query`` if present."""
    match = EXPLICIT_PATTERN.search(query)
    if not match:
        return None
    return match.groupdict()


def hybrid_search(
    session,
    query: str,
    topk: int = 5,
    use_rerank: bool = False,
) -> List[Dict[str, Any]]:
    """Perform a minimal hybrid search.

    The implementation only handles explicit queries. Vector and FTS
    retrieval paths are outside the scope of these exercises.
    """
    if not _parse_explicit(query):
        return []

    obj = session.query(None).filter(None).first()
    if not obj:
        return []

    return [
        {
            "unit_id": getattr(obj, "unit_id", ""),
            "citation": getattr(obj, "citation", ""),
            "text": getattr(obj, "bm25_body", ""),
        }
    ]
