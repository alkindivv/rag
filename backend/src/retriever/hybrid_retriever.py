from __future__ import annotations
"""Hybrid retrieval combining explicit regex and placeholders for FTS/vector search."""

import re
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.db.models import LegalDocument, LegalUnit, UnitType

EXPLICIT_RE = re.compile(
    r"UU\s*(\d+)\s*Tahun\s*(\d{4})\s*Pasal\s*(\d+[A-Z]?)\s*(?:ayat\s*\((\d+)\))?\s*(?:huruf\s*([a-z]))?",
    re.IGNORECASE,
)


def _parse_explicit(query: str) -> Optional[Dict[str, str]]:
    m = EXPLICIT_RE.search(query)
    if not m:
        return None
    number, year, pasal, ayat, huruf = m.groups()
    return {
        "doc_number": number,
        "doc_year": year,
        "pasal": pasal,
        "ayat": ayat,
        "huruf": huruf,
    }


def hybrid_search(db: Session, query: str, topk: int = 10, use_rerank: bool = False) -> List[Dict[str, Any]]:
    """Return candidates for a query."""

    exp = _parse_explicit(query)
    if exp:
        q = (
            db.query(LegalUnit)
            .join(LegalDocument)
            .filter(LegalDocument.number == exp["doc_number"], LegalDocument.year == int(exp["doc_year"]))
        )
        unit_type = UnitType.AYAT if exp.get("ayat") else UnitType.PASAL
        q = q.filter(LegalUnit.unit_type == unit_type, LegalUnit.ordinal == (exp.get("ayat") or exp["pasal"]))
        if exp.get("huruf"):
            q = q.join(LegalUnit, LegalUnit.parent_unit_id == LegalUnit.unit_id)
        unit = q.first()
        if unit:
            return [{"unit_id": unit.unit_id, "citation": unit.citation, "text": unit.bm25_body}]
        return []
    # Placeholder for semantic + FTS search
    return []
