from __future__ import annotations
"""Document indexing pipeline."""

from typing import Any, Dict, List

from src.db.models import DocForm, DocStatus, DocumentVector, LegalDocument, LegalUnit, UnitType
from src.db.session import SessionLocal
from src.services.embedding.jina_embedder import JinaEmbedder
from src.utils.logging import get_logger


logger = get_logger(__name__)


def ingest_document_json(doc: Dict[str, Any]) -> None:
    """Ingest parsed document JSON into the database."""

    tree = doc.get("document_tree", {})
    if not tree.get("children"):
        logger.error("empty_document_tree")
        return
    with SessionLocal() as session:
        document = LegalDocument(
            form=DocForm(doc["form"]),
            number=doc["number"],
            year=doc["year"],
            status=DocStatus(doc.get("status", "berlaku")),
            title=doc.get("title"),
        )
        session.add(document)
        session.flush()
        pasal_units: List[LegalUnit] = []
        pasal_texts: List[str] = []
        for pasal in tree["children"]:
            if pasal["type"] != "pasal":
                continue
            pasal_id = f"pasal-{pasal['number']}"
            combined = " ".join(
                [leaf.get("text", "") for leaf in pasal.get("children", [])]
            )
            p_unit = LegalUnit(
                document_id=document.id,
                unit_id=pasal_id,
                unit_type=UnitType.PASAL,
                ordinal=pasal["number"],
                bm25_body=combined,
                citation=f"Pasal {pasal['number']}",
            )
            session.add(p_unit)
            pasal_units.append(p_unit)
            pasal_texts.append(combined)
            for ayat in pasal.get("children", []):
                ay_unit = LegalUnit(
                    document_id=document.id,
                    unit_id=f"{pasal_id}-ayat-{ayat['number']}",
                    unit_type=UnitType.AYAT,
                    parent_unit_id=pasal_id,
                    ordinal=ayat["number"],
                    bm25_body=ayat.get("text"),
                    citation=f"Pasal {pasal['number']} ayat ({ayat['number']})",
                )
                session.add(ay_unit)
        session.commit()
        if pasal_texts:
            embedder = JinaEmbedder()
            vectors = embedder.embed(pasal_texts)
            for unit, vec in zip(pasal_units, vectors):
                session.add(
                    DocumentVector(
                        document_id=document.id,
                        unit_id=unit.unit_id,
                        embedding=vec,
                        doc_form=document.form,
                        doc_year=document.year,
                        doc_number=document.number,
                        doc_status=document.status,
                        pasal_number=unit.ordinal,
                        hierarchy_path=unit.unit_id,
                        token_count=len(unit.bm25_body.split()),
                        char_count=len(unit.bm25_body),
                    )
                )
            session.commit()
