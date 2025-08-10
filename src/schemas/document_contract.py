from __future__ import annotations
"""Canonical document JSON contract with validation."""

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError


class RelationshipItem(BaseModel):
    regulation_reference: str
    reference_link: Optional[str] = None


class Relationships(BaseModel):
    mengubah: List[RelationshipItem] = Field(default_factory=list)
    diubah_dengan: List[RelationshipItem] = Field(default_factory=list)
    mencabut: List[RelationshipItem] = Field(default_factory=list)
    dicabut_dengan: List[RelationshipItem] = Field(default_factory=list)
    menetapkan: List[RelationshipItem] = Field(default_factory=list)


class UjiMateriItem(BaseModel):
    decision_number: str
    pdf_url: str
    decision_content: str
    pasal_affected: List[str] = Field(default_factory=list)
    ayat_affected: List[str] = Field(default_factory=list)
    huruf_affected: List[str] = Field(default_factory=list)
    decision_type: str
    legal_basis: str
    binding_status: str
    conditions: Optional[str] = None
    interpretation: Optional[str] = None


class PathItem(BaseModel):
    type: str
    label: str
    unit_id: str


class TreeNode(BaseModel):
    type: str
    unit_id: str
    number_label: Optional[str] = None
    ordinal_int: Optional[int] = None
    ordinal_suffix: str = ""
    label_display: Optional[str] = None
    seq_sort_key: Optional[str] = None
    citation_string: Optional[str] = None
    path: List[PathItem]

    title: Optional[str] = None
    content: Optional[str] = None
    parent_pasal_id: Optional[str] = None
    local_content: Optional[str] = None
    display_text: Optional[str] = None
    bm25_body: Optional[str] = None
    span: Optional[List[int]] = None
    tags_semantik: Optional[List[str]] = None
    entities: Optional[List[str]] = None

    children: List['TreeNode'] = Field(default_factory=list)


TreeNode.model_rebuild()


class DocumentRoot(BaseModel):
    doc_source: str
    doc_id: str
    doc_type: str
    doc_title: str
    doc_teu: str
    doc_number: str
    doc_form: str
    doc_form_short: str
    doc_year: int
    doc_place_enacted: str
    doc_date_enacted: date
    doc_date_promulgated: date
    doc_date_effective: date
    doc_subject: List[str]
    doc_status: str
    doc_language: str
    doc_location: str
    doc_field: str

    relationships: Relationships

    detail_url: str
    source_url: str
    pdf_url: str
    uji_materi_pdf_url: Optional[str] = None
    uji_materi: List[UjiMateriItem] = Field(default_factory=list)

    pdf_path: str
    text_path: str
    doc_content: Optional[str] = None
    doc_processing_status: str
    last_updated: datetime

    document_tree: TreeNode


def validate_document_json(obj: dict) -> DocumentRoot:
    """Validate and return DocumentRoot, raising verbose errors."""
    try:
        return DocumentRoot.model_validate(obj)
    except ValidationError as exc:  # pragma: no cover - pydantic already tested
        errors = [f"{e['loc']}: {e['msg']}" for e in exc.errors()]
        raise ValueError("Invalid document JSON: " + "; ".join(errors))
