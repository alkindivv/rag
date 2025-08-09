from __future__ import annotations
"""Database models for legal RAG system."""

import uuid
from enum import Enum
from typing import List

from sqlalchemy import (
    Column,
    DateTime,
    Enum as PgEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector


Base = declarative_base()


class DocForm(str, Enum):
    """Legal document form."""

    UU = "uu"
    PP = "pp"


class DocStatus(str, Enum):
    """Legal document status."""

    BERLAKU = "berlaku"
    TIDAK_BERLAKU = "tidak_berlaku"


class UnitType(str, Enum):
    """Type of legal unit."""

    BUKU = "buku"
    BAB = "bab"
    BAGIAN = "bagian"
    PARAGRAF = "paragraf"
    PASAL = "pasal"
    AYAT = "ayat"
    ANGKA = "angka"
    HURUF = "huruf"


class LegalDocument(Base):
    """Metadata of a legal document."""

    __tablename__ = "legal_documents"
    __allow_unmapped__ = True

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    form = Column(PgEnum(DocForm), nullable=False)
    number = Column(String(50), nullable=False)
    year = Column(Integer, nullable=False)
    status = Column(PgEnum(DocStatus), nullable=False, default=DocStatus.BERLAKU)
    title = Column(Text)

    units: List["LegalUnit"] = relationship(
        "LegalUnit", back_populates="document", cascade="all, delete-orphan"
    )
    vectors: List["DocumentVector"] = relationship(
        "DocumentVector", back_populates="document", cascade="all, delete-orphan"
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("form", "number", "year", name="uq_doc_identity"),
    )


class LegalUnit(Base):
    """Smallest queryable unit (ayat/huruf/angka)."""

    __tablename__ = "legal_units"
    __allow_unmapped__ = True

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("legal_documents.id", ondelete="CASCADE"),
        index=True,
    )
    document: LegalDocument = relationship("LegalDocument", back_populates="units")

    unit_id = Column(String(500), nullable=False)
    unit_type = Column(PgEnum(UnitType), nullable=False, index=True)
    parent_unit_id = Column(String(500), index=True)
    ordinal = Column(String(50))
    ordinal_int = Column(Integer)

    title = Column(Text)
    bm25_body = Column(Text)
    content_vector = Column(
        TSVECTOR,
        server_default=func.to_tsvector("indonesian", func.coalesce(bm25_body, "")),
        nullable=False,
    )
    path = Column(Text)
    citation = Column(Text)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_units_doc_unitid", "document_id", "unit_id", unique=True),
        Index("idx_units_type_ord", "unit_type", "ordinal_int"),
        Index("idx_units_bm25_fts", "content_vector", postgresql_using="gin"),
    )


class DocumentVector(Base):
    """Embedding per pasal."""

    __tablename__ = "document_vectors"
    __allow_unmapped__ = True

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("legal_documents.id", ondelete="CASCADE"),
        index=True,
    )
    document: LegalDocument = relationship("LegalDocument", back_populates="vectors")
    unit_id = Column(String(500), nullable=False, index=True)
    embedding = Column(Vector(1024), nullable=False)
    embedding_model = Column(String(100), default="jina-embeddings-v4")
    doc_form = Column(PgEnum(DocForm), nullable=False, index=True)
    doc_year = Column(Integer, nullable=False, index=True)
    doc_number = Column(String(100), nullable=False, index=True)
    doc_status = Column(PgEnum(DocStatus), nullable=False, index=True)
    pasal_number = Column(String(20), index=True)
    hierarchy_path = Column(Text, index=True)
    token_count = Column(Integer, default=0)
    char_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index(
            "idx_vec_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("idx_vec_doc_meta", "doc_form", "doc_year", "doc_number"),
    )
