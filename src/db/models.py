"""SQLAlchemy models for Legal RAG system."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class DocForm(enum.Enum):
    """Document form types."""
    UU = "UU"
    PP = "PP"
    PERPU = "PERPU"
    PERPRES = "PERPRES"
    POJK = "POJK"
    PERMEN = "PERMEN"
    PERDA = "PERDA"
    LAINNYA = "LAINNYA"
    SE = "SE"


class DocStatus(enum.Enum):
    """Document status types."""
    BERLAKU = "Berlaku"
    TIDAK_BERLAKU = "Tidak Berlaku"


class UnitType(enum.Enum):
    """Legal unit hierarchy types."""
    DOKUMEN = "dokumen"
    BUKU = "buku"
    BAB = "bab"
    BAGIAN = "bagian"
    PARAGRAF = "paragraf"
    PASAL = "pasal"
    ANGKA_AMANDEMENT = "angka_amandement"
    AYAT = "ayat"
    HURUF = "huruf"
    ANGKA = "angka"


# Association table for many-to-many relationship between documents and subjects
document_subject = Table(
    "document_subject",
    Base.metadata,
    Column("document_id", UUID(as_uuid=True), ForeignKey("legal_documents.id", ondelete="CASCADE"), primary_key=True),
    Column("subject_id", UUID(as_uuid=True), ForeignKey("subjects.id", ondelete="CASCADE"), primary_key=True),
)


class Subject(Base):
    """Subject/topic areas for legal documents."""

    __tablename__ = "subjects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), unique=True, nullable=False, index=True)

    # Relationship back to documents
    documents = relationship("LegalDocument", secondary=document_subject, back_populates="subjects")


class LegalDocument(Base):
    """Main legal document table storing metadata and content."""

    __tablename__ = "legal_documents"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Core document metadata
    doc_source = Column(String(100), nullable=False, index=True)
    doc_type = Column(String(100), nullable=False)
    doc_title = Column(Text, nullable=False, index=True)
    doc_id = Column(String(100), unique=True, index=True)  # e.g. 'UU-2025-2'

    # Document identification
    doc_number = Column(String(100), nullable=False, index=True)
    doc_form = Column(Enum(DocForm), nullable=False, index=True)
    doc_form_short = Column(String(20), nullable=False, index=True)
    doc_year = Column(Integer, nullable=False, index=True)

    # Additional metadata
    doc_teu = Column(String(255))
    doc_place_enacted = Column(String(255))
    doc_language = Column(String(100), default="Bahasa Indonesia")
    doc_location = Column(String(255))
    doc_field = Column(String(255), index=True)

    # Raw JSON data for relationships and court decisions
    doc_relationships = Column(JSONB)  # Raw relationships from crawler
    doc_uji_materi = Column(JSONB)     # Raw court decisions from crawler

    # Important dates
    doc_date_enacted = Column(Date)
    doc_date_promulgated = Column(Date)
    doc_date_effective = Column(Date)

    # Status
    doc_status = Column(Enum(DocStatus), nullable=False, default=DocStatus.BERLAKU, index=True)

    # URLs and file paths
    doc_detail_url = Column(Text)
    doc_source_url = Column(Text)
    doc_pdf_url = Column(Text)
    doc_uji_materi_pdf_url = Column(Text)
    doc_pdf_path = Column(Text)
    doc_text_path = Column(Text)

    # Content
    doc_content = Column(Text)
    doc_content_length = Column(Integer)

    # Processing status
    doc_processing_status = Column(String(50), default="pending", index=True)
    doc_last_updated = Column(DateTime(timezone=True))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    subjects = relationship("Subject", secondary=document_subject, back_populates="documents")
    units = relationship("LegalUnit", back_populates="document", cascade="all, delete-orphan")
    vectors = relationship("DocumentVector", back_populates="document", cascade="all, delete-orphan")

    # Table constraints and indexes
    __table_args__ = (
        UniqueConstraint('doc_form', 'doc_number', 'doc_year', name='uq_doc_form_number_year'),
        Index('idx_doc_form_year', 'doc_form', 'doc_year'),
        Index('idx_doc_source_status', 'doc_source', 'doc_status'),
        Index('idx_doc_relationships_gin', 'doc_relationships', postgresql_using='gin'),
        Index('idx_doc_uji_materi_gin', 'doc_uji_materi', postgresql_using='gin'),
    )


class LegalUnit(Base):
    """Hierarchical legal units (pasal, ayat, huruf, angka) from document tree."""

    __tablename__ = "legal_units"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign key to parent document
    document_id = Column(UUID(as_uuid=True), ForeignKey("legal_documents.id", ondelete="CASCADE"), index=True)

    # Unit identification
    unit_type = Column(Enum(UnitType), nullable=False, index=True)
    unit_id = Column(String(500), nullable=False, index=True)  # Unique path like "UU-2025-2/pasal-1/ayat-2"

    # Numbering and ordering
    number_label = Column(String(50))           # e.g., "1", "a", "i"
    ordinal_int = Column(Integer, default=0)    # Numeric ordering
    ordinal_suffix = Column(String(10), default="")  # e.g., "bis", "ter"
    label_display = Column(String(50))          # e.g., "Pasal 1", "a."
    seq_sort_key = Column(String(50), index=True)  # For sorting

    # Content
    title = Column(Text)                        # Title if any
    content = Column(Text)                      # Full content including children
    local_content = Column(Text)                # Just this unit's content
    display_text = Column(Text)                 # Formatted display text
    bm25_body = Column(Text)                    # Text for FTS indexing

    # Hierarchy and navigation
    path = Column(JSONB)                        # Full path from root
    citation_string = Column(Text)              # Human-readable citation
    parent_pasal_id = Column(String(500), index=True, nullable=True)  # Parent pasal for leaf nodes
    parent_ayat_id = Column(String(500), index=True, nullable=True)   # Parent ayat for huruf/angka
    parent_huruf_id = Column(String(500), index=True, nullable=True)  # Parent huruf for angka
    hierarchy_path = Column(Text, index=True)   # Text hierarchy path

    # Full-text search vector
    content_vector = Column(TSVECTOR)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship back to document
    document = relationship("LegalDocument", back_populates="units")

    # Table constraints and indexes
    __table_args__ = (
        UniqueConstraint('document_id', 'unit_id', name='uq_units_doc_unitid'),
        Index('idx_units_type_ord', 'unit_type', 'ordinal_int'),
        Index('idx_units_content_vector_gin', 'content_vector', postgresql_using='gin'),
        Index('idx_units_parent_pasal', 'parent_pasal_id'),
        Index('idx_units_parent_ayat', 'parent_ayat_id'),
        Index('idx_units_parent_huruf', 'parent_huruf_id'),
    )


class DocumentVector(Base):
    """Vector embeddings for pasal-level units."""

    __tablename__ = "document_vectors"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign key to parent document
    document_id = Column(UUID(as_uuid=True), ForeignKey("legal_documents.id", ondelete="CASCADE"), index=True)

    # Unit reference
    unit_id = Column(String(500), nullable=False, index=True)  # Pasal unit_id
    content_type = Column(String(50), nullable=False, index=True, default="pasal")

    # Vector embedding
    embedding = Column(Vector(1024), nullable=False)  # Jina v4 1024-dimensional
    embedding_model = Column(String(100), default='jina-embeddings-v4')
    embedding_version = Column(String(20), default='v1')

    # Document metadata for fast filtering
    doc_form = Column(Enum(DocForm), nullable=False, index=True)
    doc_year = Column(Integer, nullable=False, index=True)
    doc_number = Column(String(100), nullable=False, index=True)
    doc_status = Column(Enum(DocStatus), nullable=False, index=True)

    # Hierarchy metadata for filtering
    bab_number = Column(String(20), index=True)
    pasal_number = Column(String(20), index=True)
    ayat_number = Column(String(20), index=True)
    hierarchy_path = Column(Text, index=True)

    # Content statistics
    token_count = Column(Integer, default=0)
    char_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship back to document
    document = relationship("LegalDocument", back_populates="vectors")

    # Table constraints and indexes
    __table_args__ = (
        Index('idx_vec_embedding_hnsw', 'embedding',
              postgresql_using='hnsw',
              postgresql_with={'m': 16, 'ef_construction': 64},
              postgresql_ops={'embedding': 'vector_cosine_ops'}),
        Index('idx_vec_doc_meta', 'doc_form', 'doc_year', 'doc_number'),
    )


class VectorSearchLog(Base):
    """Logging table for vector search queries."""

    __tablename__ = "vector_search_logs"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Query information
    query_text = Column(Text)
    query_vector_hash = Column(String(64))  # Hash of query vector for deduplication
    filters_used = Column(Text)             # JSON string of filters applied

    # Search parameters and results
    limit_requested = Column(Integer, default=10)
    results_found = Column(Integer, default=0)
    search_duration_ms = Column(Integer, default=0)

    # Session tracking
    user_session = Column(String(100), index=True)

    # Timestamp
    searched_at = Column(DateTime(timezone=True), server_default=func.now())

    # Table indexes
    __table_args__ = (
        Index('idx_searchlog_session_time', 'user_session', 'searched_at'),
    )
