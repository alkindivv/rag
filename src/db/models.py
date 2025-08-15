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
    text,
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
    BERLAKU = "BERLAKU"
    TIDAK_BERLAKU = "TIDAK_BERLAKU"


class UnitType(enum.Enum):
    """Legal unit hierarchy types."""
    DOKUMEN = "DOKUMEN"
    BUKU = "BUKU"
    BAB = "BAB"
    BAGIAN = "BAGIAN"
    PARAGRAF = "PARAGRAF"
    PASAL = "PASAL"
    AYAT = "AYAT"
    HURUF = "HURUF"
    ANGKA = "ANGKA"


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
    # bm25_body = Column(Text)                    # Text for FTS indexing

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    subjects = relationship("Subject", secondary=document_subject, back_populates="documents")
    units = relationship("LegalUnit", back_populates="document", cascade="all, delete-orphan")
    # Removed legacy pasal-only vector storage (`DocumentVector`). Vectors consolidate on `legal_units`.

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

    # Numbering and display
    number_label = Column(String(50))           # e.g., "1", "a", "i"
    label_display = Column(String(50))          # e.g., "Pasal 1", "a."

    # Content
    title = Column(Text)                        # Title if any
    content = Column(Text)                      # Full content including children
    local_content = Column(Text)                # Just this unit's content
    display_text = Column(Text)                 # Formatted display text
 
    # Server-side FTS maintenance column (kept in sync by trigger)
    tsv_content = Column(TSVECTOR)              # Populated by trigger (Indonesian config)
    # Vector embedding for unit-level similarity search (dim=384)
    embedding = Column(Vector(384))

    # Hierarchy and navigation
    # ltree path stored server-side; map as Text here to avoid extra deps
    unit_path = Column(Text)                    # ltree in DB; represented as Text in ORM
    citation_string = Column(Text)              # Human-readable citation
    hierarchy_path = Column(Text, index=True)   # Text hierarchy path
    # Explicit parent linkage (self-referential FK used by ltree traversal helpers)
    parent_unit_id = Column(UUID(as_uuid=True),
                            ForeignKey("legal_units.id", ondelete="SET NULL"),
                            index=True,
                            nullable=True)



    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship back to document
    document = relationship("LegalDocument", back_populates="units")

    # Table constraints and indexes
    __table_args__ = (
        UniqueConstraint('document_id', 'unit_id', name='uq_units_doc_unitid'),
        # Helpful indexes declared at ORM-level
        Index('idx_legal_units_tsv_gin', 'tsv_content', postgresql_using='gin'),
    )


## Removed: DocumentVector model. Vector search will use `legal_units.embedding`.


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


# --- DB extras (extensions, triggers, ANN index, enum cleanup) centralised here ---


def setup_db_extras(engine) -> None:
    """Apply DB-wide DDL that is not expressible purely via ORM metadata.

    - Create extensions: vector, ltree, pg_trgm, unaccent
    - Drop legacy artifacts (document_vectors table, old columns)
    - Ensure unit_path uses ltree type
    - Ensure tsv trigger function and trigger exist
    - Create ANN index on legal_units.embedding (HNSW preferred, fallback IVFFLAT)
    - Cleanup unit_type enum to remove deprecated values
    """
    if engine.url.get_backend_name() != 'postgresql':
        return

    with engine.begin() as conn:
        # Extensions
        conn.execute(text(
            """
            DO $$
            BEGIN
              CREATE EXTENSION IF NOT EXISTS vector;
              CREATE EXTENSION IF NOT EXISTS ltree;
              CREATE EXTENSION IF NOT EXISTS pg_trgm;
              CREATE EXTENSION IF NOT EXISTS unaccent;
            END$$;
            """
        ))

        # Drop legacy table/columns
        conn.execute(text(
            """
            DO $$
            BEGIN
              IF to_regclass('public.document_vectors') IS NOT NULL THEN
                EXECUTE 'DROP TABLE IF EXISTS document_vectors CASCADE';
              END IF;

              IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='legal_units' AND column_name='parent_pasal_id'
              ) THEN EXECUTE 'ALTER TABLE legal_units DROP COLUMN IF EXISTS parent_pasal_id'; END IF;

              IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='legal_units' AND column_name='parent_ayat_id'
              ) THEN EXECUTE 'ALTER TABLE legal_units DROP COLUMN IF EXISTS parent_ayat_id'; END IF;

              IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='legal_units' AND column_name='parent_huruf_id'
              ) THEN EXECUTE 'ALTER TABLE legal_units DROP COLUMN IF EXISTS parent_huruf_id'; END IF;

              IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='legal_units' AND column_name='path'
              ) THEN EXECUTE 'ALTER TABLE legal_units DROP COLUMN IF EXISTS path'; END IF;
            END$$;
            """
        ))

        # Ensure unit_path is ltree-typed
        conn.execute(text(
            """
            DO $$
            DECLARE col_type text;
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns WHERE table_name='legal_units' AND column_name='unit_path'
              ) THEN
                EXECUTE 'ALTER TABLE legal_units ADD COLUMN unit_path ltree';
              ELSE
                SELECT data_type INTO col_type FROM information_schema.columns
                WHERE table_name='legal_units' AND column_name='unit_path';
                IF col_type IS NULL OR col_type LIKE 'character%' OR col_type = 'text' THEN
                  EXECUTE 'ALTER TABLE legal_units ALTER COLUMN unit_path TYPE ltree USING text2ltree(unit_path::text)';
                END IF;
              END IF;
            END$$;
            """
        ))

        # Create GIST index on unit_path (ltree) after type ensured
        conn.execute(text(
            """
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'idx_legal_units_unit_path_gist' AND n.nspname = 'public'
              ) THEN
                EXECUTE 'CREATE INDEX idx_legal_units_unit_path_gist ON legal_units USING GIST (unit_path)';
              END IF;
            END$$;
            """
        ))

        # Ensure tsv_content exists
        conn.execute(text(
            """
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns WHERE table_name='legal_units' AND column_name='tsv_content'
              ) THEN
                EXECUTE 'ALTER TABLE legal_units ADD COLUMN tsv_content tsvector';
              END IF;
            END$$;
            """
        ))

        # Trigger function and trigger
        conn.execute(text(
            """
            CREATE OR REPLACE FUNCTION legal_units_tsvector_update() RETURNS trigger AS $$
            BEGIN
              NEW.tsv_content :=
                setweight(to_tsvector('indonesian', coalesce(unaccent(NEW.title), '')), 'A') ||
                setweight(to_tsvector('indonesian', coalesce(unaccent(NEW.content), '')), 'B') ||
                setweight(to_tsvector('indonesian', coalesce(unaccent(NEW.local_content), '')), 'C');
              RETURN NEW;
            END
            $$ LANGUAGE plpgsql;
            """
        ))
        conn.execute(text(
            """
            DO $$
            BEGIN
              IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_legal_units_tsvector') THEN
                EXECUTE 'CREATE TRIGGER trg_legal_units_tsvector BEFORE INSERT OR UPDATE \
                         ON legal_units FOR EACH ROW EXECUTE FUNCTION legal_units_tsvector_update()';
              END IF;
            END$$;
            """
        ))

        # Vector ANN index (partial)
        conn.execute(text(
            """
            DO $$
            DECLARE has_hnsw boolean := false;
            BEGIN
              SELECT EXISTS (SELECT 1 FROM pg_am WHERE amname = 'hnsw') INTO has_hnsw;
              IF has_hnsw THEN
                EXECUTE 'CREATE INDEX IF NOT EXISTS idx_legal_units_embedding_hnsw \
                         ON legal_units USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL';
              ELSE
                EXECUTE 'CREATE INDEX IF NOT EXISTS idx_legal_units_embedding_ivfflat \
                         ON legal_units USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100) WHERE embedding IS NOT NULL';
              END IF;
            END$$;
            """
        ))

        # Enum cleanup for unit_type
        conn.execute(text(
            """
            DO $$
            DECLARE old_type regtype; old_type_name text;
            BEGIN
              IF to_regclass('public.legal_units') IS NULL THEN RETURN; END IF;
              PERFORM 1 FROM information_schema.columns WHERE table_name='legal_units' AND column_name='unit_type';
              IF NOT FOUND THEN RETURN; END IF;

              SELECT atttypid::regtype, t.typname INTO old_type, old_type_name
              FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid
              WHERE a.attrelid = 'legal_units'::regclass AND a.attname = 'unit_type';

              EXECUTE 'UPDATE legal_units SET unit_type = ''ANGKA'' WHERE unit_type::text = ''ANGKA_AMANDEMENT''';

              IF EXISTS (
                SELECT 1 FROM pg_enum e WHERE e.enumtypid = old_type::oid AND e.enumlabel NOT IN (
                  'DOKUMEN','BUKU','BAB','BAGIAN','PARAGRAF','PASAL','AYAT','HURUF','ANGKA')
              ) THEN
                EXECUTE 'ALTER TABLE legal_units ALTER COLUMN unit_type DROP DEFAULT';
                EXECUTE 'ALTER TABLE legal_units ADD COLUMN unit_type_tmp text';
                EXECUTE 'UPDATE legal_units SET unit_type_tmp = unit_type::text';
                EXECUTE 'ALTER TABLE legal_units DROP COLUMN unit_type';
                EXECUTE format('DROP TYPE %s', old_type);
                EXECUTE format('CREATE TYPE %I AS ENUM (''DOKUMEN'',''BUKU'',''BAB'',''BAGIAN'',''PARAGRAF'',''PASAL'',''AYAT'',''HURUF'',''ANGKA'')', old_type_name);
                EXECUTE format('ALTER TABLE legal_units ADD COLUMN unit_type %I', old_type_name);
                EXECUTE 'UPDATE legal_units SET unit_type = unit_type_tmp::' || quote_ident(old_type_name);
                EXECUTE 'ALTER TABLE legal_units DROP COLUMN unit_type_tmp';
              END IF;
            END$$;
            """
        ))
