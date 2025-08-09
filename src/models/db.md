


# Database Spec: Legal-RAG (PostgreSQL + SQLAlchemy)

Tujuan: skema database yang konsisten, queryable, dan audit-friendly untuk dokumen hukum Indonesia, mendukung FTS + vektor + relasi hukum.

Konten:
- Enums, naming rules, dan konvensi ID
- DDL PostgreSQL (siap dieksekusi)
- Model SQLAlchemy (satu-satu dengan DDL)
- Indeks, FTS, dan pgvector
- Mapping ingestion JSON (termasuk relationships)
- Catatan migrasi dan validasi

---

## 1) Konvensi Penamaan & ID

- Primary key semua tabel: `id UUID`.
- Kolom metadata dokumen menggunakan prefiks `doc_...` pada tabel turunan (mis. `document_vectors`).
- Kolom relasi antar tabel menggunakan nama eksplisit: `document_id`, `unit_id`, dll.
- `unit_id` (string) adalah path logis unik per unit, contoh: `"UU-2009-37/pasal-1/ayat-2/huruf-b"`.

---

BELOW IS JUST A TEMPLATE, YOU NEED TO IMPLEMENT AND ADJUST IT BASED ON CRAWLER + PDF .JSON EXTRACTED THAT CAN BE ACCESED HERE
data/json

<!-- ## 2) DDL PostgreSQL (implementasi)

```sql
-- Enums
CREATE TYPE doc_form AS ENUM ('UU','PP','PERPU','PERPRES','POJK','PERMEN','PERDA','LAINNYA','SE');
CREATE TYPE doc_status AS ENUM ('Berlaku','Tidak Berlaku');
CREATE TYPE unit_type AS ENUM ('dokumen','buku','bab','bagian','paragraf','pasal','angka_amandement','ayat','huruf','angka');

-- subjects
CREATE TABLE subjects (
  id uuid PRIMARY KEY,
  name varchar(200) UNIQUE NOT NULL
);

-- legal_documents
CREATE TABLE legal_documents (
  id uuid PRIMARY KEY,
  doc_source varchar(100) NOT NULL,
  doc_type varchar(100) NOT NULL,
  doc_title text NOT NULL,
  doc_id varchar(100) UNIQUE, -- e.g. 'UU-2023-6' from source JSON

  doc_number varchar(100) NOT NULL,
  doc_form doc_form NOT NULL,
  doc_form_short varchar(20) NOT NULL,
  doc_year int NOT NULL,

  doc_teu varchar(255),
  doc_place_enacted varchar(255),
  doc_language varchar(100) DEFAULT 'Bahasa Indonesia',
  doc_location varchar(255),
  doc_field varchar(255),
  doc_relationships jsonb,
  doc_uji_materi jsonb,

  doc_date_enacted date,
  doc_date_promulgated date,
  doc_date_effective date,

  doc_status doc_status NOT NULL DEFAULT 'Berlaku',

  doc_detail_url text,
  doc_source_url text,
  doc_pdf_url text,
  doc_uji_materi_pdf_url text,
  doc_pdf_path text,
  doc_text_path text,

  doc_content text,
  doc_content_length int,

  doc_processing_status varchar(50) DEFAULT 'pending',
  doc_last_updated timestamptz,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- legal_units
CREATE TABLE legal_units (
  id uuid PRIMARY KEY,
  document_id uuid REFERENCES legal_documents(id) ON DELETE CASCADE,
  unit_type unit_type NOT NULL,
  unit_id varchar(500) NOT NULL,           -- path unik
  number_label varchar(50),
  ordinal_int int DEFAULT 0,
  ordinal_suffix varchar(10) DEFAULT '',
  label_display varchar(50),
  seq_sort_key varchar(50),
  title text,
  content text,
  local_content text,
  display_text text,
  bm25_body text,
  path jsonb,
  citation_string text,
  parent_pasal_id varchar(500),
  hierarchy_path text,
  content_vector tsvector,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- document_vectors
CREATE TABLE document_vectors (
  id uuid PRIMARY KEY,
  document_id uuid REFERENCES legal_documents(id) ON DELETE CASCADE,
  unit_id varchar(500) NOT NULL,              -- pasal unit_id
  content_type varchar(50) NOT NULL DEFAULT 'pasal',
  embedding vector(1024) NOT NULL,
  embedding_model varchar(100) DEFAULT 'jina-embeddings-v4',
  embedding_version varchar(20) DEFAULT 'v1',
  doc_form doc_form NOT NULL,
  doc_year int NOT NULL,
  doc_number varchar(100) NOT NULL,
  doc_status doc_status NOT NULL,
  bab_number varchar(20),
  pasal_number varchar(20),
  ayat_number varchar(20),
  hierarchy_path text,
  token_count int DEFAULT 0,
  char_count int DEFAULT 0,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- logs
CREATE TABLE vector_search_logs (
  id uuid PRIMARY KEY,
  query_text text,
  query_vector_hash varchar(64),
  filters_used text,
  limit_requested int DEFAULT 10,
  results_found int DEFAULT 0,
  search_duration_ms int DEFAULT 0,
  user_session varchar(100),
  searched_at timestamptz DEFAULT now()
);

-- Indexes
CREATE UNIQUE INDEX uq_doc_form_number_year ON legal_documents(doc_form, doc_number, doc_year);
CREATE INDEX idx_doc_form_year ON legal_documents(doc_form, doc_year);
CREATE INDEX idx_doc_source_status ON legal_documents(doc_source, doc_status);
CREATE INDEX idx_doc_relationships_gin ON legal_documents USING gin (doc_relationships jsonb_path_ops);
CREATE INDEX idx_doc_uji_materi_gin ON legal_documents USING gin (doc_uji_materi jsonb_path_ops);

CREATE UNIQUE INDEX idx_units_doc_unitid ON legal_units(document_id, unit_id);
CREATE INDEX idx_units_type_ord ON legal_units(unit_type, ordinal_int);
CREATE INDEX idx_units_content_vector_gin ON legal_units USING gin (content_vector);

-- HNSW index (pgvector >= 0.7.0, Postgres >= 16)
CREATE INDEX idx_vec_embedding_hnsw ON document_vectors USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX idx_vec_doc_meta ON document_vectors(doc_form, doc_year, doc_number);

CREATE INDEX idx_searchlog_session_time ON vector_search_logs(user_session, searched_at);
```

---

## 3) Model SQLAlchemy (Python)

```python
from sqlalchemy import (
    Column, String, Integer, Text, Date, DateTime, Boolean, ForeignKey, Enum,
    Index, UniqueConstraint, Table
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, TSVECTOR
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import uuid
import enum

Base = declarative_base()

class DocForm(enum.Enum):
    UU = "UU"; PP = "PP"; PERPU = "PERPU"; PERPRES = "PERPRES"; POJK = "POJK"; PERMEN = "PERMEN"; PERDA = "PERDA"; LAINNYA = "LAINNYA"; SE = "SE"

class DocStatus(enum.Enum):
    BERLAKU = "Berlaku"; TIDAK_BERLAKU = "Tidak Berlaku"

class UnitType(enum.Enum):
    DOKUMEN = "dokumen"; BUKU = "buku"; BAB = "bab"; BAGIAN = "bagian"; PARAGRAF = "paragraf"; PASAL = "pasal";
    ANGKA_AMANDEMENT = "angka_amandement"; AYAT = "ayat"; HURUF = "huruf"; ANGKA = "angka"

document_subject = Table(
    "document_subject", Base.metadata,
    Column("document_id", UUID(as_uuid=True), ForeignKey("legal_documents.id", ondelete="CASCADE"), primary_key=True),
    Column("subject_id", UUID(as_uuid=True), ForeignKey("subjects.id", ondelete="CASCADE"), primary_key=True),
)

class Subject(Base):
    __tablename__ = "subjects"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), unique=True, nullable=False, index=True)

class LegalDocument(Base):
    __tablename__ = "legal_documents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_source = Column(String(100), nullable=False, index=True)
    doc_type = Column(String(100), nullable=False)
    doc_title = Column(Text, nullable=False, index=True)
    doc_id = Column(String(100), unique=True, index=True)  # e.g. 'UU-2023-6'
    doc_number = Column(String(100), nullable=False, index=True)
    doc_form = Column(Enum(DocForm), nullable=False, index=True)
    doc_form_short = Column(String(20), nullable=False, index=True)
    doc_year = Column(Integer, nullable=False, index=True)
    doc_teu = Column(String(255))
    doc_place_enacted = Column(String(255))
    doc_language = Column(String(100), default="Bahasa Indonesia")
    doc_location = Column(String(255))
    doc_field = Column(String(255), index=True)
    doc_relationships = Column(JSONB)
    doc_uji_materi = Column(JSONB)
    doc_date_enacted = Column(Date)
    doc_date_promulgated = Column(Date)
    doc_date_effective = Column(Date)
    doc_status = Column(Enum(DocStatus), nullable=False, default=DocStatus.BERLAKU, index=True)
    doc_detail_url = Column(Text)
    doc_source_url = Column(Text)
    doc_pdf_url = Column(Text)
    doc_uji_materi_pdf_url = Column(Text)
    doc_pdf_path = Column(Text)
    doc_text_path = Column(Text)
    doc_content = Column(Text)
    doc_content_length = Column(Integer)
    doc_processing_status = Column(String(50), default="pending", index=True)
    doc_last_updated = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    subjects = relationship("Subject", secondary=document_subject, backref="documents", lazy="joined")
    units = relationship("LegalUnit", back_populates="document", cascade="all, delete-orphan")
    vectors = relationship("DocumentVector", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint('doc_form', 'doc_number', 'doc_year', name='uq_doc_form_number_year'),
        Index('idx_doc_form_year', 'doc_form', 'doc_year'),
        Index('idx_doc_source_status', 'doc_source', 'doc_status'),
    )


class LegalUnit(Base):
    __tablename__ = "legal_units"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("legal_documents.id", ondelete="CASCADE"), index=True)
    document = relationship("LegalDocument", back_populates="units")
    unit_type = Column(Enum(UnitType), nullable=False, index=True)
    unit_id = Column(String(500), nullable=False, index=True)
    number_label = Column(String(50))
    ordinal_int = Column(Integer, default=0)
    ordinal_suffix = Column(String(10), default="")
    label_display = Column(String(50))
    seq_sort_key = Column(String(50), index=True)
    title = Column(Text)
    content = Column(Text)
    local_content = Column(Text)
    display_text = Column(Text)
    bm25_body = Column(Text)
    path = Column(JSONB)
    citation_string = Column(Text)
    parent_pasal_id = Column(String(500), index=True, nullable=True)
    hierarchy_path = Column(Text, index=True)
    content_vector = Column(TSVECTOR)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    __table_args__ = (
        UniqueConstraint('document_id','unit_id', name='uq_units_doc_unitid'),
        Index('idx_units_type_ord', 'unit_type', 'ordinal_int'),
        Index('idx_units_bm25_fts', 'content_vector', postgresql_using='gin'),
    )

class DocumentVector(Base):
    __tablename__ = "document_vectors"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("legal_documents.id", ondelete="CASCADE"), index=True)
    document = relationship("LegalDocument", back_populates="vectors")
    unit_id = Column(String(500), nullable=False, index=True)
    content_type = Column(String(50), nullable=False, index=True, default="pasal")
    embedding = Column(Vector(1024), nullable=False)
    embedding_model = Column(String(100), default='jina-embeddings-v4')
    embedding_version = Column(String(20), default='v1')
    doc_form = Column(Enum(DocForm), nullable=False, index=True)
    doc_year = Column(Integer, nullable=False, index=True)
    doc_number = Column(String(100), nullable=False, index=True)
    doc_status = Column(Enum(DocStatus), nullable=False, index=True)
    bab_number = Column(String(20), index=True)
    pasal_number = Column(String(20), index=True)
    ayat_number = Column(String(20), index=True)
    hierarchy_path = Column(Text, index=True)
    token_count = Column(Integer, default=0)
    char_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    __table_args__ = (
        Index('idx_vec_embedding_hnsw', 'embedding', postgresql_using='hnsw',
              postgresql_with={'m': 16, 'ef_construction': 64},
              postgresql_ops={'embedding': 'vector_cosine_ops'}),
        Index('idx_vec_doc_meta', 'doc_form', 'doc_year', 'doc_number'),
    )

class VectorSearchLog(Base):
    __tablename__ = "vector_search_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text = Column(Text)
    query_vector_hash = Column(String(64))
    filters_used = Column(Text)
    limit_requested = Column(Integer, default=10)
    results_found = Column(Integer, default=0)
    search_duration_ms = Column(Integer, default=0)
    user_session = Column(String(100), index=True)
    searched_at = Column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        Index('idx_searchlog_session_time', 'user_session', 'searched_at'),
    )
```

---

## 4) Mapping Ingestion JSON → DB & Neo4j

* __Raw provenance in Postgres__
  - Simpan apa adanya ke `legal_documents.doc_relationships` dan `legal_documents.doc_uji_materi`.
  - Tujuan: auditability, pelacakan sumber, dan kemudahan ETL ke graph.

* __Graph-only normalization (Neo4j)__
  - Tidak ada normalisasi relasi atau uji materi di Postgres.
  - Pipeline ETL: ekspor dari JSONB → Neo4j nodes/edges.

Contoh ETL ringkas (pseudo-Python):

```python
# relationships → Neo4j
rels = doc.doc_relationships or {}
for rel_type, items in rels.items():
    for it in items:
        neo4j.merge_relationship(
            src_key=doc.doc_id,
            rel_type=rel_type,  # e.g. 'diubah_dengan','mencabut','menetapkan'
            tgt_citation=it.get('regulation_reference'),
            tgt_link=it.get('reference_link')
        )

# uji_materi → Neo4j
for um in (doc.doc_uji_materi or []):
    neo4j.merge_uji_materi(
        doc_key=doc.doc_id,
        decision_number=um.get('decision_number'),
        pdf_url=um.get('pdf_url'),
        content=um.get('decision_content')
    )
```

Catatan: `doc_id` berasal dari sumber (mis. `UU-2023-6`) dan berguna sebagai natural key untuk sinkronisasi Postgres ↔ Neo4j.

---

## 5) Indeks & FTS

- GIN pada `legal_units.content_vector` untuk FTS bahasa Indonesia (gunakan trigger `to_tsvector('indonesian', bm25_body)`).
- HNSW pada `document_vectors.embedding` untuk cosine search.
- GIN pada `legal_documents.doc_relationships` untuk filter JSONB.

---

## 6) Catatan Migrasi & Validasi

- Gunakan Alembic untuk: pembuatan enums, tabel, index, dan trigger FTS.
- Validasi unik: `(doc_form, doc_number, doc_year)` di `legal_documents`.
- Constraint arah relasi: definisikan kamus relasi (mis. `MENCABUT` kebalikan `DICABUT_DENGAN`) untuk materialized view arah balik bila perlu.

## 7 

Neo4j (opsional tapi disertakan)
	•	Nodes: Document{doc_id,form,number,year}, Pasal{unit_id,number}, Ayat{unit_id,number}, Huruf{unit_id,letter}, PutusanMK{no,year}.
	•	Rels:
(:Document)-[:MENGUBAH]->(:Document),
(:Document)-[:MENCABUT]->(:Document),
(:PutusanMK)-[:MENGUJI]->(:Pasal|:Ayat|:Huruf),
(:Document)-[:MEMUAT]->(:Pasal); (:Pasal)-[:MEMUAT]->(:Ayat); (:Ayat)-[:MEMUAT]->(:Huruf).


Contoh query penting (final)

1) Cari leaf eksplisit (FTS):

```sql
SELECT id, unit_id, citation_string, ts_rank(content_vector, plainto_tsquery('indonesian', :q)) AS rank
FROM legal_units
WHERE unit_type IN ('ayat','huruf','angka')
  AND content_vector @@ plainto_tsquery('indonesian', :q)
ORDER BY rank DESC
LIMIT 20;
```

2) Cari pasal semantik (vector) dengan filter meta:
```sql
SELECT id, unit_id, pasal_number
FROM document_vectors
WHERE doc_form='UU' AND doc_year BETWEEN 1980 AND 2025
ORDER BY embedding <=> :qvec
LIMIT 10;
```

3) Ambil semua leaf di satu pasal (untuk render jawaban):
```sql
SELECT id, unit_id, citation_string
FROM legal_units
WHERE unit_type IN ('ayat','huruf','angka')
  AND parent_pasal_id = :pasal_id
ORDER BY ordinal_int, ordinal_suffix;
```
 -->
