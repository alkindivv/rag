Struktur Proyek

apps/legal-rag/
├─ backend/
│  ├─ src/
│  │  ├─ api/
│  │  │  └─ app.py
│  │  ├─ config/
│  │  │  └─ settings.py
│  │  ├─ db/
│  │  │  ├─ models.py
│  │  │  ├─ session.py
│  │  │  └─ migrations/        # alembic scripts (DDL/index)
│  │  ├─ orchestrator/
│  │  │  └─ qa_orchestrator.py
│  │  ├─ pipeline/
│  │  │  ├─ indexer.py
│  │  │  └─ post_index.py
│  │  ├─ retriever/
│  │  │  └─ hybrid_retriever.py
│  │  ├─ services/
│  │  │  ├─ embedding/
│  │  │  │  └─ jina_embedder.py
│  │  │  ├─ rerank/
│  │  │  │  ├─ base.py
│  │  │  │  └─ jina_reranker.py
│  │  │  └─ llm/
│  │  │     ├─ base.py
│  │  │     ├─ factory.py
│  │  │     └─ providers/
│  │  │        ├─ gemini.py
│  │  │        ├─ openai_.py
│  │  │        └─ anthropic_.py
│  │  ├─ prompts/
│  │  │  ├─ system.py
│  │  │  └─ answer_with_citations.py
│  │  ├─ service/
│  │  │  ├─ crawler/            # adapter ke crawler kamu (refactor ringan)
│  │  │  └─ pdf/
│  │  │     ├─ orchestrator.py  # parser legal tree rapi (refactor menyeluruh)
│  │  │     └─ extractor.py     # UnifiedPDFExtractor + IO
│  │  ├─ utils/
│  │  │  ├─ pattern_manager.py  # aturan hierarki legal (konfigurabel)
│  │  │  ├─ text_cleaner.py     # pipeline modular
│  │  │  ├─ http.py             # retry/backoff, DI untuk HTTP client
│  │  │  └─ logging.py          # structured logging (JSON)
│  │  └─ __main__.py            # CLI entry (parse arg terpisah dari bisnis)
│  ├─ requirements.txt
│  └─ run.sh
└─ frontend/
   ├─ app/
   │  ├─ layout.tsx
   │  ├─ page.tsx
   │  └─ chat/
   │     ├─ page.tsx
   │     └─ actions.ts
   ├─ app/api/ask/route.ts
   ├─ components/
   │  ├─ ChatInput.tsx
   │  ├─ MessageList.tsx
   │  ├─ MessageBubble.tsx
   │  └─ SourceCard.tsx
   ├─ lib/
   │  ├─ api.ts
   │  └─ types.ts
   ├─ styles/globals.css
   └─ package.json

Backend

1) config/settings.py — konfigurasi terpusat (Pydantic)

Tujuan: satu tempat untuk environment, gampang dites, dan konsisten di seluruh layer.

# apps/legal-rag/backend/src/config/settings.py
from pydantic import BaseSettings, Field
from typing import Optional

class Settings(BaseSettings):
    # DB
    database_url: str = Field(..., env="DATABASE_URL")

    # LLM
    llm_provider: str = Field("gemini", env="LLM_PROVIDER")  # gemini|openai|anthropic
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")

    # Embedding (Jina v4)
    jina_api_key: str = Field(..., env="JINA_API_KEY")
    jina_embed_base: str = Field("https://api.jina.ai/v1/embeddings", env="JINA_EMBED_BASE")
    jina_embed_model: str = Field("jina-embeddings-v4", env="JINA_EMBED_MODEL")
    embed_batch_size: int = Field(16, env="EMBED_BATCH_SIZE")

    # Reranker (opsional)
    rerank_provider: str = Field("none", env="RERANK_PROVIDER")  # jina|none
    jina_rerank_base: str = Field("https://api.jina.ai/v1/rerank", env="JINA_RERANK_BASE")
    jina_rerank_model: str = Field("jina-reranker-v1", env="JINA_RERANK_MODEL")

    # Server
    enable_stream: bool = Field(True, env="ENABLE_STREAM")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"

settings = Settings()

2) db/models.py — Skema Database (lengkap & terstruktur)

Tujuan: menyimpan metadata dokumen, node hierarki (untuk FTS), vektor (pgvector), uji materi, dan relasi antardokumen. Dirancang tidak over‑engineered tapi enterprise‑ready.

# apps/legal-rag/backend/src/db/models.py
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

# --- ENUMS ---------------------------------------------------------

class DocForm(enum.Enum):
    UU = "UU"
    PP = "PP"
    PERPRES = "PERPRES"
    PERMEN = "PERMEN"
    PERDA = "PERDA"
    LAINNYA = "LAINNYA"

class DocStatus(enum.Enum):
    BERLAKU = "Berlaku"
    DICABUT = "Dicabut"
    BERUBAH = "Berubah Sebagian"

class UnitType(enum.Enum):
    DOKUMEN = "dokumen"
    BUKU = "buku"
    BAB = "bab"
    BAGIAN = "bagian"
    PARAGRAF = "paragraf"
    PASAL = "pasal"
    AYAT = "ayat"
    HURUF = "huruf"
    ANGKA = "angka"

# --- ASSOCIATION TABLES -------------------------------------------

document_subject = Table(
    "document_subject",
    Base.metadata,
    Column("document_id", UUID(as_uuid=True), ForeignKey("legal_documents.id", ondelete="CASCADE"), primary_key=True),
    Column("subject_id", UUID(as_uuid=True), ForeignKey("subjects.id", ondelete="CASCADE"), primary_key=True),
    UniqueConstraint("document_id", "subject_id", name="uq_document_subject")
)

# --- CORE TABLES ---------------------------------------------------

class Subject(Base):
    __tablename__ = "subjects"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), unique=True, nullable=False, index=True)

class LegalDocument(Base):
    """
    Satu baris per dokumen peraturan.
    """
    __tablename__ = "legal_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String(100), nullable=False, index=True)
    type = Column(String(100), nullable=False)  # "Peraturan Perundang-undangan"
    title = Column(Text, nullable=False, index=True)

    number = Column(String(100), nullable=False, index=True)
    form = Column(Enum(DocForm), nullable=False, index=True)
    form_short = Column(String(20), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)

    teu = Column(String(255))
    place_enacted = Column(String(255))
    language = Column(String(100), default="Bahasa Indonesia")
    location = Column(String(255))
    field = Column(String(255), index=True)

    date_enacted = Column(Date)
    date_promulgated = Column(Date)
    date_effective = Column(Date)

    status = Column(Enum(DocStatus), nullable=False, default=DocStatus.BERLAKU, index=True)

    detail_url = Column(Text)
    source_url = Column(Text)
    pdf_url = Column(Text)
    uji_materi_pdf_url = Column(Text)
    pdf_path = Column(Text)
    text_path = Column(Text)

    content = Column(Text)  # full text (opsional)
    content_length = Column(Integer)

    processing_status = Column(String(50), default="pending", index=True)
    last_updated = Column(DateTime(timezone=True))

    # Relations
    subjects = relationship("Subject", secondary=document_subject, backref="documents", lazy="joined")
    units = relationship("LegalUnit", back_populates="document", cascade="all, delete-orphan")
    vectors = relationship("DocumentVector", back_populates="document", cascade="all, delete-orphan")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint('form', 'number', 'year', name='uq_form_number_year'),
        Index('idx_doc_form_year', 'form', 'year'),
        Index('idx_doc_source_status', 'source', 'status'),
    )

    @property
    def identifier(self) -> str:
        return f"{self.form.value} No. {self.number} Tahun {self.year}"

class LegalRelationship(Base):
    """
    Menyimpan relasi antar dokumen (mengubah, diubah_dengan, mencabut, menetapkan, dsb).
    """
    __tablename__ = "legal_relationships"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    from_document_id = Column(UUID(as_uuid=True), ForeignKey("legal_documents.id", ondelete="CASCADE"), index=True)
    to_document_ref = Column(Text, nullable=False)  # referensi teks/URL (kalau belum terdaftar di DB)
    relation_type = Column(String(50), nullable=False, index=True)  # "mengubah","mencabut","diubah_dengan","menetapkan"
    reference_link = Column(Text)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

class UjiMateriDecision(Base):
    """
    Putusan MK terkait dokumen (uji materi).
    """
    __tablename__ = "uji_materi"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("legal_documents.id", ondelete="CASCADE"), index=True)

    decision_number = Column(String(100), nullable=False, index=True)
    pdf_url = Column(Text)
    decision_content = Column(Text)

    decision_type = Column(String(50))
    legal_basis = Column(Text)
    binding_status = Column(Text)
    conditions = Column(JSONB)     # array/string
    interpretation = Column(JSONB) # array/string

    pasal_affected = Column(JSONB)
    ayat_affected = Column(JSONB)
    huruf_affected = Column(JSONB)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

class LegalUnit(Base):
    """
    Node hierarki (dokumen->buku->bab->bagian->paragraf->pasal->ayat->huruf->angka).
    Leaf (ayat/huruf/angka) dipakai FTS, pasal menyimpan content lengkap utk embedding.
    """
    __tablename__ = "legal_units"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("legal_documents.id", ondelete="CASCADE"), index=True)
    document = relationship("LegalDocument", back_populates="units")

    unit_type = Column(Enum(UnitType), nullable=False, index=True)
    unit_id = Column(String(500), nullable=False, index=True)  # "UU-2025-3/pasal-7/ayat-2/huruf-a"
    number_label = Column(String(50))         # "1", "1A", "IV", "a", "3"
    ordinal_int = Column(Integer, default=0)  # untuk sort
    ordinal_suffix = Column(String(10), default="")
    label_display = Column(String(50))        # "Pasal 7", "(2)", "a.", "3."
    seq_sort_key = Column(String(50), index=True)  # "0007|" dst

    title = Column(Text)      # judul node (untuk bab/bagian/paragraf/pasal)
    content = Column(Text)    # khusus PASAL: full isi pasal (join ayat/huruf/angka)
    local_content = Column(Text)  # khusus leaf: isi ayat/huruf/angka
    display_text = Column(Text)   # leaf: label + isi (untuk UI)
    bm25_body = Column(Text)      # leaf: bahan FTS

    path = Column(JSONB)          # breadcrumb array objek {type,label,unit_id}
    citation_string = Column(Text)  # "UU X/XXXX, Pasal 1 ayat (2) huruf b"
    parent_pasal_id = Column(String(500), index=True, nullable=True)  # unit_id pasal induk
    hierarchy_path = Column(Text, index=True)  # gabungan untuk pencarian trigram

    # FTS vektor (opsional via trigger / materialized)
    content_vector = Column(TSVECTOR)  # to_tsvector('indonesian', bm25_body)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_units_doc_unitid', 'document_id', 'unit_id', unique=True),
        Index('idx_units_type_ord', 'unit_type', 'ordinal_int'),
        Index('idx_units_bm25_fts', 'content_vector', postgresql_using='gin'),
    )

class DocumentVector(Base):
    """
    Vektor embedding per PASAL (default 1 vektor/pasal).
    """
    __tablename__ = "document_vectors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("legal_documents.id", ondelete="CASCADE"), index=True)
    document = relationship("LegalDocument", back_populates="vectors")

    unit_id = Column(String(500), nullable=False, index=True)  # unit_id pasal
    content_type = Column(String(50), nullable=False, index=True, default="pasal")

    embedding = Column(Vector(1024), nullable=False)  # Jina v4: 1024 dims
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
        Index('idx_vec_embedding_hnsw', 'embedding',
              postgresql_using='hnsw',
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


4) Alembic Migration (DDL + Index FTS)

Contoh (inti yang penting):
-- pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- FTS index untuk legal_units.bm25_body
CREATE INDEX IF NOT EXISTS idx_legal_units_bm25_tsv
ON legal_units
USING GIN (to_tsvector('indonesian', coalesce(bm25_body,'')));

-- Opsional: generated column + trigger untuk content_vector (tsvector)
-- (Jika mau materialized: buat kolom materialized supaya query cepat)

Kita bisa pakai generated column di Postgres 12+:
ALTER TABLE legal_units ADD COLUMN content_vector tsvector GENERATED ALWAYS AS (to_tsvector('indonesian', coalesce(bm25_body,''))) STORED;
Lalu index GIN di kolom itu (sudah di models juga).

⸻

5) utils/pattern_manager.py — pola hierarki (bisa dikonfigurasi)

Tujuan: seluruh regex & aturan deteksi unit legal dipusatkan di sini (mudah diuji & diubah

# apps/legal-rag/backend/src/utils/pattern_manager.py
import re
from typing import Dict, Tuple

def get_hierarchy_patterns() -> Dict[str, Tuple[re.Pattern, float]]:
    """
    Urutan penting: buku -> bab -> bagian -> paragraf -> pasal -> ayat -> angka -> huruf.
    Angka/huruf inline menangkap konten setelah label.
    """
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
    return bool(re.search(r'\b(disisipkan|diubah|dicabut|ditambahkan|ketentuan)\b', line, re.IGNORECASE))


6) utils/text_cleaner.py — pembersih teks modular

Tujuan: pisahkan step cleaning (mudah di‑profiling & toggle), tidak lagi “god-method”.

# apps/legal-rag/backend/src/utils/text_cleaner.py
import re
from typing import Callable, List

class TextCleaner:
    def __init__(self):
        self.steps: List[Callable[[str], str]] = [
            self._normalize_ws,
            self._drop_headers_footers,
            self._drop_watermarks,
            self._fix_inline_numbering,
            self._squash_triple_newlines,
        ]

    def clean_legal_document_comprehensive(self, text: str) -> str:
        for step in self.steps:
            text = step(text)
        return text.strip()

    def _normalize_ws(self, t: str) -> str:
        return re.sub(r"[ \t]+", " ", t)

    def _drop_headers_footers(self, t: str) -> str:
        # Buang pola "www. djpp. kemenkumham. go. id" & halaman
        t = re.sub(r"www\.\s*djpp\.\s*kemenkumham\.\s*go\.\s*id", "", t, flags=re.IGNORECASE)
        t = re.sub(r"^\s*Page\s+\d+\s*$", "", t, flags=re.MULTILINE)
        return t

    def _drop_watermarks(self, t: str) -> str:
        return re.sub(r"REPUBLIK\s+INDONESIA.*?$", "", t, flags=re.IGNORECASE|re.MULTILINE)

    def _fix_inline_numbering(self, t: str) -> str:
        # Satukan "1. a. ..." yang kepotong line menjadi satu baris rapi jika perlu
        t = re.sub(r"\n\s+([a-z]\.\s)", r" \1", t)
        return t

    def _squash_triple_newlines(self, t: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", t)

  7) service/pdf/extractor.py — ekstraksi PDF (streaming, context manager)

  Tujuan: IO aman (context manager), streaming ke disk (tidak tahan di memory), error handling jelas.

  # apps/legal-rag/backend/src/service/pdf/extractor.py
  from dataclasses import dataclass
  from typing import Optional
  import time
  import pathlib

  @dataclass
  class ExtractionResult:
      success: bool
      text: Optional[str]
      method: str
      confidence: float
      processing_time: float
      page_count: int

  class UnifiedPDFExtractor:
      def extract_text(self, file_path: str) -> ExtractionResult:
          start = time.time()
          # Contoh minimal: pakai pdfminer.six / pymupdf (pilih sesuai preferensi)
          try:
              import fitz  # PyMuPDF
              p = pathlib.Path(file_path)
              if not p.exists():
                  return ExtractionResult(False, None, "pymupdf", 0.0, 0.0, 0)
              doc = fitz.open(file_path)
              pages = []
              for page in doc:
                  pages.append(page.get_text("text"))
              text = "\n".join(pages)
              return ExtractionResult(True, text, "pymupdf", 0.98, time.time()-start, len(doc))
          except Exception:
              return ExtractionResult(False, None, "pymupdf", 0.0, time.time()-start, 0)



⸻

8) service/pdf/orchestrator.py — parser legal tree (refactor menyeluruh)

Tujuan: strukturisasi → node konsisten → pasal.content digabung lengkap → leaf (ayat/huruf/angka) punya bm25_body. (Ini versi disederhanakan tapi lengkap alur; kamu sudah punya versi panjang—di sini sudah selaras dengan pattern manager & text_cleaner.)

# apps/legal-rag/backend/src/service/pdf/orchestrator.py
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.service.pdf.extractor import UnifiedPDFExtractor, ExtractionResult
from src.utils.text_cleaner import TextCleaner
from src.utils.pattern_manager import get_hierarchy_patterns, is_amendment_line

@dataclass
class LegalNode:
    type: str
    number: str
    title: str = ""
    content: str = ""
    children: List["LegalNode"] = field(default_factory=list)
    level: float = 0.0

class PDFOrchestrator:
    def __init__(self):
        self.extractor = UnifiedPDFExtractor()
        self.cleaner = TextCleaner()
        self.patterns = get_hierarchy_patterns()

    def process_pdf(self, doc_meta: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(doc_meta)
        pdf_path = data.get("pdf_path")
        if not pdf_path:
            data["doc_processing_status"] = "no_pdf"
            return data

        res: ExtractionResult = self.extractor.extract_text(pdf_path)
        if not res.success or not res.text:
            data["doc_processing_status"] = "pdf_extract_failed"
            return data

        text = self.cleaner.clean_legal_document_comprehensive(res.text)
        root = self._build_tree(text)
        self._aggregate_pasal_content(root)

        data["doc_content"] = text
        data["document_tree"] = self._serialize(root, doc_meta)
        data["doc_processing_status"] = "pdf_processed"
        data["pdf_extraction_metadata"] = {
            "method": res.method, "confidence": res.confidence,
            "processing_time": res.processing_time, "page_count": res.page_count,
        }
        return data

    # --- Tree builder -------------------------------------------------
    def _build_tree(self, text: str) -> LegalNode:
        root = LegalNode("document", "root", title="Document Root", level=0)
        stack = [root]
        in_amendment = False

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue

            if is_amendment_line(line):
                in_amendment = True

            matched = self._match_any(line, in_amendment)
            if matched:
                # pop sampai level < matched.level
                while len(stack) > 1 and stack[-1].level >= matched.level:
                    stack.pop()
                stack[-1].children.append(matched)
                stack.append(matched)
            else:
                # akumulasi konten
                node = stack[-1]
                if node.content:
                    node.content += "\n" + line
                else:
                    node.content = line
        return root

    def _match_any(self, line: str, in_amendment: bool) -> Optional[LegalNode]:
        order = ["buku","bab","bagian","paragraf","pasal","ayat","angka","huruf"]
        for name in order:
            pat, lvl = self.patterns[name]
            m = pat.match(line)
            if not m:
                continue
            num = m.group(1).strip()
            content_after = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""

            # angka/huruf bisa inline
            title = line
            node = LegalNode(name, num, title=title, level=lvl)
            if name in ("huruf","angka") and content_after:
                node.content = content_after
            if name == "ayat" and m.lastindex and m.lastindex >= 2 and content_after:
                node.content = content_after
            return node
        return None

    def _aggregate_pasal_content(self, node: LegalNode):
        if node.type == "pasal":
            parts: List[str] = []
            if node.content:
                parts.append(node.content)

            def walk(n: LegalNode, indent=""):
                out = []
                if n.type == "ayat":
                    s = f"({n.number})"
                    body = n.content or n.title.replace(s, "").strip()
                    if body:
                        out.append(f"{indent}{s} {body}")
                elif n.type in ("huruf","angka"):
                    s = f"{n.number}."
                    body = n.content or n.title.split(f"{n.number}.",1)[-1].strip()
                    if body:
                        out.append(f"{indent}{s} {body}")
                for ch in n.children:
                    out.extend(walk(ch, indent))
                return out

            for ch in node.children:
                parts.extend(walk(ch))
            node.content = "\n".join([p for p in parts if p.strip()])

        for ch in node.children:
            self._aggregate_pasal_content(ch)

    # --- Serializer ---------------------------------------------------
    def _serialize(self, node: LegalNode, meta: Dict[str, Any], parent_unit_id=None, path=None, pasal_id=None) -> Dict[str, Any]:
        doc_id = meta.get("doc_id","document")
        doc_title = meta.get("doc_title") or meta.get("title") or "Dokumen"
        if path is None:
            path = [{"type": "dokumen", "label": doc_title, "unit_id": doc_id}]

        if node.type == "document":
            return {
                "doc_type": "document",
                "doc_unit_id": doc_id,
                "doc_title": doc_title,
                "children": [self._serialize(ch, meta, doc_id, path, None) for ch in node.children]
            }

        parent_unit_id = parent_unit_id or doc_id
        unit_id = f"{parent_unit_id}/{node.type}-{node.number}"
        label_display = self._label(node.type, node.number)
        ord_int, ord_suf = self._ord(node.number)
        seq_key = f"{ord_int:04d}|{ord_suf}"

        cur = {"type": node.type, "label": label_display, "unit_id": unit_id}
        cur_path = path + [cur]
        citation = self._citation(meta, cur_path)
        hierarchy = " / ".join([p["label"] for p in cur_path])

        data: Dict[str, Any] = {
            "type": node.type,
            "unit_id": unit_id,
            "number_label": node.number,
            "ordinal_int": ord_int,
            "ordinal_suffix": ord_suf,
            "label_display": label_display,
            "seq_sort_key": seq_key,
            "citation_string": citation,
            "path": cur_path,
        }

        if node.type in ("bab","bagian","paragraf","pasal"):
            data["title"] = node.title

        if node.type == "pasal":
            data["content"] = node.content
            data["tags_semantik"] = []
            data["entities"] = []
            data["hierarchy_path"] = hierarchy

        if node.type in ("ayat","huruf","angka"):
            local = node.content or node.title.replace(label_display, "").strip()
            data.update({
                "parent_pasal_id": pasal_id,
                "local_content": local,
                "display_text": f"{label_display} {local}".strip(),
                "bm25_body": local,
                "span": None,
                "hierarchy_path": hierarchy
            })

        child_pasal_id = pasal_id
        if node.type == "pasal":
            child_pasal_id = unit_id

        data["children"] = [self._serialize(ch, meta, unit_id, cur_path, child_pasal_id) for ch in node.children]
        return data

    def _label(self, t: str, n: str) -> str:
        return {
            "pasal": f"Pasal {n}",
            "bab": f"BAB {n}",
            "bagian": f"BAGIAN {n}",
            "paragraf": f"PARAGRAF {n}",
            "ayat": f"({n})",
            "huruf": f"{n}.",
            "angka": f"{n}.",
        }.get(t, n)

    def _ord(self, label: str):
        # "1A" => 1, "A"; "IV" => 4, ""; "a" => pos alfabet
        if re.fullmatch(r"[IVXLCDM]+", label):
            return self._roman_to_int(label), ""
        m = re.match(r"(\d+)([A-Za-z]*)", label)
        if m:
            return int(m.group(1)), m.group(2) or ""
        if len(label) == 1 and label.isalpha():
            return ord(label.lower())-96, ""
        return 0, ""

    def _roman_to_int(self, s: str) -> int:
        vals = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
        res, prev = 0, 0
        for ch in reversed(s.upper()):
            v = vals.get(ch,0)
            res = res - v if v < prev else res + v
            prev = v
        return res

    def _citation(self, meta: Dict[str, Any], path: List[Dict[str,str]]) -> str:
        parts = []
        for p in path[1:]:
            t = p["type"]; L = p["label"]
            if t == "pasal": parts.append(L)
            elif t == "ayat": parts.append(f"ayat {L}")
            elif t == "huruf": parts.append(f"huruf {L.strip('.')}")
            elif t == "angka": parts.append(f"angka {re.sub(r'[^0-9]','',L)}")
        base = f"{meta.get('doc_title') or meta.get('title')}"
        return f"{base}" + (", " + " ".join(parts) if parts else "")


9) pipeline/indexer.py — ingest JSON → DB + FTS + Embedding

Tujuan: membaca JSON hasil ekstraksi (atau crawler), simpan legal_documents, simpan semua legal_units, ambil pasal untuk embedding Jina v4 (batch), simpan ke document_vectors.
# apps/legal-rag/backend/src/pipeline/indexer.py
from typing import Dict, Any, List, Iterable
from sqlalchemy.orm import Session
from src.db.session import SessionLocal
from src.db import models as m
from src.services.embedding.jina_embedder import JinaEmbedder
from src.config.settings import settings
import uuid

def upsert_document(db: Session, doc: Dict[str,Any]) -> m.LegalDocument:
    # mapping field -> model
    form = (doc.get("doc_form") or doc.get("form") or "UU").upper()
    ld = db.query(m.LegalDocument).filter(
        m.LegalDocument.form==m.DocForm[form if form in m.DocForm.__members__ else "UU"],
        m.LegalDocument.number==str(doc.get("doc_number")),
        m.LegalDocument.year==int(doc.get("doc_year")),
    ).one_or_none()

    if not ld:
        ld = m.LegalDocument(
            id=uuid.uuid4(),
            source=doc.get("doc_source","BPK"),
            type=doc.get("doc_type","Peraturan Perundang-undangan"),
            title=doc.get("doc_title") or doc.get("title",""),
            number=str(doc.get("doc_number")),
            form=m.DocForm[form if form in m.DocForm.__members__ else "UU"],
            form_short=doc.get("doc_form_short","UU"),
            year=int(doc.get("doc_year")),
            teu=doc.get("doc_teu"),
            place_enacted=doc.get("doc_place_enacted"),
            language=doc.get("doc_language","Bahasa Indonesia"),
            location=doc.get("doc_location"),
            field=doc.get("doc_field"),
            date_enacted=_date(doc.get("doc_date_enacted")),
            date_promulgated=_date(doc.get("doc_date_promulgated")),
            date_effective=_date(doc.get("doc_date_effective")),
            status=m.DocStatus.BERLAKU,
            detail_url=doc.get("detail_url"),
            source_url=doc.get("source_url"),
            pdf_url=doc.get("pdf_url"),
            uji_materi_pdf_url=doc.get("uji_materi_pdf_url"),
            pdf_path=doc.get("pdf_path"),
            text_path=doc.get("text_path"),
            content=None,
            processing_status=doc.get("doc_processing_status","pending")
        )
        db.add(ld)
    else:
        # update ringan
        ld.title = doc.get("doc_title") or ld.title
        ld.processing_status = doc.get("doc_processing_status") or ld.processing_status

    # subjects
    subs = doc.get("doc_subject") or []
    if subs:
        ensure_subjects(db, ld, subs)

    db.flush()
    return ld

def ensure_subjects(db: Session, ld: m.LegalDocument, subjects: List[str]):
    seen = {s.name for s in ld.subjects}
    for name in subjects:
        if name in seen: continue
        s = db.query(m.Subject).filter(m.Subject.name==name).one_or_none()
        if not s:
            s = m.Subject(name=name)
            db.add(s); db.flush()
        ld.subjects.append(s)

def flatten_tree_and_store(db: Session, ld: m.LegalDocument, tree: Dict[str,Any]) -> List[Dict[str,Any]]:
    """
    Simpan semua node ke legal_units.
    Return daftar pasal untuk embedding: [{'unit_id':..., 'content':..., 'pasal_number':..., 'bab':...}, ...]
    """
    pasals: List[Dict[str,Any]] = []

    def walk(node: Dict[str,Any]):
        t = node.get("type")
        if t == "document":
            for ch in node.get("children", []): walk(ch)
            return

        unit = m.LegalUnit(
            document_id = ld.id,
            unit_type = m.UnitType[t.upper()],
            unit_id = node["unit_id"],
            number_label = node.get("number_label"),
            ordinal_int = node.get("ordinal_int") or 0,
            ordinal_suffix = node.get("ordinal_suffix") or "",
            label_display = node.get("label_display"),
            seq_sort_key = node.get("seq_sort_key"),
            title = node.get("title"),
            content = node.get("content"),
            local_content = node.get("local_content"),
            display_text = node.get("display_text"),
            bm25_body = node.get("bm25_body"),
            path = node.get("path"),
            citation_string = node.get("citation_string"),
            parent_pasal_id = node.get("parent_pasal_id"),
            hierarchy_path = node.get("hierarchy_path"),
        )
        db.add(unit)

        if t == "pasal" and (unit.content and unit.content.strip()):
            pasal_number = node.get("number_label")
            # coba cari info bab dari path
            bab = None
            for p in node.get("path", []):
                if p["type"] == "bab":
                    bab = p["label"].replace("BAB ","")
            pasals.append({
                "unit_id": unit.unit_id,
                "content": unit.content,
                "pasal_number": pasal_number,
                "bab": bab
            })

        for ch in node.get("children", []):
            walk(ch)

    walk(tree)
    db.flush()
    return pasals

def embed_and_store(db: Session, ld: m.LegalDocument, pasals: List[Dict[str,Any]]):
    if not pasals:
        return
    embedder = JinaEmbedder()
    texts = [p["content"] for p in pasals]
    embs = embedder.embed_texts(texts)  # List[List[float]]

    for p, vec in zip(pasals, embs):
        dv = m.DocumentVector(
            document_id = ld.id,
            unit_id = p["unit_id"],
            content_type = "pasal",
            embedding = vec,
            embedding_model = "jina-embeddings-v4",
            doc_form = ld.form,
            doc_year = ld.year,
            doc_number = ld.number,
            doc_status = ld.status,
            bab_number = p.get("bab"),
            pasal_number = p.get("pasal_number"),
            hierarchy_path = p["unit_id"],
            token_count = len(p["content"].split()),
            char_count = len(p["content"]),
        )
        db.add(dv)
    db.flush()

def _date(s):
    if not s: return None
    try:
        from datetime import date
        return date.fromisoformat(str(s))
    except Exception:
        return None

# --- CLI entry (untuk batch index folder json) ---------------------
def ingest_document_json(doc_json: Dict[str,Any]):
    with SessionLocal() as db:
        ld = upsert_document(db, doc_json)
        tree = doc_json.get("document_tree")
        if tree:
            pasals = flatten_tree_and_store(db, ld, tree)
            embed_and_store(db, ld, pasals)
        # relationships
        for cat, arr in (doc_json.get("relationships") or {}).items():
            for rel in arr:
                db.add(m.LegalRelationship(
                    from_document_id=ld.id,
                    to_document_ref=rel.get("regulation_reference"),
                    relation_type=cat,
                    reference_link=rel.get("reference_link")
                ))
        # uji materi
        for d in (doc_json.get("uji_materi") or []):
            db.add(m.UjiMateriDecision(
                document_id=ld.id,
                decision_number=d.get("decision_number"),
                pdf_url=d.get("pdf_url"),
                decision_content=d.get("decision_content"),
                decision_type=d.get("decision_type"),
                legal_basis=d.get("legal_basis"),
                binding_status=d.get("binding_status"),
                conditions=d.get("conditions"),
                interpretation=d.get("interpretation"),
                pasal_affected=d.get("pasal_affected"),
                ayat_affected=d.get("ayat_affected"),
                huruf_affected=d.get("huruf_affected"),
            ))
        db.commit()

10) services/embedding/jina_embedder.py — Jina v4 (batch, retry)

# apps/legal-rag/backend/src/services/embedding/jina_embedder.py (768Dimension)
import hashlib, json, time
from typing import List
import httpx
from src.config.settings import settings

class JinaEmbedder:
    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=60)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {settings.jina_api_key}"}
        out: List[List[float]] = []
        B = settings.embed_batch_size
        for i in range(0, len(texts), B):
            batch = texts[i:i+B]
            payload = {"input": batch, "model": settings.jina_embed_model}
            for attempt in range(5):
                try:
                    r = self.client.post(settings.jina_embed_base, headers=headers, json=payload)
                    r.raise_for_status()
                    data = r.json()
                    vecs = [item["embedding"] for item in data["data"]]
                    out.extend(vecs)
                    break
                except Exception:
                    time.sleep(0.5 * (attempt+1))
                    if attempt == 4:
                        raise
        return out

    @staticmethod
    def hash_vector(vec: List[float]) -> str:
        return hashlib.sha256(json.dumps(vec).encode()).hexdigest()

11) services/rerank/* — opsional Jina Reranker

base.py:
# apps/legal-rag/backend/src/services/rerank/base.py
from typing import List, Dict, Any, Protocol

class Reranker(Protocol):
    def rerank(self, query: str, candidates: List[Dict[str,Any]], k: int) -> List[Dict[str,Any]]:
        ...


jina_reranker.py:
# apps/legal-rag/backend/src/services/rerank/jina_reranker.py
from typing import List, Dict, Any
import httpx
from src.config.settings import settings

class JinaReranker:
    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=60)

    def rerank(self, query: str, candidates: List[Dict[str,Any]], k: int) -> List[Dict[str,Any]]:
        if not candidates: return candidates
        headers = {"Authorization": f"Bearer {settings.jina_api_key}"}
        payload = {
            "model": settings.jina_rerank_model,
            "query": query,
            "documents": [c["text"] for c in candidates]
        }
        r = self.client.post(settings.jina_rerank_base, headers=headers, json=payload)
        r.raise_for_status()
        scored = r.json()["data"]
        # map score back
        for idx, s in enumerate(scored):
            candidates[idx]["score"] = s.get("relevance_score", candidates[idx].get("score",0))
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:k]

12) services/llm/* — provider pluggable (Gemini default)

base.py:

# apps/legal-rag/backend/src/services/llm/base.py
from typing import Protocol, List, Dict

class LLM(Protocol):
    def complete(self, system_prompt: str, user_prompt: str, stream: bool = False):
        ...

factory.py:
# apps/legal-rag/backend/src/services/llm/factory.py
from src.config.settings import settings
from src.services.llm.providers.gemini import GeminiLLM
from src.services.llm.providers.openai_ import OpenAILLM
from src.services.llm.providers.anthropic_ import AnthropicLLM

def get_llm():
    p = settings.llm_provider.lower()
    if p == "openai":
        return OpenAILLM()
    if p == "anthropic":
        return AnthropicLLM()
    return GeminiLLM()

providers/gemini.py (sketsa; implement SSE optional):
# apps/legal-rag/backend/src/services/llm/providers/gemini.py
import httpx, json
from src.config.settings import settings

class GeminiLLM:
    def __init__(self):
        self.key = settings.gemini_api_key
        self.base = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
        self.client = httpx.Client(timeout=120)

    def complete(self, system_prompt: str, user_prompt: str, stream: bool=False):
        # sederhana non-stream
        headers = {"Content-Type": "application/json"}
        params = {"key": self.key}
        payload = {
            "contents": [
                {"role":"user","parts":[{"text": f"{system_prompt}\n\n{user_prompt}"}]}
            ],
            "generationConfig":{"temperature":0.2,"topK":40,"topP":0.95}
        }
        r = self.client.post(self.base, params=params, headers=headers, json=payload)
        r.raise_for_status()
        out = r.json()
        text = out["candidates"][0]["content"]["parts"][0]["text"]
        return {"text": text}
  OpenAI/Anthropic file serupa dengan endpoint masing‑masing (kita siapkan skeleton, tinggal isi API call).


  13) retriever/hybrid_retriever.py — Hybrid (FTS + Vector + optional Rerank)

  Tujuan:
	•	Eksplisit (ada “Pasal … ayat (..)” dsb) → FTS-first (tepat sasaran).
	•	Semantik → vektor (pasal) + FTS recall leaf → opsional reranker.
	# apps/legal-rag/backend/src/retriever/hybrid_retriever.py
import re
from typing import List, Dict, Any, Tuple
from sqlalchemy import text
from sqlalchemy.orm import Session
from src.db import models as m
from src.services.embedding.jina_embedder import JinaEmbedder
from src.config.settings import settings
from src.services.rerank.jina_reranker import JinaReranker

EXPLICIT_RX = re.compile(
    r"(uu|pp|perpres|permen|perda)\s+(\d+[a-zA-Z]*)\s+tahun\s+(\d{4}).*?(pasal\s+\d+[A-Z]*)?(?:.*?ayat\s*\((\d+)\))?(?:.*?huruf\s*([a-z]))?(?:.*?angka\s*(\d+))?",
    re.IGNORECASE
)

def parse_explicit(q: str):
    m = EXPLICIT_RX.search(q)
    if not m: return None
    g = m.groups()
    return {
        "form": g[0].upper(),
        "number": g[1],
        "year": g[2],
        "pasal": (g[3].split()[-1] if g[3] else None),
        "ayat": g[4],
        "huruf": g[5],
        "angka": g[6],
    }

def fts_units(db: Session, query: str, limit=20) -> List[Dict[str,Any]]:
    sql = text("""
        SELECT id, document_id, unit_id, citation_string, bm25_body
        FROM legal_units
        WHERE content_vector @@ plainto_tsquery('indonesian', :q)
        ORDER BY ts_rank_cd(content_vector, plainto_tsquery('indonesian', :q)) DESC
        LIMIT :lim
    """)
    rows = db.execute(sql, {"q": query, "lim": limit}).mappings().all()
    return [dict(r) for r in rows]

def vector_by_query(db: Session, query: str, limit=20) -> List[Dict[str,Any]]:
    emb = JinaEmbedder().embed_texts([query])[0]
    sql = text("""
        SELECT id, document_id, unit_id, pasal_number, hierarchy_path,
               1 - (embedding <=> :qv) AS score
        FROM document_vectors
        ORDER BY embedding <=> :qv
        LIMIT :lim
    """)
    rows = db.execute(sql, {"qv": emb, "lim": limit}).mappings().all()
    return [dict(r) for r in rows]

def explicit_lookup(db: Session, meta: Dict[str,str], limit=10) -> List[Dict[str,Any]]:
    # filter by doc & structure
    params = {
        "form": meta["form"],
        "number": meta["number"],
        "year": int(meta["year"]),
        "limit": limit
    }
    filters = ["d.form = :form::docform", "d.number = :number", "d.year = :year"]
    if meta.get("pasal"):
        filters.append("u.unit_id LIKE CONCAT('%/pasal-', :pasal, '%')")
        params["pasal"] = meta["pasal"]
    if meta.get("ayat"):
        filters.append("u.unit_id LIKE CONCAT('%/ayat-', :ayat, '%')")
        params["ayat"] = meta["ayat"]
    if meta.get("huruf"):
        filters.append("u.unit_id LIKE CONCAT('%/huruf-', :huruf, '%')")
        params["huruf"] = meta["huruf"]
    if meta.get("angka"):
        filters.append("u.unit_id LIKE CONCAT('%/angka-', :angka, '%')")
        params["angka"] = meta["angka"]

    sql = text(f"""
        SELECT u.id, u.document_id, u.unit_id, u.citation_string, u.bm25_body
        FROM legal_units u
        JOIN legal_documents d ON d.id = u.document_id
        WHERE {" AND ".join(filters)}
        ORDER BY u.seq_sort_key
        LIMIT :limit
    """)
    rows = db.execute(sql, params).mappings().all()
    return [dict(r) for r in rows]

def hybrid_search(db: Session, query: str, topk=12, use_rerank=True) -> List[Dict[str,Any]]:
    exp = parse_explicit(query)
    if exp:
        leafs = explicit_lookup(db, exp, limit=topk)
        # ambil pasal konteks via unit_id prefix sampai pasal
        return leafs

    # semantik
    vec_pasals = vector_by_query(db, query, limit=topk)
    fts_leafs = fts_units(db, query, limit=topk)

    # gabung kandidat: leafs (granular) + pasal (konteks)
    cands: List[Dict[str,Any]] = []
    for l in fts_leafs:
        cands.append({"type":"leaf","text":l["bm25_body"],"unit_id":l["unit_id"],"citation":l["citation_string"],"score":0.8})
    for p in vec_pasals:
        # fetch pasal content ringkas (opsional)
        cands.append({"type":"pasal","text":p["hierarchy_path"],"unit_id":p["unit_id"],"citation":p["hierarchy_path"],"score":p["score"]})

    if settings.rerank_provider == "jina" and use_rerank and cands:
        return JinaReranker().rerank(query, cands, k=topk)
    return cands[:topk]

  14) prompts/system.py & prompts/answer_with_citations.py

  System: gaya jawaban, aturan kutip.
  # apps/legal-rag/backend/src/prompts/system.py
  SYSTEM_PROMPT = """Kamu adalah asisten hukum Indonesia yang presisi.
  - Jika user meminta isi pasal/ayat/huruf/angka: kutip verbatim bagian diminta dan sertakan [citation].
  - Jika tematik: rangkum singkat, sebutkan pasal relevan beserta [citation].
  - Jangan mengarang nomor pasal. Jika tidak yakin, katakan tidak yakin dan tampilkan pasal terdekat.
  """

  Template user:
  # apps/legal-rag/backend/src/prompts/answer_with_citations.py
  def build_user_prompt(query: str, contexts: list[dict]) -> str:
      blocks = []
      for i, c in enumerate(contexts, 1):
          blocks.append(f"[{i}] {c['citation']}\n{c['text']}\n")
      ctx = "\n".join(blocks)
      return f"Pertanyaan: {query}\n\nKonteks:\n{ctx}\n\nJawab dengan menyertakan [citation] merujuk nomor blok."


  15) orchestrator/qa_orchestrator.py — rakit konteks & panggil LLM

  # apps/legal-rag/backend/src/orchestrator/qa_orchestrator.py
  from typing import List, Dict, Any
  from sqlalchemy.orm import Session
  from src.prompts.system import SYSTEM_PROMPT
  from src.prompts.answer_with_citations import build_user_prompt
  from src.services.llm.factory import get_llm
  from src.retriever.hybrid_retriever import hybrid_search

  def answer(db: Session, query: str, max_ctx_chars=12000) -> Dict[str,Any]:
      cands = hybrid_search(db, query, topk=12, use_rerank=True)
      ctx = []
      total = 0
      for c in cands:
          t = c["text"] or ""
          if not t: continue
          if total + len(t) > max_ctx_chars: break
          ctx.append({"citation": c["citation"], "text": t})
          total += len(t)

      llm = get_llm()
      up = build_user_prompt(query, ctx)
      out = llm.complete(SYSTEM_PROMPT, up, stream=False)
      return {"answer": out["text"], "candidates": cands[:len(ctx)]}

16) api/app.py — FastAPI + endpoint /ask & /index/document

# apps/legal-rag/backend/src/api/app.py
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from src.db.session import SessionLocal
from src.pipeline.indexer import ingest_document_json
from src.orchestrator.qa_orchestrator import answer

app = FastAPI(title="Legal RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_headers=["*"], allow_methods=["*"]
)

@app.post("/ask")
def ask(payload: dict = Body(...)):
    q = payload.get("query","").strip()
    if not q: return {"answer":"", "candidates":[]}
    with SessionLocal() as db:
        return answer(db, q)

@app.post("/index/document")
def index_document(doc: dict = Body(...)):
    ingest_document_json(doc)
    return {"status":"ok"}


Frontend (Next.js 14 App Router)

1) app/chat/page.tsx — halaman chat

Ringkas tapi fungsional: input, daftar pesan, render sumber.
// apps/legal-rag/frontend/app/chat/page.tsx
"use client";
import { useState } from "react";
import { ask } from "./actions";
import MessageList from "@/components/MessageList";
import ChatInput from "@/components/ChatInput";

export default function ChatPage() {
  const [messages, setMessages] = useState<any[]>([]);

  async function onSend(text: string) {
    const userMsg = { role: "user", content: text };
    setMessages((m) => [...m, userMsg]);
    const res = await ask(text);
    const assistantMsg = { role: "assistant", content: res.answer, sources: res.candidates };
    setMessages((m) => [...m, assistantMsg]);
  }

  return (
    <div className="mx-auto max-w-3xl p-4">
      <h1 className="text-2xl font-semibold mb-4">Legal RAG</h1>
      <MessageList messages={messages} />
      <ChatInput onSend={onSend} />
    </div>
  );
}



2) app/chat/actions.ts — call API backend
// apps/legal-rag/frontend/app/chat/actions.ts
"use server";

export async function ask(query: string) {
  const r = await fetch(process.env.BACKEND_URL + "/ask", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({query})
  });
  if (!r.ok) throw new Error("API error");
  return r.json();
}

3) components/MessageList.tsx, MessageBubble.tsx, SourceCard.tsx, ChatInput.tsx

Tiap komponen kecil & simple (render markdown, daftar sumber dengan citation).

⸻

Kontrak JSON hasil ekstraksi (yang saya punya sekarang)

Tetap dipakai, tapi pastikan:
	•	Root document_tree → children berurutan, tidak ada huruf di root kecuali konsiderans (boleh, tapi type‑nya huruf di bawah node konsiderans).
	•	Setiap PASAL punya content lengkap (hasil penggabungan ayat/huruf/angka).
	•	Leaf (ayat/huruf/angka) punya bm25_body, display_text, parent_pasal_id, path, citation_string.

Kalau node seperti “Pasal 1A”, parser kita sudah mendukung (\d+[A-Z]? → “1A”).
