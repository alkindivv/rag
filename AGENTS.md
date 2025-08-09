## Legal RAG Agents Spec (Codex‑friendly)

Tujuan umum:
- Akurat & cepat: hybrid retrieval (FTS leaf + vector pasal + rerank).
- Sederhana & maintainable: modul <300 LOC/file, tipe jelas, logging terstruktur.
- Production‑ready: Alembic migrations, retry/backoff, DI untuk HTTP client.

---

# Code Review and Structural Suggestions


### 1) Agent: Craweler/pdf refactor
READ 

## Ingestion Pipeline (`src/ingestion.py`)
- Separate command-line parsing from business logic for easier testing.
- Consider converting sequential PDF processing to asynchronous tasks to utilise the existing event loop.
- Use structured logging instead of print statements for consistent output and easier log aggregation.
- Wrap file I/O with context managers and handle potential JSON decoding errors explicitly.

## Crawler Services (`src/services/crawler/*`)
- Implement retry/backoff utilities as shared helpers rather than inline loops for clearer flow.
- Use dependency injection for the HTTP client session to improve testability.
- Break large methods such as `_extract_uji_materi_decisions` into smaller units and add type hints for intermediate structures.

## PDF Processing (`src/pdf/*` and `src/services/pdf/*`)
- Merge duplicate orchestrator modules and keep a single entry point for PDF handling.
- Move hierarchy pattern definitions to a configuration or data file to reduce clutter in `PDFOrchestrator`.
- Streaming downloads could write directly to disk instead of holding all chunks in memory.
- Introduce unit tests for tree-building logic to ensure Indonesian legal hierarchy is parsed correctly.

## Text Cleaning (`src/utils/text_cleaner.py`)
- The monolithic `clean_legal_document_comprehensive` performs many responsibilities; break it into a pipeline of composable steps.
- Maintain a registry of cleaning operations with clear names and allow enabling/disabling steps via configuration.
- Add profiling or benchmarking hooks to understand the cost of each cleaning stage.


- Tanggung jawab
  - `PDFOrchestrator`: bangun `document_tree` (dokumen→…→pasal→ayat/huruf/angka).
  - `text_cleaner.py`: normalisasi whitespace, hapus watermark/header/footer.
  - `pattern_manager.py`: regex hierarki (buku, bab, bagian, paragraf, pasal, ayat, huruf, angka, sisipan/amandemen pasal).

- Input
  - PDF/teks mentah.

- Output/Deliverables
  - `doc_content` bersih dan `document_tree` lengkap.
  - embedding pasal di `document_vectors`. yang berisi pasal lengkap beserta ayat/huruf/angka/amandemen angka
  - Leaf unit berisi: `bm25_body`, `citation_string`, `parent_pasal_id`, `path`.

- Acceptance criteria
  - Setiap pasal memiliki konten lengkap (gabungan ayat/huruf/angka).
  - Node sisipan (mis. Pasal 1A) terdeteksi dan tersusun benar.


### 2) Agent: Indexer/Embedding

- Tanggung jawab
  - Upsert `legal_documents` dari metadata, simpan `document_tree` → flatten ke `legal_units`.
  - Bangun konten per pasal, embed Jina v4 dim=1024 → simpan ke `document_vectors`.
  - Simpan provenance: `doc_relationships`, `doc_uji_materi` (mentah JSON dari sumber).

- Input
  - Output Parser/Cleaner, Embedding service.

- Output/Deliverables
  - Baris pada `legal_units` (leaf dan non‑leaf) + `document_vectors` (per pasal).
  - Metadata vektor lengkap: `doc_form`, `doc_year`, `doc_number`, `doc_status`, `pasal_number`.

- Acceptance criteria
  - Batch embedding configurable; retry/backoff saat error jaringan.
  - Token/char count tercatat untuk setiap vektor.

---

### 3) Agent: DB/Schema

- Tanggung jawab
  - Tabel inti: `legal_documents`, `legal_units`, `document_vectors`, `subjects`, `document_subject`, `vector_search_logs`.
  - Tidak ada tabel normalisasi relasi/ujimateri di Postgres. Simpan mentah:
    - `legal_documents.doc_relationships JSONB`
    - `legal_documents.doc_uji_materi JSONB`
  - Enums: `doc_form`, `doc_status`, `unit_type` (lihat `plan/db.md`).
  - Index: GIN pada `legal_units.content_vector`, HNSW pada `document_vectors.embedding`.

- Input
  - JSON metadata + `document_tree` hasil parser.

- Output/Deliverables
  - Alembic migrations konsisten dengan `plan/db.md`.
  - SQLAlchemy models di `apps/legal-rag/backend/src/db/models.py`.

- Acceptance criteria
  - Unique `(doc_form, doc_number, doc_year)` berlaku.
  - `doc_form` enum: UU, PP, PERPU, PERPRES, POJK, PERMEN, PERDA, LAINNYA, SE.
  - EXPLAIN menunjukkan penggunaan GIN/HNSW sesuai tipe query.

### 4) Agent: Retriever (Hybrid)

- Tanggung jawab
  - Routing:
    - Query eksplisit → explicit lookup (doc/pasal/ayat/huruf/angka) via FTS/filter.
    - Query tematik → gabung Vector top‑K (pasal) + FTS top‑K (leaf), opsional rerank.
  - Lihat rancangan di `plan/hybrid_retrieval.md`.

- Input
  - Query teks pengguna.

- Output/Deliverables
  - Daftar kandidat dengan `id`, `text`, `meta.citation`.

- Acceptance criteria
  - Query eksplisit memukul unit yang tepat (uji regex dan lookup SQL).
  - Query tematik menghasilkan kandidat dengan `citation_string` valid.

---

### 5) Agent: Reranker

- Tanggung jawab
  - Integrasi Jina rerank (opsional) untuk re‑order kandidat.
  - Lihat `src/services/search/README_reranker.md`.

- Input/Output
  - Input: `(query, items[])` → Output: `items[]` berurut dengan skor.

- Acceptance criteria
  - Timeout & error handling benar; top‑k dihormati.

---

### 6) Agent: LLM

- Tanggung jawab
  - Provider factory (Gemini/OpenAI/Anthropic) via env.
  - System/user prompt sesuai `src/prompts/system.py`.
  - Rakit konteks dari kandidat, wajib sertakan `[citation]` berdasarkan `citation_string`.

- Acceptance criteria
  - Tidak mengarang nomor pasal; jawab "tidak yakin" bila bukti tidak cukup.

---

### 7) Agent: API

- Tanggung jawab
  - FastAPI endpoints:
    - `GET /health`
    - `GET /search?q=...` → kembalikan kandidat (tanpa LLM)
    - `POST /ask` → hybrid retrieve → rerank → panggil LLM → jawaban + sources
  - Lihat `src/services/api/README_api.md`.

- Acceptance criteria
  - Validasi input, CORS, batas ukuran konteks, logging request id.

---

### 8) Agent: Frontend

- Tanggung jawab
  - Next.js App Router: chat sederhana, tampilkan jawaban + sumber (citation).

- Acceptance criteria
  - SSR/edge friendly; UI ringan, tidak bloat.

---

### 9) Agent: Infra/Quality

- Tanggung jawab
  - Logging JSON terstruktur; DI httpx client; retry/backoff util.
  - Alembic migrations repeatable; CLI terpisah dari bisnis.
  - Unit tests: parser tree, retriever explicit routing.

---

## Kontrak I/O Ringkas (untuk Codex)

- Input Parser: `data/pdfs/*.pdf` atau `data/text/*.txt` → Output: `document_tree`, `doc_content`.
- Input Indexer: JSON metadata + `document_tree` → Output: rows pada `legal_units`, `document_vectors`.
- Retriever:
  - FTS Leaf: `SELECT ... FROM legal_units WHERE content_vector @@ plainto_tsquery('indonesian', :q)`
  - Vector Pasal: `SELECT ... FROM document_vectors ORDER BY embedding <=> :qvec LIMIT :k`
- Provenance: simpan mentah ke `legal_documents.doc_relationships` dan `legal_documents.doc_uji_materi`.
- Graph: normalisasi relasi/ujimateri dilakukan di Neo4j (ETL dari JSONB), bukan Postgres.

## Gaya Kode & Batas
- Maksimal ~300 LOC per file, pecah bila perlu.
- Type hints wajib; docstring singkat untuk public API.
- Tanpa `print`; gunakan logger terstruktur.
- I/O gunakan context manager; tangani error JSON decode.
- Konfigurasi via `pydantic.BaseSettings`.
