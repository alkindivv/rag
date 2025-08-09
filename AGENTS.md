Tujuan Utama

Bangun sistem RAG hukum Indonesia yang:
	•	Akurat & cepat: hybrid retrieval (FTS leaf + vector pasal + optional rerank).
	•	Sederhana & maintainable: modul kecil <300 LOC/file.
	•	Production‑ready: logging terstruktur, retry/backoff, DI untuk HTTP client, alembic migrations.

Arsitektur Singkat (keputusan final)
	•	Index storage:
	•	FTS (Postgres tsvector) di leaf (ayat/huruf/angka) → presisi untuk query eksplisit.
	•	pgvector 1024‑dim (Jina v4) di pasal → semantik yang stabil & hemat latency.
	•	Hybrid retriever:
	1.	Jika query eksplisit → explicit lookup (filter by doc + pasal/ayat/huruf/angka).
	2.	Jika tematik → vector top‑K (pasal) + FTS top‑K (leaf) → (opsional) Jina reranker.
	•	LLM: default Gemini 1.5 Flash (bisa pilih OpenAI/Anthropic via env).
	•	Parser & cleaner: pipeline modular, pola hierarki di pattern_manager.py.

⸻

Peran Agent & Batasan

1) Agent: DB/Schema

Tanggung jawab
	•	Skema SQLAlchemy: LegalDocument, LegalUnit, DocumentVector, UjiMateriDecision, LegalRelationship, Subject.
	•	Indeks:
	•	content_vector tsvector (FTS) pada legal_units.bm25_body (generated column/GIN).
	•	pgvector HNSW pada document_vectors.embedding.
	•	Alembic migrations lengkap & repeatable.

Input/Output
	•	Input: kontrak JSON parsed tree.
	•	Output: tabel & indeks siap query.

Keberhasilan
	•	Migrations idempotent, query plan EXPLAIN menunjukkan penggunaan index.

Do/Don’t
	•	Do: enum types (form/status/unit_type), unique constraints, FK cascade.
	•	Don’t: menyimpan embedding pada leaf (hanya pasal).

⸻

2) Agent: Parser/Cleaner

Tanggung jawab
	•	PDFOrchestrator: bangun tree legal (dokumen→…→pasal→ayat/huruf/angka).
	•	TextCleaner: pipeline modular (normalize ws, drop watermark/header/footer, fix inline numbering).
	•	PatternManager: regex hierarki (buku, bab, bagian, paragraf, pasal, ayat, angka, huruf).

Input/Output
	•	Input: path PDF / teks mentah.
	•	Output: document_tree JSON + doc_content bersih.

Keberhasilan
	•	Setiap pasal punya content lengkap (gabungan ayat/huruf/angka).
	•	Leaf memiliki bm25_body, display_text, parent_pasal_id, path, citation_string.

Do/Don’t
	•	Do: tangani Pasal 1A dan sisipan/amandemen.
	•	Don’t: menaruh huruf di root kecuali konsiderans (kalau iya, tetap type=huruf).

⸻

3) Agent: Indexer/Embedding

Tanggung jawab
	•	Upsert dokumen + flatten tree → simpan unit.
	•	Batch embed pasal (Jina v4) → simpan DocumentVector.
	•	Simpan relationships + uji materi.

Keberhasilan
	•	Batch size configurable, retry/backoff, latency embed stabil.
	•	Metadata vektor (form/number/year/status, pasal_number) terisi.

⸻

4) Agent: Retriever (Hybrid)

Tanggung jawab
	•	Explicit lookup (regex): form/nomor/tahun/pasal/ayat/huruf/angka → ambil leaf tepat.
	•	Semantic: vector top‑K (pasal) + FTS top‑K (leaf).
	•	Rerank opsional (Jina).

Keberhasilan
	•	Query eksplisit selalu memukul unit benar.
	•	Query tematik relevan dengan citation yang valid.

⸻

5) Agent: LLM

Tanggung jawab
	•	Factory provider: Gemini/OpenAI/Anthropic.
	•	Prompting: system prompt ketat untuk kutip & citation.

Keberhasilan
	•	Tidak “mengarang” nomor pasal, sebut “tidak yakin” bila ragu.

⸻

6) Agent: API

Tanggung jawab
	•	FastAPI:
	•	POST /index/document → ingest JSON.
	•	POST /ask → hybrid retrieve → prompt LLM → jawab + candidates.

Keberhasilan
	•	Error handling rapi; CORS; batas konteks (chars).

⸻

7) Agent: Frontend

Tanggung jawab
	•	Next.js 14 (App Router) mirip ChatGPT (chat thread singkat).
	•	Tampilkan jawaban + daftar sumber (citation_string).

Keberhasilan
	•	UX ringan, tidak bloat, SSR/edge‑friendly.

⸻

8) Agent: Infra/Quality

Tanggung jawab
	•	Logging JSON (structured), DI http client, retry/backoff util.
	•	CLI terpisah dari bisnis (pars/exec).
	•	Unit tests minimal (parser tree, retriever explicit).

⸻

Gaya Kode & Batas
	•	Setiap file ≤ 300 LOC (kalau lewat, pecah).
	•	Type hints wajib, docstring singkat di public funcs.
	•	No prints → gunakan logger terstruktur.
	•	I/O gunakan context manager; tangani JSON decode error eksplisit.
	•	Konfigurasi via pydantic.BaseSettings.
