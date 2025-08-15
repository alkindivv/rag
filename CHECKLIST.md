# CHECKLIST: Prioritas Implementasi

Dokumen ini memandu agentic AI/kontributor berikutnya terkait urutan implementasi, feature flags, acceptance gates, dan kriteria penghapusan aman untuk refactor retrieval sistem RAG hukum Indonesia.

Sumber konteks utama: `TODO_REFACTOR_UNIT.md`.

# MASTER CHECKLIST BY COMPONENT (EXHAUSTIVE)

Catatan: Daftar ini tidak mengasumsikan ada implementasi sebelumnya. Jika modul belum ada, BUAT. Jika ada namun legacy, GANTI/REFactor sesuai panduan.

## 0) Repo Baseline & Konfigurasi
- [ ] Pastikan `.env`/settings memuat kunci API dan flag yang diperlukan (tanpa hardcode).
- [x] Tambahkan feature flags di `src/config/settings.py`: `NEW_PG_RETRIEVAL`, `USE_SQL_FUSION`, `USE_RERANKER`.
- [x] Konfigurasi DB extensions aktif via kode (bukan Alembic): `ltree`, `pg_trgm`, `unaccent`, `pgvector` (`setup_db_extras()`)

## 1) Database & Migrasi (Alembic)
- [x] Tambah kolom ke `legal_units`: `unit_path` (ltree), `parent_unit_id` (FK), `tsv_content` (tsvector), `embedding vector(384)`
- [x] Buat index: GIN(`tsv_content`), BTREE untuk `parent_unit_id` (implicit by FK)
- [ ] Buat index: GIST untuk `unit_path` (dipindah ke DDL setelah konversi ltree; perlu ditambahkan di `setup_db_extras`)
- [x] Buat extensions: `ltree`, `unaccent`, `pg_trgm`, `vector`
- [x] ANN index `embedding`: prefer `HNSW`, fallback `IVFFLAT` (partial WHERE embedding IS NOT NULL)
- [ ] (Opsional) Buat MV `pasal_fts_mv` + index jika menurunkan latensi FTS.
- [ ] Migrasi/backfill data lama: isi `unit_path` dan `parent_unit_id` untuk dokumen existing; `tsv_content` diisi via trigger.
- [x] Centralize DDL ke `src/db/models.py::setup_db_extras()`; `init_db()` memanggil fungsi ini.

## 2) Orchestrator (`src/services/pdf/pdf_orchestrator.py`)
- [ ] Emit hanya field sumber kebenaran: `content`, `local_content`, `display_text`, `citation_string`, `label_display`, `unit_path`, `parent_unit_id`.
- [x] Deprecate `bm25_body` (pemakaian dihentikan; kolom di-comment di model)
- [ ] Pastikan konsistensi normalisasi whitespace dan penanda ayat/huruf sesuai standar hukum.

## 3) Indexer (`src/pipeline/indexer.py`)
- [x] Tulis `content`/`local_content` ke DB; `tsv_content` dipelihara via trigger DB (Indonesian + unaccent)
- [x] Pastikan penulisan `unit_path` (ltree via `text2ltree`) dan `parent_unit_id` benar untuk tiap unit
- [x] Tulis embedding 384-dim langsung ke `legal_units.embedding` (granular); skip teks boilerplate (dihapus/diubah/dicabut)
- [ ] Jadikan pure consumer penuh (minim formatting) jika masih ada formatting residual

## 4) Model & Session (`src/db/models.py`, `src/db/session.py`)
- [x] Tipe kolom/ekstensi diatur via `setup_db_extras()` (termasuk trigger FTS, ANN index, enum cleanup)
- [x] `init_db --reset` aman (drop legacy `document_vectors` sebelum `drop_all()`)
- [ ] Pastikan session/engine siap untuk CTE berat dan `statement_timeout` aman.

## 5) Query Layer (BARU) `src/db/queries.py`
- [ ] Buat modul query terpusat dengan fungsi:
  - `search_explicit(params)`: ltree exact/subtree untuk pasal/ayat/huruf.
  - `search_fts(params)`: `to_tsquery` + `ts_rank_cd` terhadap `tsv_content` atau MV.
  - `search_vector(params)`: pgvector ANN dengan casting benar.
  - `search_fusion(params)`: CTE fusion explicit/fts/vector + normalisasi skor + dedup by `unit_id` + `match_type` + `explain_score`.
  - `count_siblings(unit_id)`: sibling counters (huruf/ayat dalam satu pasal), untuk pertanyaan “ada berapa huruf ...”.
- [ ] Semua fungsi mengembalikan bentuk yang bisa dipetakan ke `SearchResponse`.

## 6) Search Services
- [ ] `src/services/search/hybrid_search.py`:
  - Gunakan `queries.search_fusion` saat `USE_SQL_FUSION=true`.
  - Enforce output `SearchResponse` (schema terpadu).
  - Implement fallback ke jalur lama hanya jika flag off.
- [x] `src/services/search/vector_search.py`: gunakan `legal_units.embedding` + cast `CAST(:query_vector AS vector)`; COALESCE konten untuk hindari output kosong
- [ ] Ganti seluruh explicit lookup ke ltree predicates; jangan gunakan `parent_*` atau array `path`.
- [ ] Hapus/arahkan placeholder rerank ke `reranker.py`.
  - [ ] `src/services/search/bm25_search.py`: tetap ada sampai P2, TIDAK dipakai saat flags aktif.
  - [ ] `src/services/search/rrf_fusion.py`: tetap ada sampai P2, TIDAK dipakai saat flags aktif.

## 7) Explicit Reference Resolution (Regex Terpusat)
- [x] Buat `src/services/search/explicit/regex.py` yang menampung pola lengkap:
  - "UU X/Y Pasal N ayat (M) huruf a/b/...", rentang, daftar, kombinasi.
  - Roman numeral (Bab I–XX), pasal suffix (14A, 27B, bis, ter), angka/nomor, variasi penulisan.
  - Cross-reference detector: "sebagaimana dimaksud dalam ...".
- [x] Tambah unit test komprehensif untuk parser ini.

## 8) Schemas & Validators
- [ ] Buat `src/schemas/search.py` dan gunakan di seluruh service pencarian.
- [ ] Buat validator JSON V2 (mis. `src/validators/json_validator.py`) untuk response builder; enforce sebelum keluar API.

## 9) API Layer (`src/api/main.py`)
- [ ] `/search` dan `/ask` selalu mengembalikan `SearchResponse` valid.
- [ ] `/ask`: force hybrid retrieval sesuai flag; gunakan reranker jika `USE_RERANKER=true` dengan pinning explicit-first.
- [ ] Tangani error dengan fallback provider sesuai konfigurasi.

## 10) Reranker Policy (`src/services/search/reranker.py`)
- [ ] Pastikan ada opsi NoOp dan Jina (atau penyedia lain) dengan circuit breaker dan time budget.
- [ ] Terapkan setelah fusion; jangan mengubah urutan explicit-first (pinning).

## 11) Logging & Observability
- [x] Structured logs: strategi terpilih, waktu tiap tahap (explicit/fts/vector/fusion/rerank), hit counts, circuit breaker states.
  - [ ] Tambah tracing ID per request untuk auditability.

## 12) Testing
- [x] Unit: regex explicit parser, queries, schemas, reranker.
- [ ] Integration: DB queries (explicit/fts/vector/fusion), API `/search` dan `/ask`.
- [x] E2E: golden explicit queries dan beberapa governance queries; verifikasi `match_type`, `citation_string`, dan latensi.
- [ ] Performance smoke: ukur p95 untuk FTS dan hybrid.

## 13) Data Backfill & Normalisasi
- [ ] Backfill `unit_path` dan `parent_unit_id` untuk dokumen yang sudah ada di DB.
- [x] Pastikan `content/local_content/display_text` konsisten saat indexing baru.
- [x] `tsv_content` disinkronkan via trigger pada insert/update.

## 14) Safe Deletion (Setelah Acceptance Lulus)
- [ ] Hapus `src/services/search/bm25_search.py` jika seluruh import sudah dilepas dan jalur FTS SQL stabil.
- [ ] Hapus `src/services/search/rrf_fusion.py` jika tidak ada impor residual dan fusion 100% via CTE.
- [ ] Rapikan `vector_search.py` dari rerank placeholder (gunakan `reranker.py`).
- [ ] Jangan hapus migrasi lama; buat migrasi baru untuk drop kolom legacy bila perlu.

## 15) ADR & Dokumentasi
- [ ] Tambah ADR untuk: deprecate RRF, Postgres-only retrieval, schema respons terpadu, feature flags.
- [ ] Update `TODO_REFACTOR_UNIT.md` jika ada keputusan baru.

## 16) CI Gates
- [ ] Tambah checks: schema validation harus lulus, tidak ada impor ke file yang akan dihapus, coverage minimal untuk regex dan queries.

---

# ORDER OF OPERATIONS (EKSEKUSI BERTAHAP)
1) P0: schemas + feature flags + enforce `SearchResponse` + explicit via ltree + putus impor RRF/BM25 dari bootstrap.
2) Migrasi DB untuk `unit_path`, `parent_unit_id`, `tsv_content`, index, extensions; backfill data minimal.
3) Implement FTS dan (opsional) MV; ukur latensi.
4) Implement SQL CTE fusion dan hubungkan di `hybrid_search.py` saat `USE_SQL_FUSION=true`.
5) Aktifkan reranker (opsional) dengan pinning explicit-first; ukur dampak.
6) Tambah observability dan E2E tests; pastikan SLA.
7) Safe deletion: hapus BM25/RRF setelah verifikasi bebas impor dan tests hijau.
