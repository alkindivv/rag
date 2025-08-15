# Refactor Plan: Orchestrator-Only Aggregation + Simplified Hierarchy (parent_unit_id + ltree)

## Objective
- Make `src/pipeline/indexer.py` a pure consumer of orchestrator output (no aggregation/formatting logic).
- Keep all content assembly and marker formatting in `src/services/pdf/pdf_orchestrator.py` only.
- Replace specialized parent anchors (`parent_pasal_id`, `parent_ayat_id`, `parent_huruf_id`) with:
  - `parent_unit_id` (immediate parent)
  - `unit_path` (Postgres ltree) for fast ancestor/descendant queries.
- Remove unused `UnitType.ANGKA_AMANDEMENT` and any logic that references it.

## Reading Guide & Execution Order
- __Langkah cepat__ (ikuti urutan ini saat implementasi):
  1) Baca "Keputusan Ringkas: Konten & Indeks" (ringkasan design final yang berlaku).
  2) Skim "Postgres-only Retrieval Blueprint" untuk gambaran besar (ltree + FTS + vector + fusion SQL).
  3) Lakukan migrasi DB di "Database Schema (Alembic)".
  4) Update Orchestrator untuk emit `unit_path`/`parent_unit_id` dan finalisasi `content/local_content/display_text`.
  5) Sederhanakan Indexer menjadi pure consumer.
  6) Update Models dan Query Layer (ganti parent_* ke ltree).
  7) Implement FTS (tsv_content) dan MV `pasal_fts_mv`.
  8) Implement Hybrid fusion (explicit-first) dan enforce `SearchResponse`.
  9) Jalankan Acceptance Criteria + Tests.

Catatan: Bagian detail tetap di bawah sesuai komponen. Checklist P0→P2 berada di akhir dokumen.

## Scope
- No shared text util module will be introduced. The orchestrator is the single source of truth for aggregation and formatting.
- Indexer consumes fields produced by orchestrator: `content`, `local_content`, `display_text`, `bm25_body`, `label_display`, `path`, `citation_string`, `parent_unit_id`, `unit_path`.
  - Catatan: `bm25_body` dinyatakan redundan dan akan dihapus (lihat bagian "Keputusan Ringkas: Konten & Indeks"). Untuk saat ini tetap tercantum agar TODO tidak hilang.

## Changes by Component

### 1) Database Schema (Alembic Migrations)
- Add columns to `legal_units`:
  - `parent_unit_id TEXT` with index.
  - `unit_path LTREE` with GIST (or GIN) index. Ensure `CREATE EXTENSION ltree`.
- Backfill `unit_path` from existing `unit_id`/`path` using a deterministic converter:
  - Lowercase; replace `-` and spaces with `_`; strip non-alnum; join with dots, e.g.: `uu_2011_24.bab_1.pasal_5.ayat_2.huruf_b`.
- Backfill `parent_unit_id` using the immediate parent `unit_id` derived from `path`.
- Update all read paths (services/queries) to use `unit_path` operations:
  - Descendants of a node: `WHERE unit_path <@ '...pasal_5'`.
  - Ancestors of a node: `WHERE unit_path @> '...ayat_2'` (if needed).
- Deprecate and later drop columns:
  - `parent_pasal_id`, `parent_ayat_id`, `parent_huruf_id`.

### 2) Orchestrator (`src/services/pdf/pdf_orchestrator.py`)
- Ensure fallback cleaning works: in `_clean_text()`, call `self._basic_clean(text)` when `TextCleaner` unavailable.
- Keep and finalize aggregation logic here only:
  - `_build_full_content_for_pasal()` remains and must assemble complete pasal content (children included) and correct markers (angka uses `"1."`).
  - Ensure all PASAL units have non-empty `content` (already patched per SUMMARY_TODO_FIX).
- During `_serialize_tree()`:
  - Set `parent_unit_id` (immediate parent `unit_id`).
  - Build and set `unit_path` (ltree-friendly) for every node.
  - Populate `content/local_content/display_text/bm25_body/label_display/path/citation_string`.

### 3) Indexer (`src/pipeline/indexer.py`)
- Remove duplicated aggregation/formatting:
  - Delete/disable `_build_pasal_aggregated_content()` and `_build_full_content()`.
- Persist exactly what orchestrator emits:
  - `content`, `local_content`, `display_text`, `bm25_body`.
  - `parent_unit_id`, `unit_path`.
- Embeddings:
  - Use `unit.content` for PASAL units directly (no rebuild).
- Clean imports and remove any references to `UnitType.ANGKA_AMANDEMENT`.

### 4) Models (`src/db/models.py`)
- Add mapped columns:
  - `parent_unit_id = Column(Text, index=True)`
  - `unit_path = Column(LtreeType or custom ltree, index=True)`
- Remove enum `ANGKA_AMANDEMENT` from `UnitType`.
- Keep existing indexes for vectors and doc metadata; add ltree indexes for `legal_units.unit_path`.

### 5) Query Layer / Services
- Replace filters based on `parent_*` with `unit_path` predicates.
  - Example: all ayat under pasal X → `unit_path <@ :pasal_path`.
- Where needed, still use `parent_unit_id` for direct parent hops.

## Rollout Plan
1. Migration 1: Add `parent_unit_id`, `unit_path`, create indexes, enable `ltree`.
2. Code change: Orchestrator populates `parent_unit_id` + `unit_path`. Indexer writes them.
3. Backfill task: compute `unit_path` and `parent_unit_id` for existing rows.
4. Switch all queries to `unit_path` (and/or `parent_unit_id`).
5. Migration 2: Drop `parent_pasal_id`, `parent_ayat_id`, `parent_huruf_id` after verification.

## Validation & Tests
- Unit tests:
  - Orchestrator builds correct `unit_path` and `parent_unit_id` for mixed hierarchies.
  - Every PASAL has non-empty `content` and correct `angka` markers ("1.").
- Integration tests:
  - Query descendants using `unit_path <@` vs previous logic → same result set.
  - Indexer persists orchestrator fields without re-aggregation; embeddings match `content`.
- E2E:
  - Hybrid search returns comparable/better results. No regression in latency.

## Risks & Mitigations
- ltree availability: ensure `CREATE EXTENSION ltree` in provisioning; document fallback (text + trigram) if absolutely necessary.
- Backfill correctness: write deterministic converter; add audit query to find rows with invalid `unit_path`.
- Query changes: wrap `unit_path` predicates in helper functions to avoid repeated string building.

## Tasks Checklist
- [ ] Alembic migration: add `parent_unit_id`, `unit_path`, indexes, enable `ltree`.
- [ ] Orchestrator: populate `parent_unit_id` and `unit_path`; ensure content aggregation finalized.
- [ ] Indexer: remove aggregation functions; consume orchestrator fields; cleanup imports.
- [ ] Models: add new columns/types; remove `ANGKA_AMANDEMENT`.
- [ ] Backfill script for historical data.
- [ ] Update query services to use `unit_path`.
- [ ] Remove old parent_* columns (follow-up migration).
- [ ] Tests: unit, integration, E2E updated and passing.

---

## Audit: Citation Parser and Ltree Integration

### Findings (Current State)
- __Parser location__: `src/services/citation/parser.py` with `CitationMatch` and `LegalCitationParser`.
- __Scope__: Regex-based extraction for UU/PP; supports pasal/ayat/huruf/angka; no ltree path emitted.
- __Gaps vs ltree plan__:
  - __No unit_path output__: Parser does not construct ltree-compatible `unit_path` or `unit_path_prefix`.
  - __Roman numerals__: Sections like `Bab I/II`, `Bagian A/B` not parsed/mapped.
  - __Pasal suffix__: Variants `14A`, `27B`, `bis/ter/quater` not normalized.
  - __Ranges__: `ayat (1)-(3)` or `huruf a–c` not supported.
  - __Cross-reference__: Phrases `sebagaimana dimaksud pada Pasal ...` not detected.
  - __Doc alias resolution__: `UU Minerba`, `UU Cipta Kerja` → number/year mapping absent.
  - __No span metadata__: Start/end offsets for highlighting not captured.
  - __Confidence routing__: OK, but not integrated with ltree output.
  - __Duplication risk__: `src/utils/citation.py` builds display strings; scope says avoid shared text utils—prefer orchestrator as SoT.

### Design Decision
- __Keep parser, make it thin__: Parser focuses on extraction + normalization; traversal and hierarchy are handled by Postgres `ltree`.
- __Single source of truth for aggregation/formatting__: Keep in `src/services/pdf/pdf_orchestrator.py` as already planned.

### Minimal Interface (Proposed)
- Function: `parse_citation(text: str) -> List[CitationMatch]`
- Extend `CitationMatch` with ltree-ready fields:
  - `unit_path_exact: Optional[str]`  e.g., `uu_2009_4.pasal_149.ayat_2.huruf_b`
  - `unit_path_prefix: Optional[str]` e.g., `uu_2009_4.pasal_149`
  - `span: Optional[Tuple[int,int]]` (optional; useful for highlighting)
- Normalized components (lowercase, underscore, alnum-only) consistent with unit_path converter in this plan.

### Normalization Rules (Authoritative)
- Document node: `uu_{year}_{number}` (or `pp_{year}_{number}`, etc.)
- Pasal: `pasal_{num}` where `num` may include suffix e.g., `14a`, `27b`, `14bis` → map to `14bis`.
- Ayat: `ayat_{n}` (digits only)
- Huruf: `huruf_{a..z}`
- Angka (sublist): `angka_{n}`
- Roman sections if present:
  - `bab_{roman_to_int}` (e.g., `Bab IV` → `bab_4`),
  - `bagian_{roman_to_int}` when applicable.

### Query Usage (ltree)
- Exact node: `WHERE unit_path = :unit_path_exact`
- Subtree: `WHERE unit_path <@ :unit_path_prefix`
- Parent/child hops: `parent_unit_id` when a direct parent is needed; otherwise prefer ltree.

### Concrete Actions
- __A1. Extend `CitationMatch`__ in `src/services/citation/parser.py` with `unit_path_exact`, `unit_path_prefix`, optional `span`.
- __A2. Build unit_path__ inside parser using deterministic converter (same rules as backfill): lowercase, replace `[-\s]`→`_`, strip non-alnum except `_`, join with dots.
- __A3. Add pattern support__:
  - Roman numerals for `Bab/Bagian` (non-blocking; optional field in path).
  - Suffix pasal (`A/B`, `bis/ter/quater`).
  - Ranges for ayat/huruf (expand to multiple matches or mark prefix at parent).
  - Cross-ref phrasing detection to bump confidence for explicit path.
- __A4. Doc alias map (optional)__: lightweight in-memory alias→(form, number, year) for common acts (Minerba, Cipta Kerja). Pluggable via config.
- __A5. Emit ltree strings only__: no DB traversal in parser. Return both `unit_path_exact` (if pasal/ayat/huruf complete) and `unit_path_prefix` (if only pasal/doc level).
- __A6. Update consumers__:
  - `VectorSearchService._lookup_citation_units`: switch to `unit_path` predicates from emitted fields.
  - `QueryOptimizationService`: use `is_explicit_citation()` as-is; no change in contract.
- __A7. Deprecate duplicate display utils__:
  - Prefer orchestrator for display strings; if `src/utils/citation.py` is used for rendering only, move or reference from orchestrator to avoid divergence.

### Test Plan Additions
- Parser unit tests: parsing matrix for UU/PP, suffixes, ranges, roman sections, cross-ref phrasings.
- Integration: explicit queries resolve to exact nodes via `unit_path` equality; subtree queries via prefix produce children (huruf/angka) correctly.

### Checklist Addendum (Parser)
- [ ] `parser.py`: add `unit_path_exact`, `unit_path_prefix`, `span` to `CitationMatch`.
- [ ] `parser.py`: implement deterministic unit_path builder.
- [ ] `parser.py`: add patterns for roman, suffix, ranges, cross-ref.
- [ ] `vector_search.py`: use `unit_path` filters from parser outputs.
- [ ] Optional: alias mapping for popular act nicknames.
- [ ] Review/remove duplicate `src/utils/citation.py` usage; centralize in orchestrator if needed.

---

## Postgres-only Retrieval Blueprint (Simpler, Accurate, Best-Practice)

### Overview
- __Goal__: Sederhanakan retrieval menjadi Postgres-only dengan tiga pilar:
  - ltree untuk explicit citation (exact dan subtree).
  - FTS yang disiplin (unaccent + phrase/proximity + cover density rank).
  - pgvector (HNSW) untuk semantic recall, digabung via SQL CTE sederhana.
- __Prinsip__: explicit-first. Fusion skor deterministik di SQL, dedup by `unit_id`, dan audit `match_type`.

### Methods Used
- __ltree__
  - Exact: `unit_path = text2ltree(:exact)`
  - Subtree: `unit_path <@ text2ltree(:prefix)`
  - Index: `CREATE INDEX legal_units_unit_path_gist ON legal_units USING GIST (unit_path);`
- __FTS__
  - Build `tsv_content` dengan `unaccent` saat konstruksi (materialized/precomputed), bukan saat query.
  - Query: `to_tsquery('indonesian', 'dana <-> perimbangan')` (phrase) atau `'<N>'` untuk proximity.
  - Rank: `ts_rank_cd(tsv_content, to_tsquery(...))` (cover density).
  - Index: `CREATE INDEX ... USING GIN (tsv_content);`
  - MV: `pasal_fts_mv` berisi agregasi satu Pasal (termasuk ayat/huruf/angka) + `tsv_content` siap GIN.
- __pgvector__
  - Index HNSW pada `embedding` kolom: `CREATE INDEX ... USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=200);`
  - Query kandidat: `ORDER BY embedding <-> :qvec LIMIT :k`
- __Fusion SQL (CTE)__
  - CTE `explicit` (ltree): skor 1.0 (exact) / 0.9 (descendant).
  - CTE `fts`: skor `rank_fts = ts_rank_cd(...)` dinormalisasi.
  - CTE `vec`: skor `rank_vec = 1 - (dist / max_dist_window)` atau z-score lokal.
  - UNION ALL → dedup (prefer explicit) → skor akhir `GREATEST(explicit_score, 0.6*rank_fts + 0.4*rank_vec)`.
  - Tambahkan kolom `match_type` dan `explain_score` ringkas untuk audit.

### Implementation Plan (per Komponen)
- __Schemas__ (`src/schemas/search.py`)
  - `SearchFilters`, `SearchResult`, `SearchResponse` (dataclass/Pydantic) untuk konsistensi respons: `{results: [...], metadata: {...}}`.
- __Citation Parser__ (`src/services/citation/parser.py`)
  - Parser tipis: emisi `unit_path_exact`, `unit_path_prefix`, `span`, `confidence` (tanpa traversal DB).
  - Normalisasi path mengikuti aturan backfill ltree di dokumen ini.
- __Hybrid Search__ (`src/services/search/hybrid_search.py`)
  - Routing explicit-first: jika parser mengembalikan `unit_path_*`, jalankan jalur explicit dan gabungkan FTS/Vector opsional.
  - Enforce satu bentuk respons (dict). Hilangkan fallback list.
- __Vector Search__ (`src/services/search/vector_search.py`)
  - Konsumsi `src/schemas/search.py`. Normalisasi skor vector untuk fusion. Panggil CTE fusion ketika perlu.
- __BM25/FTS Search__ (`src/services/search/bm25_search.py`)
  - Gunakan `to_tsquery` dengan `<->`/`<N>` dan `ts_rank_cd`; bangun `tsv_content` dengan `unaccent`.
  - Pindahkan builder SQL (query) dan mapper hasil (ke `SearchResult`) secara terpisah untuk keterbacaan.
- __DB & MV__ (migrations + DDL)
  - Pastikan `CREATE EXTENSION ltree, unaccent`.
  - GIST untuk `unit_path`, GIN untuk `tsv_content`, HNSW untuk `embedding`.
  - Buat MV `pasal_fts_mv` + index; refresh terjadwal.
- __Orchestrator__ (`src/services/pdf/pdf_orchestrator.py`)
  - Tetapkan `unit_path`, `parent_unit_id`, `citation_string`, `content` (SoT agregasi/formatting).
  - Hindari util terpisah yang dapat menyebabkan drift format; bila perlu helper khusus internal orchestrator.

### Acceptance Criteria (per Implementasi)
- __Schemas__
  - Semua service (`Hybrid`, `Vector`, `BM25/FTS`) mengembalikan `SearchResponse` identik struktur dan tipe.
  - CI gagal bila bentuk berubah (snapshot test / pydantic validation strict).
- __Citation Parser__
  - 100% test coverage untuk pola dasar: UU/PP + pasal/ayat/huruf/angka, pasal suffix (A/B/bis/ter), roman (Bab/Bagian), range (ayat/huruf), cross-ref.
  - `unit_path_exact`/`prefix` konsisten dengan hasil migrasi/backfill ltree; subtree query `unit_path <@` mengembalikan anak yang benar.
- __Hybrid Search__
  - Jika explicit ada, hasil explicit menempati top-1 dengan `match_type='explicit_*'` dan `final_score ≥ 0.95`.
  - Dedup by `unit_id` bekerja; tidak ada duplikasi antar sumber.
- __BM25/FTS__
  - Phrase queries (contoh: "dana perimbangan") → top-3 mengandung pasal relevan dengan `ts_rank_cd` tinggi.
  - Proximity queries (`<N>`) berperilaku sesuai; latensi P50 < 120 ms pada N≈1e5 entri.
- __Vector__
  - Recall kuat pada kueri deskriptif; top-100 kandidat stabil untuk fusion.
  - HNSW indeks menghasilkan P90 < 250 ms pada hybrid.
- __Fusion SQL__
  - Normalisasi skor antar sumber konsisten; dokumentasi `explain_score` tersedia.

### Deprecations & Replacements
- __Buang/Refactor__
  - `except Exception` generik di search services → ganti pengecualian spesifik + `logger.exception` dengan context (query/session_id/search_type).
  - Duplikasi builder sitasi di `src/utils/citation.py` bila tidak digunakan orchestrator → rujuk/rapikan agar tidak terjadi drift.
  - Fallback respons list di `hybrid_search.py` → wajib dict `SearchResponse`.
  - Fungsi agregasi konten di `src/pipeline/indexer.py` → hapus; konsumsi output orchestrator.
  - Pencarian parent_* → ganti `unit_path` (ltree) dan `parent_unit_id` untuk hop langsung.

### Risks & Mitigations (Tambahan)
- __FTS Bahasa Indonesia__: ketiadaan stemming resmi → gunakan config `simple` + stopwords kustom; hindari stemming agresif agar presisi hukum terjaga.
- __Index bloat__: kombinasi GIN/GIST/HNSW → indeks hanya pada kolom yang dipakai; monitor bloat dan reindex berkala.
- __Score drift antar sumber__: gunakan normalisasi lokal (window) di CTE; golden queries untuk regression.

### Feature Flags (baru)
- `USE_SQL_FUSION` (default: true): gunakan CTE fusion di DB; jika false, fallback ke jalur lama.
- `USE_RERANKER` (default: false): aktifkan reranker pasca-fusi untuk query ambigu.
- `NEW_PG_RETRIEVAL` (payung): mengaktifkan ltree predicates + unified schema + explicit-first routing.

### Safe Deletion Matrix (bersyarat)
- __Hapus `src/services/search/bm25_search.py`__ jika:
  - `hybrid_search.py` tidak lagi mengimpor/menjalankan BM25 service.
  - `scripts/setup_hybrid_search.py` tidak lagi mengimpor factory BM25.
  - FTS sepenuhnya via SQL CTE (GIN `tsv_content`).
- __Hapus `src/services/search/rrf_fusion.py`__ jika:
  - `hybrid_search.py` & `scripts/setup_hybrid_search.py` tidak lagi mengimpor RRF.
  - Fusion digantikan sepenuhnya oleh CTE SQL.
- __Bersihkan placeholder rerank di `vector_search.py`__:
  - Arahkan ke `src/services/search/reranker.py` atau hapus jika tidak dipakai.
- __Jangan hapus migrasi lama 002/003__:
  - Pertahankan sejarah migrasi. Tambahkan migrasi baru untuk drop kolom BM25 bila diperlukan.

### Hotspots untuk Refactor Ringan
- `vector_search.py`: ganti predicate explicit ke ltree; lepas ketergantungan parent_*; rapikan fungsi panjang (split helper SQL builder, score normalize).
- `hybrid_search.py`: enforce `SearchResponse` + feature flags; lepas impor RRF/BM25.
- `indexer.py`: hentikan konsumsi `bm25_body`/`number_label`; pure consumer orchestrator.
- `pdf_orchestrator.py`: pastikan emit `unit_path`/`parent_unit_id`; stop `bm25_body`.
- `document_contract.py`: drop `bm25_body` dan `number_label` dari kontrak.

### Keputusan Ringkas: Konten & Indeks (hasil diskusi terbaru)
- __Satu sumber teks__: `content` dari orchestrator (`src/services/pdf/pdf_orchestrator.py`) adalah sumber kebenaran untuk semua unit.
- __Pertahankan `local_content`__: teks atomik per unit (`ayat/huruf/angka`) yang dipakai untuk membangun `content` (agregasi Pasal) dan `display_text`. Jangan dihapus.
- __Hapus `bm25_body`__: redundan. Hentikan pengisian di orchestrator; ganti semua konsumsi di indexer dengan `local_content` atau `display_text` sesuai konteks. Rencanakan migrasi untuk drop kolom di model/DB.
- __FTS pakai `tsv_content`__: kolom `tsvector` yang DITURUNKAN dari `content` (atau dari MV `pasal_fts_mv`) dengan `unaccent` saat konstruksi; index GIN. Query dengan `to_tsquery` + operator frasa `<->`/proksimitas `<N>` dan ranking `ts_rank_cd`.
- __Vector embeddings__: DITURUNKAN dari `content` (utama di level Pasal) dan diindeks HNSW (pgvector). Berbeda total dengan `tsv_content`.
- __Hierarki__: gunakan `unit_path` (ltree) untuk explicit match (`=`) dan subtree (`<@`). Array `path` hanya untuk UI; tidak dipakai retrieval.
- __Schema respons (API)__: `SearchResponse` tidak membawa `bm25_body`, `path` (array), atau `number_label`. Sertakan `unit_id`, `unit_path`, `label_display`, `citation_string`, `display_text` (snippet), dan bila perlu cuplikan `content` yang diringkas.
- __Acceptance tambahan__:
  - FTS akurat dan cepat tanpa `bm25_body` (mengandalkan `tsv_content`).
  - UI tetap kaya karena `display_text`/`label_display`/`citation_string` dipertahankan.
  - Hybrid explicit-first + dedup by `unit_id` konsisten di SQL CTE.

### Checklist Tambahan (P0 → P2)
- __P0__ (Schema + Respons + Explicit)
  - [ ] Buat `src/schemas/search.py` dan migrasikan semua service.
  - [ ] `hybrid_search.py`: paksa `SearchResponse` (hapus fallback list).
  - [ ] `parser.py`: emit `unit_path_exact/prefix` + `span`.
  - [ ] `vector_search.py`: `_lookup_citation_units` pakai ltree predicates.
  - [ ] Tambah feature flags: `NEW_PG_RETRIEVAL`, `USE_SQL_FUSION`, `USE_RERANKER` di `src/config/settings.py`.
  - [ ] Lepas impor RRF & BM25 dari `scripts/setup_hybrid_search.py` (sementara di-comment bila perlu).
- __P1__ (FTS + MV + Fusion)
  - [ ] Implement `tsv_content` + GIN (unaccent saat konstruksi), query `to_tsquery` dengan `<->`/`<N>` + `ts_rank_cd`.
  - [ ] Buat MV `pasal_fts_mv` + index; integrasikan ke jalur FTS di SQL.
  - [ ] Implementasi CTE fusion (explicit/fts/vector) + dedup + normalisasi skor + `match_type` + `explain_score`.
  - [ ] `hybrid_search.py`: gunakan CTE fusion sesuai `USE_SQL_FUSION`.
- __P2__ (Refactor Kompleksitas + Guardrail)
  - [ ] Pecah fungsi cyclomatic tinggi (lihat daftar di ringkasan audit) + unit test per util.
  - [ ] Guardrail intent amount: post-filter nominal wajib ada di konteks; LLM tidak menjawab nominal tanpa bukti.
  - [ ] Hapus file: `src/services/search/bm25_search.py` jika syarat P1 terpenuhi (lihat Safe Deletion Matrix).
  - [ ] Hapus file: `src/services/search/rrf_fusion.py` jika RRF sudah tidak diimpor di mana pun.
  - [ ] Rapikan `vector_search.py` rerank placeholder: gunakan `reranker.py` atau hapus.

