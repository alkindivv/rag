# AGENT — SQL Contracts & Queries
**Tujuan:** Mengimplementasikan SQL contracts untuk explicit retrieval, FTS, dan vector search dengan parameterized queries

## Overview
Modul ini bertanggung jawab untuk semua database queries yang digunakan dalam sistem RAG hukum Indonesia. Ini mencakup explicit chain resolution, sibling counters, FTS leaf-level search, dan vector pasal-level search. Semua queries diimplementasikan sebagai parameterized statements untuk mencegah injection.

## Scope & Boundaries
- In-scope:
  - Explicit chain SQL (doc → pasal → ayat → huruf → angka)
  - Sibling counters (count huruf/ayat/angka)
  - FTS leaf-level queries
  - Vector pasal-level queries
  - Parameterized query implementation
- Out-of-scope:
  - Database connection management
  - Embedding generation
  - Application logic

## Inputs & Outputs
- Input utama:
  - Document metadata (form, number, year)
  - Unit references (pasal, ayat, huruf, angka)
- Output:
  - Legal units with content
  - Count results
  - Vector search results
- Artefak file:
  - `src/db/queries.py`
  - `src/db/models.py`

## Dependencies
- Membutuhkan anchor checklist:
  - data.json.schema (anchor:data.json.schema)
- Menyediakan anchor checklist:
  - db.explicit.sql (anchor:db.explicit.sql)
  - db.fts.sql (anchor:db.fts.sql)
  - db.vector.sql (anchor:db.vector.sql)

## [PLANNING]
1. Implementasi explicit chain resolution queries
2. Membangun sibling counter queries
3. Menambahkan FTS leaf-level queries
4. Mengembangkan vector pasal-level queries
5. Validasi semua queries parameterized

## [EXECUTION]
1. Buat `src/db/queries.py` dengan semua SQL contracts:
   - Explicit chain resolution
   - Sibling counters
   - FTS leaf-level
   - Vector pasal-level
2. Pastikan semua queries parameterized dan tidak vulnerable terhadap injection
3. Implementasi parent_* relationships:
   - parent_pasal_id untuk ayat
   - parent_ayat_id untuk huruf
   - parent_huruf_id untuk angka
4. Jalankan `pytest -q tests/unit/test_db_queries.py`
5. Dokumentasi di `src/db/README.md`

## [VERIFICATION]
- Explicit chain queries menghasilkan unit yang tepat
- Sibling counters mengembalikan jumlah yang akurat
- FTS queries menggunakan ts_rank dan @@ tsquery('indonesian')
- Vector queries join dengan legal_units untuk citation_string
- Semua queries parameterized dan aman dari injection

## [TESTS]
- Unit:
  - `tests/unit/test_explicit_sql.py` - resolve huruf via parent_ayat_id, angka via parent_huruf_id
  - `tests/unit/test_db_counters.py` - sibling counters functionality
  - `tests/unit/test_fts_queries.py` - FTS leaf-level search
  - `tests/unit/test_vector_queries.py` - vector pasal-level search
- E2E:
  - Explicit resolution dari dokumen hukum
  - Sibling counting untuk pasal/ayat/huruf
- Cara jalan cepat: `python tests/run_tests.py --quick src/db`

## Acceptance Criteria
- (1) Explicit chain resolution 100% deterministic menggunakan parent relationships
- (2) Sibling counters mengembalikan count yang akurat untuk semua unit levels
- (3) FTS queries menggunakan indexing yang efisien dan parameterized
- (4) Vector queries join dengan legal_units dan mengembalikan citation_string
- (5) Semua queries parameterized tanpa string formatting

## Checklist Update Commands
> Gunakan **scripts/checklist.py** untuk menandai anchor terkait.

```bash
python scripts/checklist.py --mark anchor:db.explicit.sql
python scripts/checklist.py --mark anchor:db.fts.sql
python scripts/checklist.py --mark anchor:db.vector.sql
```

## Notes
- Gunakan GIN index di legal_units.content_vector (TSVECTOR)
- Gunakan HNSW pgvector index di document_vectors.embedding (cosine)
- Parent relationships harus sesuai dengan JSON schema
