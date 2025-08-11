# AGENT — Indexer & Post-Index Housekeeping
**Tujuan:** Mengimplementasikan pipeline indexing dokumen hukum dan post-index housekeeping tasks

## Overview
Modul ini bertanggung jawab untuk indexing dokumen hukum ke dalam database vector, termasuk ekstraksi units, embedding generation, dan storage. Modul ini juga melakukan housekeeping tasks setelah indexing seperti statistik dan maintenance hints.

## Scope & Boundaries
- In-scope:
  - Document indexing pipeline
  - Pasal-level indexing
  - Post-index housekeeping (stats, maintenance)
  - Batch processing capabilities
- Out-of-scope:
  - Real-time retrieval
  - Query processing
  - LLM integration

## Inputs & Outputs
- Input utama:
  - Legal documents (JSON + raw text)
  - Embedding service
- Output:
  - Indexed documents in database
  - Statistics and metadata
- Artefak file:
  - `src/pipeline/indexer.py`
  - `src/pipeline/housekeeping.py`
  - `src/pipeline/batch_processor.py`

## Dependencies
- Membutuhkan anchor checklist:
  - data.json.schema (anchor:data.json.schema)
  - embed.jina.v4 (anchor:embed.jina.v4)
  - db.vector.sql (anchor:db.vector.sql)
- Menyediakan anchor checklist:
  - pipeline.indexer (anchor:pipeline.indexer)
  - pipeline.postindex (anchor:pipeline.postindex)

## [PLANNING]
1. Implementasi indexing pipeline
2. Membangun pasal-level indexer
3. Menambahkan post-index housekeeping tasks
4. Mengembangkan batch processing capabilities
5. Validasi indexing consistency

## [EXECUTION]
1. Buat `src/pipeline/indexer.py` untuk indexing dokumen:
   - Ekstraksi legal units dari JSON documents
   - Generate embeddings menggunakan embedding service
   - Store vectors di document_vectors table
2. Implementasi `src/pipeline/housekeeping.py`:
   - Kumpulkan statistik indexing
   - Berikan maintenance hints (vacuum, analyze)
3. Buat `src/pipeline/batch_processor.py` untuk batch indexing
4. Pastikan indexer sinkron dengan EMBEDDING_DIM settings
5. Jalankan `pytest -q tests/unit/test_indexer.py`

## [VERIFICATION]
- Indexer berhasil menyimpan embeddings dengan dimensi yang benar
- Pasal-level indexing mencakup semua units
- Post-index housekeeping menghasilkan statistik yang akurat
- Batch processor dapat menangani multiple documents
- Indexing pipeline tidak menghasilkan duplicates

## [TESTS]
- Unit:
  - `tests/unit/test_indexer.py` - indexing pipeline
  - `tests/unit/test_housekeeping.py` - stats collection
- E2E:
  - End-to-end indexing untuk sample documents
  - Batch processing untuk multiple documents
- Cara jalan cepat: `python tests/run_tests.py --quick pipeline`

## Acceptance Criteria
- (1) Indexer dapat menyimpan document vectors dengan dimensi yang sesuai
- (2) Pasal-level indexing sinkron dengan EMBEDDING_DIM
- (3) Post-index housekeeping menghasilkan statistik dan maintenance hints
- (4) Batch processor efisien dan reliable
- (5) Tidak ada duplicate indexing atau data corruption

## Checklist Update Commands
> Gunakan **scripts/checklist.py** untuk menandai anchor terkait.

```bash
python scripts/checklist.py --mark anchor:pipeline.indexer
python scripts/checklist.py --mark anchor:pipeline.postindex
```

## Notes
- Indexer harus memvalidasi document structure sebelum indexing
- Housekeeping tasks harus memberikan hints untuk database maintenance
- Batch processing harus memiliki error handling yang robust
