# AGENT — Embedding Service (Jina v4)
**Tujuan:** Mengimplementasikan embedding service dengan Jina v4, caching, dan dimension guard

## Overview
Modul ini bertanggung jawab untuk menghasilkan embeddings untuk dokumen hukum menggunakan Jina v4 model. Modul ini juga mengimplementasikan LRU caching untuk query embeddings dan dimension guard untuk memastikan konsistensi vector dimensions.

## Scope & Boundaries
- In-scope:
  - Jina v4 embedding generation
  - LRU caching untuk query embeddings
  - Dimension validation and guard
  - HTTP client with retries/jitter/circuit breaker
- Out-of-scope:
  - Database storage
  - Retrieval logic
  - Reranking

## Inputs & Outputs
- Input utama:
  - Text content untuk di-embed
  - Task type (query/passage)
- Output:
  - Embedding vectors
  - Cached embeddings untuk queries
- Artefak file:
  - `src/services/embedding/embedder.py`
  - `src/services/embedding/cache.py`

## Dependencies
- Membutuhkan anchor checklist:
  - utils.http (anchor:utils.http)
- Menyediakan anchor checklist:
  - embed.jina.v4 (anchor:embed.jina.v4)

## [PLANNING]
1. Implementasi Jina v4 embedder dengan task differentiation
2. Membangun LRU cache untuk query embeddings
3. Menambahkan dimension guard untuk vector consistency
4. Integrasi dengan HTTP client untuk retries/jitter
5. Validasi output embeddings

## [EXECUTION]
1. Implementasi `src/services/embedding/embedder.py` dengan Jina v4
2. Buat `src/services/embedding/cache.py` untuk LRU caching
3. Tambahkan dimension guard pada startup dan embedding generation
4. Integrasi dengan `src/utils/http.py` untuk HTTP calls
5. Jalankan `pytest -q tests/unit/test_embedder.py`
6. Dokumentasi di `src/services/embedding/README.md`

## [VERIFICATION]
- Jina v4 embedder menghasilkan vectors dengan dimensi yang benar
- LRU cache bekerja untuk query embeddings
- Dimension guard mencegah mismatch pada pgvector columns
- HTTP retries/jitter/circuit breaker dihandle dengan benar

## [TESTS]
- Unit:
  - `tests/unit/test_embedder.py` - embedding generation, dim guard
  - `tests/unit/test_embedding_cache.py` - LRU cache functionality
- E2E:
  - Embedding generation untuk passage dan query tasks
  - Cache hit/miss scenarios
- Cara jalan cepat: `python tests/run_tests.py --quick src/services/embedding`

## Acceptance Criteria
- (1) Jina v4 embedder terpasang dan berfungsi dengan task differentiation
- (2) LRU cache mengurangi redundant embedding calls
- (3) Dimension guard memvalidasi vector sizes dengan jelas
- (4) HTTP client dengan retries/jitter/circuit breaker diimplementasikan

## Checklist Update Commands
> Gunakan **scripts/checklist.py** untuk menandai anchor terkait.

```bash
python scripts/checklist.py --mark anchor:embed.jina.v4
```

## Notes
- TASK_QUERY="retrieval.query" dan TASK_PASSAGE="retrieval.passage"
- EMBEDDING_DIM default 368
- Validasi dimensi harus dilakukan pada startup dan saat embedding generation
