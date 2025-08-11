# AGENT — Hybrid Search Orchestrator
**Tujuan:** Mengimplementasikan orchestrator untuk hybrid search pipeline dan glue logic untuk reranker

## Overview
Modul ini bertanggung jawab untuk mengkoordinasikan alur hybrid search, menggabungkan hasil dari berbagai retrieval paths, dan memastikan konsistensi strategi. Modul ini juga menyediakan glue logic untuk integrasi dengan reranker.

## Scope & Boundaries
- In-scope:
  - Hybrid search orchestration
  - Strategy consistency
  - Timing logs
  - Reranker integration
- Out-of-scope:
  - Low-level retrieval implementation
  - LLM processing
  - Embedding generation

## Inputs & Outputs
- Input utama:
  - Query from API
  - Retrieval results from retriever
- Output:
  - Coordinated search strategy
  - Timing and performance logs
- Artefak file:
  - `src/services/search/hybrid_search.py`
  - `src/services/search/search_service.py`

## Dependencies
- Membutuhkan anchor checklist:
  - retrieval.router (anchor:retrieval.router)
  - rerank.jina (anchor:rerank.jina)
  - utils.logging (anchor:utils.logging)
- Menyediakan anchor checklist:
  - Tidak ada anchor baru (modul ini lebih sebagai glue/coordinator)

## [PLANNING]
1. Implementasi hybrid search orchestration logic
2. Membangun strategy consistency mechanism
3. Menambahkan timing logs untuk setiap stage
4. Integrasi dengan reranker service
5. Validasi output format

## [EXECUTION]
1. Refactor `src/services/search/hybrid_search.py` untuk orchestration
2. Implementasi strategy selection berdasarkan query type
3. Tambahkan timing logs untuk retrieval dan reranking stages
4. Integrasi dengan reranker untuk post-processing results
5. Jalankan `pytest -q tests/unit/test_hybrid_search.py`

## [VERIFICATION]
- Hybrid search pipeline berjalan sesuai urutan
- Timing logs mencakup semua stages
- Strategy selection konsisten dengan query patterns
- Integrasi reranker bekerja dengan fallback mechanism

## [TESTS]
- Unit:
  - `tests/unit/test_hybrid_search.py` - orchestration logic
  - `tests/unit/test_search_service.py` - timing dan strategy
- E2E:
  - Hybrid search dengan berbagai query types
  - Fallback mechanism ketika reranker timeout
- Cara jalan cepat: `python tests/run_tests.py --quick src/services/search`

## Acceptance Criteria
- (1) Hybrid search orchestrates retrieval paths dengan benar
- (2) Strategy selection konsisten dan deterministic
- (3) Timing logs mencakup semua stages dengan akurat
- (4) Reranker integration memiliki fallback mechanism

## Checklist Update Commands
> Gunakan **scripts/checklist.py** untuk menandai anchor terkait.

```bash
python scripts/checklist.py --mark anchor:search.hybrid
python scripts/checklist.py --mark anchor:search.reranker
```

## Notes
- Fokus pada orchestration dan coordination, bukan implementation details
- Timing logs harus mencakup retrieval, merging, dan reranking stages
- Strategy harus jelas antara explicit dan contextual paths
