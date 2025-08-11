# AGENT — Retrieval Router + Parallel FTS + Vector
**Tujuan:** Mengimplementasikan router retrieval dan pipeline paralel FTS+Vector untuk query hukum Indonesia

## Overview
Modul ini bertanggung jawab untuk routing query antara explicit retrieval (SQL deterministic) dan contextual retrieval (FTS+Vector paralel). Modul ini juga mengimplementasikan mekanisme parallel retrieval dan merging results dari FTS dan Vector search.

## Scope & Boundaries
- In-scope:
  - Query routing (explicit vs contextual)
  - Parallel FTS and Vector retrieval
  - Result merging and normalization
  - Regex pattern matching untuk dokumen hukum
- Out-of-scope:
  - Embedding generation
  - Reranking logic
  - LLM processing

## Inputs & Outputs
- Input utama:
  - User query
  - Retrieval settings (K values, timeouts)
- Output:
  - Ranked retrieval results
  - Strategy classification (explicit/contextual)
- Artefak file:
  - `src/services/retriever/hybrid_retriever.py`
  - `src/services/retriever/framework_adapters.py`

## Dependencies
- Membutuhkan anchor checklist:
  - db.explicit.sql (anchor:db.explicit.sql)
  - db.fts.sql (anchor:db.fts.sql)
  - db.vector.sql (anchor:db.vector.sql)
  - rerank.jina (anchor:rerank.jina)
- Menyediakan anchor checklist:
  - retrieval.router (anchor:retrieval.router)
  - retrieval.parallel (anchor:retrieval.parallel)

## [PLANNING]
1. Implementasi regex patterns untuk semua jenis dokumen hukum
2. Membangun routing logic antara explicit dan contextual paths
3. Membuat parallel retrieval mechanism untuk FTS dan Vector
4. Mengembangkan result merging dan score normalization
5. Integrasi dengan framework adapters (Haystack/LangChain/LlamaIndex)

## [EXECUTION]
1. Refactor `src/services/retriever/hybrid_retriever.py` untuk implementasi router
2. Buat `src/services/retriever/framework_adapters.py`:
   - SqlFTSRetriever
   - PgVectorRetriever
   - ParallelRetriever
   - JinaRerankNode
3. Implementasi regex patterns:
   - Explicit patterns (dokumen + pasal/ayat/huruf/angka)
   - Pasal first patterns
4. Jalankan `pytest -q tests/unit/test_hybrid_retriever.py`
5. Dokumentasi di `src/services/retriever/README.md`

## [VERIFICATION]
- Unit test router dengan berbagai explicit patterns lulus
- Parallel retrieval FTS+Vector berjalan sesuai timeout
- Result merging menghasilkan unique unit_id
- Score normalization bekerja dengan benar

## [TESTS]
- Unit:
  - `tests/unit/test_query_router.py` - ekstraksi explicit references
  - `tests/unit/test_parallel_retrieval.py` - FTS+Vector paralel, merge unik
- E2E:
  - Query explicit seperti "UU 4/2009 Pasal 149 ayat (2) huruf b"
  - Query contextual dengan fallback mechanism
- Cara jalan cepat: `python tests/run_tests.py --quick src/services/retriever`

## Acceptance Criteria
- (1) Router dapat mendeteksi explicit patterns dengan akurat
- (2) Contextual path selalu menjalankan FTS+Vector secara paralel
- (3) Result merging menghasilkan unique unit_id
- (4) Score normalization menggunakan min-max atau z-score

## Checklist Update Commands
> Gunakan **scripts/checklist.py** untuk menandai anchor terkait.

```bash
python scripts/checklist.py --mark anchor:retrieval.router
python scripts/checklist.py --mark anchor:retrieval.parallel
```

## Notes
- Regex patterns harus mengikuti spec AGENTS.md dengan tepat
- Parallel retrieval harus menggunakan threading/async
- Merging harus mempertahankan metadata dari kedua sources
