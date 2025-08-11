# AGENT — Test Coverage & Validation
**Tujuan:** Mengimplementasikan comprehensive test coverage untuk semua components sistem RAG hukum

## Overview
Modul ini bertanggung jawab untuk memastikan semua components dalam sistem RAG hukum memiliki test coverage yang memadai, termasuk unit tests, E2E smoke tests, dan performance validation. Modul ini juga mengimplementasikan test runner untuk quick validation.

## Scope & Boundaries
- In-scope:
  - Unit test coverage untuk semua modules
  - E2E smoke test suite
  - Performance validation tests
  - Test runner dengan --quick option
- Out-of-scope:
  - Application logic implementation
  - Production deployment testing
  - External service mocking

## Inputs & Outputs
- Input utama:
  - Module implementations
  - Test requirements dari AGENTS.md
- Output:
  - Test suites (unit & E2E)
  - Test runner script
  - Validation reports
- Artefak file:
  - `tests/unit/*.py`
  - `tests/e2e/*.py`
  - `tests/run_tests.py`

## Dependencies
- Membutuhkan anchor checklist:
  - Tidak ada dependencies eksternal
- Menyediakan anchor checklist:
  - tests.unit (anchor:tests.unit)
  - tests.e2e (anchor:tests.e2e)
  - tests.quick (anchor:tests.quick)

## [PLANNING]
1. Implementasi unit tests untuk semua modules
2. Membangun E2E smoke test suite
3. Menambahkan performance validation tests
4. Mengembangkan test runner dengan --quick option
5. Validasi semua tests passing

## [EXECUTION]
1. Buat unit tests di `tests/unit/`:
   - `test_query_router.py` - ekstraksi explicit references
   - `test_explicit_sql.py` - resolve huruf/ayat dengan parent relationships
   - `test_parallel_retrieval.py` - FTS+Vector paralel, merge unik
   - `test_reranker.py` - happy path, timeout fallback, circuit breaker
   - `test_answer_builder.py` - semua templates valid JSON, citations
   - `test_llm_router.py` - provider switch, backoff/timeout
2. Implementasi E2E smoke tests di `tests/e2e/`:
   - Explicit query resolution
   - Sibling counting
   - Contextual queries dengan reranking
   - Multi-document definition queries
3. Buat `tests/run_tests.py` dengan --quick option
4. Jalankan semua tests untuk memastikan coverage

## [VERIFICATION]
- Unit tests mencakup semua functionality requirements
- E2E smoke tests validasi integration paths
- Performance tests memastikan timing budgets
- Test runner --quick option bekerja dengan efisien
- Semua tests passing dalam waktu wajar

## [TESTS]
- Unit:
  - `tests/unit/test_query_router.py` - pattern extraction
  - `tests/unit/test_explicit_sql.py` - parent relationships
  - `tests/unit/test_parallel_retrieval.py` - retrieval merging
  - `tests/unit/test_reranker.py` - fallback mechanism
  - `tests/unit/test_answer_builder.py` - JSON validation
  - `tests/unit/test_llm_router.py` - provider switching
- E2E:
  - "apa isi pasal 149 ayat (2) huruf b uu 4/2009"
  - "ada berapa huruf dalam pasal 5 ayat (1) uu 2/2024"
  - "uu 2/2024 mengatur tentang apa?"
  - "apa itu kewenangan khusus?"
- Cara jalan cepat: `python tests/run_tests.py --quick`

## Acceptance Criteria
- (1) Unit tests mencakup semua core functionality
- (2) E2E smoke tests validasi user scenarios
- (3) Performance tests memastikan timing budgets
- (4) Test runner dengan --quick option memberikan validasi cepat
- (5) Semua tests hijau dan dalam waktu wajar

## Checklist Update Commands
> Gunakan **scripts/checklist.py** untuk menandai anchor terkait.

```bash
python scripts/checklist.py --mark anchor:tests.unit
python scripts/checklist.py --mark anchor:tests.e2e
python scripts/checklist.py --mark anchor:tests.quick
```

## Notes
- Tests harus mencakup edge cases dari AGENTS.md
- E2E tests harus menggunakan real components tanpa mocking berlebihan
- Performance tests harus memvalidasi timing SLAs
