# AGENT — HTTP Client, Logging, Circuit Breaker, Metrics
**Tujuan:** Mengimplementasikan utility services untuk HTTP client, logging, circuit breaker, dan metrics collection

## Overview
Modul ini bertanggung jawab untuk utility services yang digunakan di seluruh sistem RAG hukum, termasuk HTTP client dengan retries/jitter, structured logging, circuit breaker implementation, dan basic metrics collectors.

## Scope & Boundaries
- In-scope:
  - HTTP client dengan retries/jitter/circuit breaker
  - Structured logging untuk semua components
  - Circuit breaker implementation
  - Basic metrics collection
- Out-of-scope:
  - Business logic implementation
  - Database queries
  - LLM processing

## Inputs & Outputs
- Input utama:
  - HTTP requests dengan timeout specs
  - Logging requirements dari modules
- Output:
  - HTTP responses dengan error handling
  - Structured log entries
  - Circuit breaker states
  - Metrics data
- Artefak file:
  - `src/utils/http.py`
  - `src/utils/logger.py`
  - `src/utils/circuit_breaker.py`
  - `src/utils/metrics.py`

## Dependencies
- Membutuhkan anchor checklist:
  - Tidak ada dependencies eksternal
- Menyediakan anchor checklist:
  - utils.http (anchor:utils.http)
  - utils.logging (anchor:utils.logging)
  - metrics (anchor:metrics)

## [PLANNING]
1. Implementasi HTTP client dengan retries dan jitter
2. Membangun structured logging system
3. Menambahkan circuit breaker implementation
4. Mengembangkan basic metrics collectors
5. Validasi integrasi dengan semua modules

## [EXECUTION]
1. Refactor `src/utils/http.py`:
   - Tambahkan retry mechanism dengan exponential backoff
   - Implementasi jitter untuk menghindari retry storms
   - Tambahkan circuit breaker untuk external calls
   - Validasi timeout handling
2. Buat `src/utils/logger.py`:
   - Structured logging dengan JSON format
   - Timing logs untuk performance tracking
   - Error logs dengan context information
3. Implementasi `src/utils/circuit_breaker.py`:
   - Circuit state management (closed, open, half-open)
   - Failure threshold dan cooldown periods
4. Buat `src/utils/metrics.py`:
   - Basic metrics collectors untuk timing dan counts
   - In-memory metrics storage
5. Integrasi dengan semua modules yang membutuhkan
6. Jalankan `pytest -q tests/unit/test_utils.py`

## [VERIFICATION]
- HTTP client retries/jitter/circuit breaker bekerja sesuai spec
- Structured logs mencakup semua required fields
- Circuit breaker mencegah cascading failures
- Metrics collectors mengumpulkan data yang akurat

## [TESTS]
- Unit:
  - `tests/unit/test_http_client.py` - retries, jitter, circuit breaker
  - `tests/unit/test_logger.py` - structured logging
  - `tests/unit/test_circuit_breaker.py` - state transitions
  - `tests/unit/test_metrics.py` - data collection
- E2E:
  - HTTP calls dengan various failure scenarios
  - Logging untuk complete request flows
- Cara jalan cepat: `python tests/run_tests.py --quick src/utils`

## Acceptance Criteria
- (1) HTTP client memiliki retries dengan jitter dan circuit breaker
- (2) Structured logging mencakup timing dan context information
- (3) Circuit breaker mencegah external service failures
- (4) Metrics collectors mengumpulkan performance data
- (5) Semua modules dapat menggunakan utilities dengan mudah

## Checklist Update Commands
> Gunakan **scripts/checklist.py** untuk menandai anchor terkait.

```bash
python scripts/checklist.py --mark anchor:utils.http
python scripts/checklist.py --mark anchor:utils.logging
python scripts/checklist.py --mark anchor:metrics
```

## Notes
- HTTP client harus digunakan untuk semua external calls
- Logs harus mencakup {query, strategy, timings, counts}
- Circuit breaker harus memiliki configurable thresholds
- Metrics harus lightweight dan tidak mempengaruhi performance
