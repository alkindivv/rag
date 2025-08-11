# AGENT — LLM Router & Intent-Aware Prompts
**Tujuan:** Mengimplementasikan routing LLM provider dan prompt templates yang aware terhadap intent query hukum Indonesia

## Overview
Modul ini bertanggung jawab untuk routing antara berbagai LLM providers (Gemini, OpenAI, Anthropic) dengan mekanisme fallback, serta menyediakan prompt templates yang disesuaikan dengan intent query hukum Indonesia. Modul ini juga mengimplementasikan 6 template khusus untuk penjelasan huruf/karakter dalam dokumen hukum.

## Scope & Boundaries
- In-scope:
  - Routing LLM providers dengan fallback mechanism
  - Intent-aware prompt templates
  - 6 template huruf/karakter khusus
  - Timeout dan backoff handling
  - Output JSON-first dengan citations
  - Out-of-scope:
  - Embedding generation
  - Database queries
  - Retrieval logic

## Inputs & Outputs
- Input utama:
  - Query pengguna
  - Context dari retrieval results
  - Intent classification
- Output:
  - Structured JSON response dengan quotes, citations, reasoning_trace
  - Human-readable answer
- Artefak file:
  - `src/services/llm/router.py`
  - `src/services/llm/prompts.py`

## Dependencies
- Membutuhkan anchor checklist:
  - retrieval.router (anchor:retrieval.router)
  - answers.builder (anchor:answers.builder)
  - utils.http (anchor:utils.http)
- Menyediakan anchor checklist:
  - llm.router (anchor:llm.router)
  - prompts.intent (anchor:prompts.intent)

## [PLANNING]
1. Implementasi BaseLLMProvider interface
2. Membuat provider implementations untuk Gemini, OpenAI, Anthropic
3. Membangun LLMFactory untuk dynamic provider selection
4. Mengembangkan intent-aware prompt templates
5. Menambahkan 6 template huruf/karakter khusus

## [EXECUTION]
1. Refactor `src/services/llm/router.py` untuk implementasi provider switch dan backoff mechanism
2. Implementasi `src/services/llm/prompts.py` dengan semua template intent-aware
3. Tambahkan 6 template huruf/karakter khusus:
   - Template Ringkas
   - Ekspansi Konseptual
   - Perbandingan Redaksi
   - Penomoran Hierarkis
   - Penjelasan Umum/Pasal
   - Meta-karakter
4. Jalankan `pytest -q tests/unit/test_llm_router.py`
5. Dokumentasi di `src/services/llm/README.md`

## [VERIFICATION]
- Unit test untuk LLM router dengan semua providers lulus
- Prompt templates menghasilkan output JSON yang valid
- Timeout/backoff mechanism bekerja sesuai spec
- 6 template huruf/karakter tersedia dan sesuai format

## [TESTS]
- Unit:
  - `tests/unit/test_llm_router.py` - provider switch, timeout/backoff
  - `tests/unit/test_prompts.py` - semua intent templates valid JSON
- E2E:
  - Query dengan intent explicit_cite menggunakan template yang tepat
  - Query dengan intent definition menghasilkan definisi yang akurat
- Cara jalan cepat: `python tests/run_tests.py --quick src/services/llm`

## Acceptance Criteria
- (1) Router dapat switch antara Gemini/GPT/Claude dengan fallback mechanism
- (2) Semua intent templates tersedia dan menghasilkan JSON yang valid
- (3) 6 template huruf/karakter khusus diimplementasikan sesuai spec
- (4) Timeout dan backoff 429 dihandle dengan benar

## Checklist Update Commands
> Gunakan **scripts/checklist.py** untuk menandai anchor terkait.

```bash
python scripts/checklist.py --mark anchor:llm.router
python scripts/checklist.py --mark anchor:prompts.intent
```

## Notes
- Pastikan semua providers menggunakan async HTTP calls
- Template harus mengikuti format JSON strict sesuai schema
- Citations harus berisi unit_id dan doc_form/number/year
