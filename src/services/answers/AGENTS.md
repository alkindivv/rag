# AGENT — Answer Builder (JSON-first)
**Tujuan:** Mengimplementasikan answer builder yang menghasilkan output JSON-first dengan citations dan grounding yang akurat

## Overview
Modul ini bertanggung jawab untuk membangun jawaban struktural berdasarkan retrieval results dan intent classification. Modul ini menghasilkan output JSON pertama dengan quotes, citations, dan reasoning trace, diikuti oleh rendering human-readable.

## Scope & Boundaries
- In-scope:
  - JSON-first answer building
  - Grounding dengan literal quotes
  - Citation generation dengan unit_id
  - Confidence scoring heuristic
  - JSON validation dengan retry mechanism
- Out-of-scope:
  - Retrieval logic
  - Embedding generation
  - LLM provider implementation

## Inputs & Outputs
- Input utama:
  - Query pengguna
  - Retrieval results
  - Intent classification
  - Strategy (explicit/hybrid)
- Output:
  - Structured JSON answer dengan quotes, citations, reasoning_trace
  - Human-readable answer
- Artefak file:
  - `src/services/answers/answer_builder.py`
  - `src/services/answers/json_validator.py`

## Dependencies
- Membutuhkan anchor checklist:
  - prompts.intent (anchor:prompts.intent)
  - retrieval.router (anchor:retrieval.router)
  - data.json.validator (anchor:data.json.validator)
- Menyediakan anchor checklist:
  - answers.builder (anchor:answers.builder)

## [PLANNING]
1. Implementasi answer building pipeline
2. Membangun grounding mechanism dengan literal quotes
3. Menambahkan citation generation berdasarkan unit_id
4. Mengembangkan confidence scoring heuristic
5. Implementasi JSON validation dengan retry

## [EXECUTION]
1. Refactor `src/services/answers/answer_builder.py` untuk JSON-first approach
2. Implementasi grounding dengan kutipan literal + konteks sekitar
3. Tambahkan citation generation dengan unit_id + doc_form/number/year
4. Implementasi confidence scoring:
   - 0.5 * normalized_rerank_score
   - 0.3 * (overlap FTS∩Vector)
   - 0.2 * explicit_match_flag
5. Integrasi dengan JSON validator untuk output validation
6. Jalankan `pytest -q tests/unit/test_answer_builder.py`

## [VERIFICATION]
- Answer builder menghasilkan JSON yang valid sesuai schema
- Quotes mengandung kutipan literal dari unit yang direferensikan
- Citations mencakup unit_id dan doc_form/number/year
- Confidence scoring mengikuti heuristic yang ditentukan
- JSON validation dengan retry mechanism bekerja

## [TESTS]
- Unit:
  - `tests/unit/test_answer_builder.py` - JSON building, grounding, citations
  - `tests/unit/test_answer_validation.py` - JSON validation, retry mechanism
- E2E:
  - Answer building untuk berbagai intent classifications
  - Confidence scoring untuk different retrieval strategies
- Cara jalan cepat: `python tests/run_tests.py --quick src/services/answers`

## Acceptance Criteria
- (1) Semua answers mengikuti format JSON-first
- (2) Quotes berisi kutipan literal dari legal units
- (3) Citations mencakup unit_id dan document metadata
- (4) Confidence scoring menggunakan heuristic yang ditentukan
- (5) JSON validation dengan satu retry jika invalid

## Checklist Update Commands
> Gunakan **scripts/checklist.py** untuk menandai anchor terkait.

```bash
python scripts/checklist.py --mark anchor:answers.builder
```

## Notes
- Harus selalu mengutip teks normatif secara LITERAL
- Jika konteks tidak cukup, jawab "Tidak cukup konteks dari kutipan"
- Semua jawaban harus menyertakan reasoning_trace singkat
