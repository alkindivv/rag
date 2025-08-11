# Refactor Plan — Indonesian Legal RAG System (Repo: Rag)

Status: draft v1 — 2025-08-11
Owner: Cascade (pairing with you)

---

## Objectives
- Simplify codebase, reduce redundancy, improve maintainability.
- Deterministic explicit retrieval for precise legal citations.
- Strong contextual retrieval (hybrid + rerank) for open questions.
- Robust, latency-aware LLM routing with graceful fallbacks.
- Clean, canonical data model: Extractor → JSON → DB alignment.
- Comprehensive tests, logging, and light observability.

---

## High-Level Architecture (target)
- Ingestion pipeline
  - Extract (PDF → structured segments) → Canonical JSON (validated) → DB ingest.
  - Canonical IDs with stable, path-based `unit_id` across hierarchy.
- Dual retrieval paths
  - Explicit resolver: regex → reference resolver → exact DB lookup → literal quotation + breadcrumb.
  - Contextual retriever: parallel FTS + vector → merge → rerank → LLM answer builder.
- LLM layer
  - Router (Gemini/OpenAI/Anthropic) with timeout/circuit-breaker.
  - Intent-aware prompts per category, JSON-first answer builder.
- Observability
  - Structured logging, timings, counters, circuit transitions.
  - Optional lightweight metrics (latency, hit-rate).

---

## Canonical Data Model & JSON

Files to align/rework:
- `src/schemas/document_contract.py`
- `src/validators/` (add `json_v2_validator.py`)
- `data/json/*.json` (e.g., `data/json/undang_undang_3_2025.json`)
- `src/db` (models/migrations/queries)

Design principles:
- Hierarchy: `document → buku → bab → bagian → paragraf → pasal → ayat → huruf → angka`.
- Stable ID: `unit_id` is path-based, lowercase, slash-delimited.
  - Example: `uu/2/2024/pasal/5/ayat/1/huruf/a`.
  - Roman numerals normalized for path, retain original in `metadata.original_label`.
- Fields:
  - Common: `unit_id`, `unit_type`, `title`, `local_order`, `parent_unit_id`, `doc_id`, `source`, `status`, `effective_from`, `effective_to`, `amendment_meta`.
  - Text: `local_content` for leaves (ayat/huruf/angka). `composite_content` for `pasal` (optionally concatenated children synopsis for context).
  - Penjelasan:
    - `penjelasan_umum`: one per document; `ref_unit_id = doc_id`.
    - `penjelasan_per_pasal`: per pasal/ayat/huruf, each with `ref_unit_id` to its target unit.
- Cross-refs & amendments:
  - `references`: [{ `type`: definition|exception|sanction|cross_ref|impl_reg, `target_unit_id`, `text_span` }].
  - `amendments`: normalized records for insert/modify/repeal.

Actions:
- [ ] Define `src/schemas/json_v2.py` (canonical Pydantic schema; if already exists, consolidate to single source of truth).  
- [ ] Implement `src/validators/json_v2_validator.py` to validate `data/json/*.json` pre-ingest.  
- [ ] Write `scripts/validate_json.sh` and `tests/unit/test_json_validator.py`.

DB alignment:
- [ ] Ensure tables align with schema (consider minimal, clean set):
  - `legal_units(unit_id PK, unit_type, doc_id, parent_unit_id, local_order, title, local_content, composite_content, source, status, effective_from, effective_to, metadata JSONB)`
  - `penjelasan(id, ref_unit_id FK→legal_units.unit_id, content, type)`
  - `cross_refs(id, source_unit_id, target_unit_id, type, text_span)`
  - `amendments(id, doc_id, operation, target_unit_id, payload JSONB)`
  - Search: `legal_units.fts tsvector`, `legal_units.embedding vector` (pgvector)
- [ ] Create/adjust alembic migrations.
- [ ] Update `src/db/queries.py` to centralize all reads/writes (explicit resolvers, counters, cross-refs).

---

## Explicit Retrieval (deterministic)

Files to introduce/rework:
- `src/services/retriever/explicit_resolver.py` (new)
- `src/services/search/explicit_search.py` (new) vs existing `src/services/retriever/` and `src/services/search/`
- `src/services/llm/prompts/explicit_templates.py` (only for formatting, no generation for lookup)

Capabilities:
- Regex parsing for Indonesian legal citations with variants: UU/PP/Perpu/Perpres/Permen/POJK/Perda/SEMA/SEOJK; No/Nomor/Nmr; Tahun/ YYYY; Pasal [0-9][A-Z]?; Ayat (n); Huruf a-z; Angka 1). Handle pasal suffix (14A, 27B, bis, ter).
- Roman numerals for Bab recognized; normalize for path.
- Construct `unit_id` deterministically; resolve via DB.
- Return:
  - Literal quote of `local_content` (or `composite_content` for pasal) + exact breadcrumb (document → … → target).
  - Sibling counters (e.g., number of huruf in a given ayat; number of ayat in pasal).  
- Fallback: if partial reference not found, degrade to the nearest ancestor (e.g., `pasal` if `ayat` missing) with clear notice.

Actions:
- [ ] Centralize regex patterns in `src/services/retriever/patterns.py` (unit-tested).  
- [ ] Implement `ExplicitResolver.parse(text) -> ExplicitRef` and `.resolve(ExplicitRef) -> LegalUnitMatch`.
- [ ] Add DB helpers in `src/db/queries.py` for exact lookups and sibling counts.
- [ ] E2E tests: `tests/e2e/test_explicit_queries.py` covering citation variants.

---

## Contextual Retrieval (hybrid + rerank)

Existing:
- `src/services/search/hybrid_search.py` (hybrid merge), `src/services/search/reranker.py` (with circuit breaker), `src/services/embedding/embedder.py`

Target:
- Parallel FTS + Vector searches with dynamic biasing (gov-style boost preserved).  
- Reranker: Jina v2 or BGE/Cohere, with circuit-breaker and timeouts; if down, fallback to blend heuristic.
- Output top-N passages at unit granularity (pref: `pasal` snippets or leaf units) with clean text.

Actions:
- [ ] Ensure FTS uses cleaned text from `legal_units.local_content` and/or `composite_content` with `COALESCE` safeguards.  
- [ ] Vector: embed `pasal` (and optionally span-chunked long pasal).  
- [ ] Parallel retrieval orchestration in `src/services/search/hybrid_search.py` (confirm existing logic; simplify if needed).  
- [ ] Implement `src/services/search/blend.py`: deterministic fallback ranker (BM25 score norm + embedding cosine if available).  
- [ ] Tests: `tests/integration/test_hybrid_search.py` and `tests/e2e/test_contextual_queries.py` for provided sample questions.

---

## LLM Router, Prompting, and Answers

Existing:
- Router with backoff/fallback: `src/services/llm/router.py`, providers under `src/services/llm/`.
- Answer builder and JSON v2: `src/services/answers/answer_builder.py`, `src/schemas/json_v2.py`.

Planned:
- Intent detection (light, rule-first): explicit vs contextual + sub-intents: definition, procedure, sanction, exception, authority, rights-duties, transitional, cross-ref, amendment, explanation, huruf/karakter specials.
- Prompt templates per intent: `src/services/llm/prompts/`.
- Always include literal quotation blocks + breadcrumb + citations.

Actions:
- [ ] Enrich `src/services/llm/intents.py` (new) with rules/keywords + optional model-assisted fallback.  
- [ ] Add templates: `definisi.py`, `prosedur.py`, `sanksi.py`, `pengecualian.py`, `otoritas.py`, `hak_kewajiban.py`, `transisional.py`, `cross_ref.py`, `amendment.py`, `penjelasan.py`, `huruf_karakter.py`.  
- [ ] Ensure `answer_builder` composes JSON-first, then human string; includes confidence and reasoning_trace.
- [ ] Tests: unit tests per template and end-to-end `/ask` behavior.

---

## Amendments & Cross-References

Goals:
- Parse phrases: “di antara Pasal X dan Y disisipkan Pasal Z”, “diubah menjadi…”, “dicabut…”.
- Maintain latest-effective view, while retaining historical records.
- Cross-reference detection: phrases “sebagaimana dimaksud…”, “diatur dalam…”.

Actions:
- [ ] Extend extractor to capture amendment statements as structured events into `amendments` table.  
- [ ] Build materialized view or query layer for “current law state”.  
- [ ] Cross-ref detector in `pipeline` or `validators` that populates `cross_refs` (re-run safe, idempotent).  
- [ ] Surface cross-refs in answers for context and related bases.

---

## Ingestion Pipeline

Existing:
- `src/services/pdf/pdf_orchestrator.py`, `src/ingestion.py`, `src/utils/text_cleaner.py`.

Actions:
- [ ] Refactor `pdf_orchestrator.py` into clear stages: load → layout parse → structure map → to canonical JSON.  
- [ ] Add `src/pipeline/json_to_db.py` to ingest canonical JSON into DB (upsert by `unit_id`).  
- [ ] Add `src/pipeline/reindex.py` to compute FTS/embeddings; schedule-friendly and idempotent.  
- [ ] CLI: `src/cli/ingest.py` with commands: `extract`, `validate`, `load-db`, `reindex`, `all`.

---

## API Layer

Existing:
- `src/main.py`, `src/api/*`.

Actions:
- [ ] Split routes: `/ask/explicit` → explicit resolver; `/ask/contextual` → hybrid; `/ask` auto-router via intent.  
- [ ] Response schema: keep JSON v2 compatibility; no 400/429 leakage; attach timing metrics per stage.
- [ ] CORS and error normalization preserved.

---

## Config & Secrets

Actions:
- [ ] Centralize config in `src/config/settings.py`: API keys, timeouts, topK, weights, model IDs, circuit thresholds.  
- [ ] `.env.example` updated; validation via Pydantic Settings.

---

## Logging & Observability

Actions:
- [ ] Structured logs with context: request_id, strategy, timings (fts_ms, vec_ms, rerank_ms, llm_ms), counts, circuit states.  
- [ ] Add minimal metrics sink (in-memory counters/prometheus-client optional).  
- [ ] Tests assert presence of critical logs for e2e paths.

---

## Testing Strategy

- Unit
  - [ ] Regex parsing (`patterns.py`).
  - [ ] JSON validator.
  - [ ] DB query helpers (explicit lookups, sibling counters).
  - [ ] Blend fallback ranker.
  - [ ] Prompt templates selection.
- Integration
  - [ ] Ingestion pipeline small fixture.
  - [ ] Hybrid retrieval end-to-end without external LLM.
- E2E (key scenarios given):
  - [ ] “apa saja isi pasal 1 ayat 2 uu 8/2020?”
  - [ ] “apa yang dimaksud dengan wanprestasi, dan diatur dalam undang‑undang serta pasal berapa?”
  - [ ] “jika seseorang melakukan pencurian maka akan dipenjara berapa lama?”
  - [ ] “ada berapa huruf dalam pasal 5 ayat (1) uu 2/2024?”
  - [ ] “apa itu kewenangan khusus?”
  - [ ] “uu 2/2024 mengatur tentang apa?”

---

## Migration/Refactor Steps (small commits)

1) Baseline and safety
- [ ] Create feature branch `refactor/v2`.
- [ ] Pin dependencies where needed; run current tests green as baseline.

2) Canonical schema + validator
- [ ] Implement `json_v2.py` + validator + tests.
- [ ] Sample convert 1–2 JSON docs under `data/json/` to v2; validate.

3) DB alignment
- [ ] Alembic migrations; migrate minimal tables; backfill indexes (`fts`, `embedding`).
- [ ] Update `src/db/queries.py` centralization; fix callers.

4) Ingestion pipeline
- [ ] Refactor `pdf_orchestrator.py`; write `pipeline/json_to_db.py`; CLI commands; tests.

5) Explicit resolver
- [ ] Implement patterns, resolver, DB helpers; add `/ask/explicit`; E2E tests.

6) Contextual retrieval
- [ ] Verify hybrid parallel path; implement blend fallback; E2E tests.

7) LLM & prompts
- [ ] Intent detection + templates; ensure answer builder compatibility.

8) Amendments & cross-refs (phase 2 if time)
- [ ] Extract, store, surface; add tests.

9) Logging/metrics
- [ ] Structured logging; timings; minimal metrics; tests assert logs.

10) Docs & cleanup
- [ ] Update `README.md` (run, endpoints, dataset, examples).  
- [ ] Remove deprecated files/duplications; ensure imports match new structure.

---

## File/Directory Changes (planned)

Add:
- `src/services/retriever/patterns.py`
- `src/services/retriever/explicit_resolver.py`
- `src/services/search/blend.py`
- `src/services/llm/intents.py`
- `src/services/llm/prompts/*.py` (per-intent)
- `src/validators/json_v2_validator.py`
- `src/pipeline/json_to_db.py`
- `src/pipeline/reindex.py`
- `src/cli/ingest.py`

Modify:
- `src/services/pdf/pdf_orchestrator.py`
- `src/services/search/hybrid_search.py`
- `src/services/search/reranker.py`
- `src/services/embedding/embedder.py`
- `src/db/queries.py` and migrations
- `src/main.py`, `src/api/*`

Remove/Consolidate:
- Legacy schemas in `src/schemas/` if duplicated by `json_v2.py`.
- Unused adapters/utilities; move config to `src/config/settings.py`.

---

## Performance/Quality Guardrails
- Timeouts: FTS ≤ 300ms, Vector ≤ 500ms, Rerank ≤ 700ms, LLM ≤ 2.5s (soft), total p95 ≤ 3.5s explicit, ≤ 4.5s contextual.
- Circuit breaker thresholds aligned with current implementation (failure_threshold=3, cooldown=30s) with structured logs.
- Heuristic fallback if external reranker/LLM down.

---

## Risks & Mitigations
- Parsing variance of citations → comprehensive patterns + unit tests; nearest-ancestor fallback.
- Amendment modeling complexity → phase it; start with storage, then current-state view.
- Embedding costs/time → chunk by pasal and cache; background reindex.
- Schema drift → canonical JSON + validator + alembic migrations as single source of truth.

---

## Acceptance Criteria
- `/ask/explicit` returns exact node with literal quote, breadcrumb, and sibling counts where applicable.
- `/ask/contextual` returns high-quality answers with quotes and citations; degrades gracefully without external services.
- Ingestion of at least 2–3 sample laws end-to-end via canonical JSON validator into DB with embeddings/FTS.
- Tests green for unit, integration, and e2e scenarios listed.
- README updated with examples and timings.

---

## Next Actions (immediate)
- [ ] Confirm current DB schema and `src/db/queries.py` coverage.  
- [ ] Draft `json_v2.py` fields and validator stub.  
- [ ] Outline `patterns.py` test cases based on real citation strings from corpus.  
- [ ] Create endpoints skeleton for `/ask/explicit` and `/ask/contextual` (feature-flag off until ready).
