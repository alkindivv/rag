# CHECKLIST — RAG Hukum Indonesia (Global)

## 1. Data & Skema
- [ ] (anchor:data.json.schema) JSON schema terdokumentasi di `src/schemas/json.md`
- [ ] (anchor:data.json.validator) Validator `src/validators/json_validator.py`
- [ ] (anchor:data.extractor.sync) Extractor mengeluarkan JSON + raw .txt

## 2. DB & Query Contracts
- [ ] (anchor:db.explicit.sql) SQL explicit (pasal→ayat→huruf→angka) siap pakai
- [ ] (anchor:db.fts.sql) SQL FTS (legal_units & optional raw doc)
- [ ] (anchor:db.vector.sql) SQL vector (pgvector, join pasal)

## 3. Embedding & Indexing
- [ ] (anchor:embed.jina.v4) Jina v4 embedder terpasang & dim guard
- [ ] (anchor:pipeline.indexer) Indexer pasal-level sinkron dengan EMBEDDING_DIM
- [ ] (anchor:pipeline.postindex) Post-index housekeeping (stat, vacuum hint, etc.)

## 4. Retrieval & Ranking
- [ ] (anchor:retrieval.router) Router explicit vs contextual
- [ ] (anchor:retrieval.parallel) Parallel FTS+Vector, merge & normalize
- [x] (anchor:rerank.jina) Jina Reranker + fallback + circuit breaker

## 5. LLM & Prompts
- [x] (anchor:llm.router) LLM router (Gemini/GPT/Claude) + timeout/backoff
- [ ] (anchor:prompts.intent) Intent-aware prompts + 6 "huruf/karakter" templates
- [x] (anchor:answers.builder) Answer builder JSON-first + citation guarantees

## 6. Utils & Observability
- [ ] (anchor:utils.http) HTTP client retries/jitter/circuit
- [ ] (anchor:utils.logging) Structured logging/timing
- [ ] (anchor:metrics) In-process metrics collectors

## 7. Tests
- [ ] (anchor:tests.unit) Unit coverage inti (router, SQL, embedder, rerank, prompts)
- [ ] (anchor:tests.e2e) E2E smoke (explicit, count, contextual)
- [ ] (anchor:tests.quick) `tests/run_tests.py --quick` green

## 8. Docs
- [ ] (anchor:docs.agents.map) Root AGENTS map + link ke sub-AGENTS

**Aturan update:** Baris berformat `- [ ] (anchor:xxx)` diubah menjadi `- [x] (anchor:xxx)` oleh `scripts/checklist.py`.
