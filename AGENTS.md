Perform a deep audit & repair across the entire RAG system (crawler, PDF extractor, chunking, embeddings, vector DB + FTS, retrieval, reranker, LLM/API). Find and fix bugs, hidden errors, inconsistencies, duplication, and conflicts. Keep changes scoped, surgical, and reversible. No new architecture. No feature creep.

GROUND RULES
	•	Do not add new services/modules unless absolutely necessary.
	•	Prefer small, reviewable PR-sized commits per problem area.
	•	Keep each file ≤300 lines when feasible; extract helpers only if it reduces complexity.
	•	Preserve existing public interfaces unless they are provably wrong.
	•	All SQL literals for unit types use lowercase ('pasal','ayat','huruf','angka').
	•	Centralize config; remove hardcoded envs/params.

REPORTING OUTPUTS (MUST DELIVER)
	1.	AUDIT_REPORT.md — executive summary + risk heatmap + detailed findings per layer with exact file/line refs.
	2.	FIX_PLAN.md — ordered, minimal fix plan (checklist) with estimated impact & test coverage.
	3.	CONSISTENCY_MATRIX.md — matrix yang memetakan schema DB ↔ JSON tree ↔ embeddings ↔ FTS (field-by-field), termasuk tipe data & contoh nilai valid.
	4.	DRIFT_LOG.md — semua drift/inkonsistensi terdeteksi (schema vs code vs data).
	5.	Patch diffs (or PRs) implementing critical fixes only (see Acceptance).

⸻

SCOPE OF AUDIT (WHAT TO CHECK)

1) Configuration & Secrets
	•	All runtime knobs consolidated in src/config/settings.py (or .yaml) — no hardcoded keys, dims, models, endpoints.
	•	Validate required envs on startup with clear errors (e.g., JINA_API_KEY, DB URL, LLM keys).
	•	One source of truth for:
	•	EMBEDDING_MODEL, EMBEDDING_DIM (e.g., 768/1024), EMBEDDING_TASK_QUERY/PASSAGE
	•	RERANKER_MODEL, ENABLE_RERANKER
	•	FTS_LANGUAGE='indonesian'

2) Database Schema & Migrations
	•	Models: LegalDocument, LegalUnit, DocumentVector, Subject, VectorSearchLog.
	•	Confirm:
	•	DocumentVector.embedding dim = settings.EMBEDDING_DIM.
	•	Unique constraint on vectors: (document_id, unit_id, embedding_model, embedding_version).
	•	GIN index on legal_units.content_vector.
	•	HNSW pgvector index with cosine ops on document_vectors.embedding.
	•	Verify all Enum↔SQL literals align (lowercase for unit types).
	•	Validate ANY() filters use explicit casts:
	•	doc_form::text = ANY(:doc_forms::text[])
	•	doc_year = ANY(:doc_years::int[])
	•	doc_number = ANY(:doc_numbers::text[])

3) Crawler / PDF Extractor / Tree Builder
	•	JSON tree fields must match DB ingestion requirements:
	•	unit_id pattern <DOC-ID>/pasal-1/ayat-2/huruf-a/...
	•	number_label, ordinal_int, ordinal_suffix, label_display, seq_sort_key
	•	parent_pasal_id for leaf nodes; path (array) & citation_string
	•	bm25_body (for FTS), local_content (leaf), content (pasal full join)
	•	hierarchy_path is consistent and indexable
	•	Normalize amandement “angka” cases and Pasal 1A (suffix) handling.

4) Embedding & Indexer
	•	Use Jina Embeddings v4 (fixed already): verify request body/task for passage during indexing.
	•	Enforce embedding_dim match DB at index time; fail fast jika mismatch.
	•	Confirm batching, retry/backoff, and timeouts via a shared HTTP client.
	•	Ensure vectors point to PASAL granularity (or whatever the repo decided) consistently.

5) FTS (BM25)
	•	legal_units.content_vector = to_tsvector('indonesian', coalesce(bm25_body,'')).
	•	Ensure reindex step exists in indexer/ingestor.
	•	Verify queries use the correct analyzer and plainto_tsquery('indonesian', :q).

6) Retriever / Hybrid Search / Reranker
	•	Smart routing:
	•	If explicit legal citation → FTS-first (leaf units).
	•	Otherwise hybrid: vector (pasal) + FTS (leaf) → optional rerank.
	•	SQL vector query avoids unsafe casts; use ORDER BY dv.embedding <=> :qvec.
	•	Provide citation fallback if LEFT JOIN to legal_units returns null.
	•	Reranker calls return stable ordering; degrade gracefully on error (keep original order).

7) LLM Layer (Router + Prompts)
	•	LLM router cleanly switches Gemini/GPT/Claude by config.
	•	Prompts enforce grounded answers with citations, avoid hallucinations.
	•	API layer: consistent models, timeouts, retries, structured logging.

8) Logging, Errors, Resilience
	•	All network IO via a single helper (structured logs: url, status, duration, attempts).
	•	Replace print with logger.
	•	Errors produce actionable messages (not stack traces to end users).

9) Tests & Tooling
	•	Fix existing test failures.
	•	Add smoke tests for:
	•	FTS populated
	•	Vector search ordering
	•	Routing branch coverage
	•	Reranker mock happy path + failure path
	•	Optional: ruff/black/mypy checks (no large refactors).

⸻

AUDIT PROCEDURE (DO THESE IN ORDER)
	1.	Inventory & Static Scan
	•	Grep for hardcoded endpoints/keys/dimensions.
	•	Find SQL with uppercase unit types; replace with lowercase.
	•	Find CAST(:query_vector AS vector); switch to :qvec binding.
	2.	Schema & Index Health
	•	Dump SQLAlchemy metadata vs actual DB.
	•	Verify indexes exist (GIN/HNSW) and dimension matches.
	•	Check constraints & duplicates in document_vectors.
	3.	Tree→DB Consistency Check
	•	Write a small script scripts/validate_tree_ingest.py:
	•	Sample 20 docs: ensure unit_id pattern, parent_pasal_id, bm25_body, citation_string present.
	•	Validate no orphan ayat/huruf/angka without a parent_pasal_id.
	4.	FTS Materialization
	•	Run an UPDATE to set content_vector for units missing it.
	•	Rebuild GIN index if necessary.
	5.	Retriever Queries
	•	Add debug logs printing chosen strategy (explicit/fts-only/hybrid).
	•	Ensure filters use ANY() with casts.
	•	Implement citation fallback in vector search.
	6.	Reranker
	•	Mock a reranker response; verify final order & graceful fallback on 4xx/5xx.
	7.	LLM
	•	Ensure router picks model from config.
	•	Prompts include doc citations and disallow unsupported claims.
	8.	Write Reports & Fix Plan
	•	Create AUDIT_REPORT.md, FIX_PLAN.md, CONSISTENCY_MATRIX.md, DRIFT_LOG.md.

⸻

COMMANDS TO RUN (as part of the audit)

# Unit + smoke
./.cli ask "" > should be able explicit to huruf and angka
python tests/run_tests.py --unit
python tests/run_tests.py --quick

# Full suite
pytest -q

# Lint/format (if configured)
ruff check .
black --check .

# Example manual search to trace
python -m src.main search "pertambangan mineral" --debug
python -m src.main search "UU 4/2009 Pasal 121 ayat 149 huruf b " --debug

# Validate DB objects (replace with your DSN)
psql "$DATABASE_URL" -c "\d+ legal_units"
psql "$DATABASE_URL" -c "\d+ document_vectors"

# Spot-check FTS populated
psql "$DATABASE_URL" -c \
"SELECT count(*) FROM legal_units WHERE content_vector IS NOT NULL;"

# Spot-check duplicates in vectors
psql "$DATABASE_URL" -c \
"SELECT unit_id, embedding_model, embedding_version, count(*) 
 FROM document_vectors GROUP BY 1,2,3 HAVING count(*)>1;"




⸻

REQUIRED CODE PATCHES (SNIPPETS)

Vector search SQL (use as-is):
SELECT
  dv.id,
  dv.unit_id,
  lu.unit_type,
  lu.bm25_body AS text,
  lu.citation_string,
  dv.doc_form,
  dv.doc_year,
  dv.doc_number,
  dv.hierarchy_path,
  1 - (dv.embedding <=> :qvec) AS score
FROM document_vectors dv
LEFT JOIN legal_units lu
  ON lu.unit_id = dv.unit_id AND lu.unit_type = 'pasal'
WHERE 1=1
  -- optional filters:
  -- AND dv.doc_form::text = ANY(:doc_forms::text[])
  -- AND dv.doc_year = ANY(:doc_years::int[])
  -- AND dv.doc_number = ANY(:doc_numbers::text[])
ORDER BY dv.embedding <=> :qvec
LIMIT :k;


FTS leaf search SQL (use as-is):

SELECT
  lu.id,
  lu.unit_id,
  lu.unit_type,
  lu.bm25_body AS text,
  lu.citation_string,
  ld.doc_form,
  ld.doc_year,
  ld.doc_number,
  ts_rank(lu.content_vector, plainto_tsquery('indonesian', :q)) AS score
FROM legal_units lu
JOIN legal_documents ld ON ld.id = lu.document_id
WHERE lu.unit_type IN ('ayat','huruf','angka', 'butir')
  AND lu.content_vector @@ plainto_tsquery('indonesian', :q)
ORDER BY score DESC
LIMIT :k;

Populate FTS during ingest/update:
UPDATE legal_units
SET content_vector = to_tsvector('indonesian', coalesce(bm25_body,''))
WHERE id = :id;


Citation fallback (Python):
citation = row.citation_string or f"{row.doc_form} {row.doc_number}/{row.doc_year}, Pasal {row.unit_id.split('/pasal-')[-1].split('/')[0]}"


ACCEPTANCE CRITERIA
	•	All tests pass: python tests/run_tests.py --unit and --quick → 100%.
	•	Manual searches run without SQL/HTTP 400/500s; logs show chosen strategy.
	•	No embedding-dimension mismatch; early, clear error if misconfigured.
	•	FTS returns leaf hits; content_vector populated for all relevant rows.
	•	No duplicate vectors per (document_id, unit_id, model, version).
	•	Reports (AUDIT_REPORT.md, FIX_PLAN.md, CONSISTENCY_MATRIX.md, DRIFT_LOG.md) exist and are complete.

⸻

NON-GOALS
	•	Changing overall architecture, adding new storages, or migrating to different providers.
	•	Late-interaction / ColBERT multivector (future PR).
	•	UI overhauls.

Follow this plan exactly. Keep fixes tight and tested.