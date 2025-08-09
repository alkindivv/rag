YOURE A PROFESSIONAL RAG DEVELOPER ARCHITECTURE WITH 30 YEARS OF EXPERINCES, I WANT YOU TO HELP ME:
Your task: Refactor for end‑to‑end consistency across crawler/PDF extraction, JSON tree, DB models, pgvector, and FTS. Focus on correctness, speed, and simplicity. No feature creep.

Refactor the codebase to enforce one canonical, complete data contract across:
	1.	crawler/PDF extraction → 2) JSON document+tree → 3) DB schemas → 4) indexer (FTS + vectors) → 5) hybrid search.

Jina is already correct — do not touch embedding client configuration.
Your mission is format alignment and consistency. No “minimal fields”. Use the full field set below.

DELIVERABLES (must do)
	1.	Canonical JSON contract (full fields) with pydantic validators.
	2.	DB schema (SQLAlchemy + alembic) that maps 1:1 to the JSON and supports FTS + vectors.
	3.	Parser/Orchestrator that always emits the canonical JSON and reconstructs pasal.content from children.
	4.	Indexer that consumes the JSON, writes to DB, builds FTS, and inserts vectors (use existing embedder unchanged).
	5.	Hybrid Search that respects citation‑first routing (FTS) and falls back to hybrid if needed.
	6.	Tests that validate the contract, parsing, migrations, and search routing.
	7.	No hardcoded settings; everything in src/config/settings.py or .env.

Keep each file ≤300 lines when possible. Replace prints with structured logging.

⸻

1) CANONICAL JSON CONTRACT (FULL)

Create src/schemas/document_contract.py using pydantic v2.

1.1 Document root
{
  "doc_source": "BPK",
  "doc_id": "UU-2023-6",
  "doc_type": "Peraturan Perundang-undangan",
  "doc_title": "Undang-undang (UU) Nomor 6 Tahun 2023 ...",
  "doc_teu": "Indonesia, Pemerintah Pusat",
  "doc_number": "6",
  "doc_form": "UU",
  "doc_form_short": "UU",
  "doc_year": 2023,
  "doc_place_enacted": "Jakarta",
  "doc_date_enacted": "2023-03-31",
  "doc_date_promulgated": "2023-03-31",
  "doc_date_effective": "2023-03-31",
  "doc_subject": ["CIPTA KERJA"],
  "doc_status": "Berlaku",
  "doc_language": "Bahasa Indonesia",
  "doc_location": "Pemerintah Pusat",
  "doc_field": "HUKUM UMUM",

  "relationships": {
    "mengubah": [{"regulation_reference": "...", "reference_link": "..."}],
    "diubah_dengan": [{"regulation_reference": "...", "reference_link": "..."}],
    "mencabut": [{"regulation_reference": "...", "reference_link": null}],
    "dicabut_dengan": [],
    "menetapkan": []
  },

  "detail_url": "https://...",
  "source_url": "https://...",
  "pdf_url": "https://...",
  "uji_materi_pdf_url": "https://... or null",
  "uji_materi": [
    {
      "decision_number": "39/PUU-XXI/2023",
      "pdf_url": "https://...",
      "decision_content": "<full text>",
      "pasal_affected": ["7","10"],
      "ayat_affected": ["1","2"],
      "huruf_affected": ["i"],
      "decision_type": "bertentangan|ditolak|dikabulkan|... (string)",
      "legal_basis": "UUD 1945 | ...",
      "binding_status": "tidak mengikat|mengikat|bersyarat",
      "conditions": "string or null",
      "interpretation": "string or null"
    }
  ],

  "pdf_path": "data/pdfs/undang_undang_6_2023.pdf",
  "text_path": "data/text/undang_undang_6_2023.txt",
  "doc_content": "<optional full flat text if available>",
  "doc_processing_status": "pdf_downloaded|pdf_processed|txt_processed|failed",
  "last_updated": "2025-08-08T16:57:43.592312",

  "document_tree": { /* see 1.2 */ }
}

1.2 Tree node (uniform for ALL nodes)
{
  "type": "dokumen|buku|bab|bagian|paragraf|pasal|ayat|huruf|angka",
  "unit_id": "UU-2023-6/pasal-5/ayat-1/huruf-a",
  "number_label": "IV|Kesatu|5|1|a|1A",         // raw label as seen
  "ordinal_int": 5,                              // normalized numeric for sorting (roman→int, a→1)
  "ordinal_suffix": "",                          // e.g., "A" in "1A"
  "label_display": "BAB IV|BAGIAN Kesatu|Pasal 5|(1)|a.|1.",
  "seq_sort_key": "0005|A",                      // zero‑padded + suffix
  "citation_string": "Undang-undang ... , Pasal 5 ayat (1) huruf a",
  "path": [
    {"type":"dokumen","label":"Undang-undang (UU) Nomor 6 Tahun 2023 ...","unit_id":"UU-2023-6"},
    {"type":"bab","label":"BAB IV","unit_id":"UU-2023-6/bab-IV"},
    {"type":"bagian","label":"BAGIAN Kesatu","unit_id":"UU-2023-6/bab-IV/bagian-kesatu"},
    {"type":"pasal","label":"Pasal 5","unit_id":"UU-2023-6/bab-IV/bagian-kesatu/pasal-5"},
    {"type":"ayat","label":"(1)","unit_id":"UU-2023-6/bab-IV/bagian-kesatu/pasal-5/ayat-1"},
    {"type":"huruf","label":"a.","unit_id":"UU-2023-6/bab-IV/bagian-kesatu/pasal-5/ayat-1/huruf-a"}
  ],

  "title": "only for dokumen/buku/bab/bagian/paragraf/pasal",
  "content": "only for pasal: full rebuilt content (title + all children flattened)",
  "parent_pasal_id": "UU-2023-6/bab-IV/bagian-kesatu/pasal-5",   // required for ayat/huruf/angka
  "local_content": "only for ayat/huruf/angka",
  "display_text": "(1) ... | a. ... | 1. ...",
  "bm25_body": "same as local_content for leaf nodes",
  "span": null,                                   // optional char spans when extracted from long text
  "tags_semantik": ["definisi","sanksi", ...],    // optional
  "entities": ["Penyidik","Penyidikan"],          // optional

  "children": [ /* same shape recursively */ ]
}

Rules:
	•	Only pasal carries full content (rebuilt).
	•	ayat|huruf|angka carry local_content only.
	•	Consistent citation_string: Pasal X ayat (Y) huruf z angka n (in order).
	•	Normalize case: BAGIAN Kesatu, not BAGIAN satu.
	•	All nodes must have seq_sort_key consistent with ordinal_int + ordinal_suffix.
	•	path is mandatory and must resolve to the node’s unit_id.

Add in document_contract.py:
	•	class TreeNode(BaseModel), class DocumentRoot(BaseModel).
	•	def validate_document_json(obj) -> DocumentRoot that raises explicit errors listing missing/invalid fields.
	
2) DB SCHEMA (SQLAlchemy + Alembic)

Touch src/db/models.py and add migrations under src/db/migrations.

2.1 legal_documents
	•	id UUID (PK, uuid5 stable from doc_form-doc_number-doc_year).
	•	Full metadata columns mirroring all root fields:
	•	doc_source, doc_type, doc_title, doc_teu, doc_number, doc_form, doc_form_short,
doc_year (INT), doc_place_enacted, doc_date_enacted, doc_date_promulgated, doc_date_effective,
doc_status, doc_language, doc_location, doc_field,
detail_url, source_url, pdf_url, uji_materi_pdf_url,
pdf_path, text_path,
doc_content (TEXT),
doc_processing_status, last_updated (TIMESTAMP).
	•	JSONB fields with GIN index:
	•	doc_subject JSONB (array),
	•	relationships JSONB (object),
	•	uji_materi JSONB (array of objects) and (optional) a normalized table (see 2.3) — pick one primary, keep the other cached if needed.
	•	FTS:
	•	content_vector TSVECTOR (indonesian), GIN index, populated from doc_content.
	•	Common indexes: (doc_form, doc_year), (doc_source, doc_status).

2.2 document_vectors
	•	id UUID PK (uuid5 from document_id + hash(content_text)).
	•	document_id UUID FK → legal_documents.id.
	•	embedding pgvector(… your current dim …) — do not change your working Jina settings here.
	•	content_text TEXT (the chunk body used for semantic search).
	•	content_type ENUM(‘pasal’,‘ayat’,‘huruf’,‘angka’,‘bab’,‘bagian’,‘paragraf’,‘full_doc’).
	•	hierarchy_path TEXT (e.g., BAB IV › Bagian Kesatu › Pasal 5 › ayat (1) › huruf a).
	•	Denormalized filters: doc_form, doc_number, doc_year, doc_status.
	•	Structure fields: bab_number, pasal_number, ayat_number (TEXT), token_count INT.
	•	HNSW index on embedding (cosine), composite on (doc_form, doc_year) and (bab_number, pasal_number, ayat_number).

2.3 (optional but recommended) legal_units  — exact citation FTS
	•	Purpose: precision for explicit queries (Pasal/ayat/huruf/angka).
	•	Columns:
	•	id UUID PK (uuid5 from unit_id),
	•	document_id FK,
	•	unit_id TEXT (unique),
	•	type ENUM as above,
	•	number_label, ordinal_int, ordinal_suffix,
	•	label_display,
	•	citation_string,
	•	path JSONB (array of {type,label,unit_id}),
	•	parent_pasal_id TEXT (nullable except for pasal),
	•	local_content TEXT (for ayat/huruf/angka),
	•	full_pasal_content TEXT (only for pasal; same as pasal.content),
	•	bm25_body TSVECTOR (indexed GIN, indonesian),
	•	seq_sort_key.
	•	Create triggers or upsert logic from the indexer to keep this in sync.

2.4 uji_materi (normalized)

If you want analytics/joins, create a table:
	•	id UUID PK,
	•	document_id FK,
	•	decision_number, pdf_url, decision_content,
	•	pasal_affected TEXT[], ayat_affected TEXT[], huruf_affected TEXT[],
	•	decision_type, legal_basis, binding_status, conditions, interpretation.

If you keep JSONB only, skip 2.4; but keep JSONB indexed with GIN.

2.5 migrations
	•	Ensure CREATE EXTENSION IF NOT EXISTS vector;
	•	Ensure content_vector built with to_tsvector('indonesian', doc_content) and kept updated (trigger or refresh step in indexer).
	•	Add all GIN/HNSW indexes described.

⸻

3) PARSER / ORCHESTRATOR

Touch:
	•	src/services/pdf/orchestrator.py
	•	src/utils/text_cleaner.py
	•	src/utils/pattern_manager.py
	•	(new) src/utils/citation.py — single source of truth for build_citation_string(path).

Requirements:
	•	Parse headings: BUKU, BAB, BAGIAN, PARAGRAF, Pasal N[A], ayat (n), huruf a., angka 1.
Normalize casing (BAGIAN Kesatu, etc.).
	•	Strip watermark/noise (e.g., www.djpp.kemenkumham.go.id).
	•	Support amendments where top‑level angka lines appear (e.g., “1. Ketentuan Pasal 1 diubah …”).
	•	Rebuild pasal.content from title + all children (ayat/huruf/angka) with stable formatting.
	•	Emit the canonical JSON (full fields for root + full node shape for every node).
	•	Validate with validate_document_json(); if invalid, raise with detailed path of error.

⸻

4) INDEXER

Touch:
	•	pipeline/indexer.py (create or refactor)
	•	src/db/session.py helper if needed

Flow:
	1.	Load canonical JSON (DocumentRoot).
	2.	Upsert legal_documents with full metadata (all fields).
	3.	Compute/update doc_content (optional: concatenated clean text) → refresh content_vector FTS.
	4.	Insert/update legal_units:
	•	For each pasal: store full_pasal_content (rebuilt).
	•	For ayat|huruf|angka: store local_content → FTS precision on leaf nodes.
	5.	Insert/update document_vectors:
	•	Chunk policy (keep simple):
	•	One chunk per pasal using pasal.content.
	•	Optionally short chunks for long ayat (if you already do this, keep it; don’t change embed config).
	•	Use existing embedder; do not change Jina settings here.
	6.	Fail fast if DB vector dimension mismatches your configured dimension.
	7.	Logging: counts of documents, units, vectors upserted.

⸻

5) HYBRID SEARCH (routing unchanged conceptually, but align with DB)

Touch:
	•	src/services/retriever/hybrid_retriever.py
	•	src/services/search/hybrid_search.py

Routing:
	•	If query matches explicit citation (Pasal \d+[A-Z]?, ayat \(d+\), huruf [a-z], angka \d+):
	•	Use legal_units FTS first (bm25 over bm25_body for leaf; or over full_pasal_content for pasal).
	•	If ≥k_min, return early (format citations from stored citation_string).
	•	Else fallback to hybrid (vectors from document_vectors + doc‑level FTS).
	•	Otherwise:
	•	Hybrid = dense from document_vectors + doc‑level FTS from legal_documents.
	•	(If you already have a reranker wired, keep it as is; do not change Jina config.)

⸻

6) CONFIG & LOGGING

Touch:
	•	src/config/settings.py
	•	Centralize all keys used by crawler/extractor/orchestrator/indexer/search/DB.
	•	No hardcoded paths, models, or dimensions anywhere else.
	•	src/utils/logging.py
	•	Provide a small structured logger initializer used across modules.

⸻

7) TESTS (fix & add)

Touch tests you listed:
	•	tests/test_pdf_orchestrator.py
	•	Cases: “Pasal 1A”, amendment top‑level “angka”, leaf nodes’ parent_pasal_id, citation_string, seq_sort_key, normalization of “BAGIAN Kesatu”.
	•	tests/unit/test_embedding_service.py
	•	Keep as is if Jina is fixed. Only ensure no regression on payload shape if the client is mocked.
	•	tests/unit/test_hybrid_retriever.py
	•	Test citation‑first routing (hits legal_units) and thematic routing (hybrid).
	•	tests/e2e/test_complete_workflow.py
	•	Fix indentation at the failing with patch(...): line; assert full pipeline green path.
	•	tests/run_tests.py and tests/conftest.py
	•	Keep; just make sure they load new config/validators.

Acceptance:
	•	python tests/run_tests.py --unit → 100%
	•	python tests/run_tests.py --quick → 100%
	•	python -m src.main search "pertambangan mineral":
	•	logs show routing path (citation‑first vs hybrid),
	•	results return with valid citation_string, and consistent unit paths.

⸻

8) FILES TO TOUCH (and/or create)
	•	src/schemas/document_contract.py  (new)
	•	src/utils/citation.py              (new)  (build_citation_string/path helpers)
	•	src/utils/pattern_manager.py
	•	src/utils/text_cleaner.py
	•	src/services/pdf/orchestrator.py
	•	src/config/settings.py
	•	src/utils/logging.py
	•	src/db/models.py
	•	src/db/migrations/* (alembic revisions)
	•	pipeline/indexer.py
	•	src/services/retriever/hybrid_retriever.py
	•	src/services/search/hybrid_search.py
	•	tests/test_pdf_orchestrator.py
	•	tests/unit/test_hybrid_retriever.py
	•	tests/e2e/test_complete_workflow.py
	•	keep existing embedding code untouched (Jina is fixed).

⸻

9) IMPLEMENTATION NOTES
	•	Do not invent new partial formats. Every producer (crawler/PDF) and consumer (indexer/search) must use the canonical JSON contract exactly.
	•	Prefer upserts for DB writes (idempotent indexer).
	•	Keep parsing logic simple and robust; move legal patterns to pattern_manager.py.
	•	Centralize citation formatting in utils/citation.py; use it in orchestrator, indexer, and result formatter.
	•	Ensure no hardcoded config left in code; refer to settings.py and .env.

⸻

DONE WHEN
	•	The JSON produced by orchestrator validates against DocumentRoot.
	•	DB rows reflect all fields; FTS on legal_documents and legal_units are usable.
	•	Indexer runs idempotently and reports counts.
	•	Hybrid search routes correctly and returns stable citations.
	•	All tests pass.

Follow this spec exactly. No shortcuts. No dropping fields.



