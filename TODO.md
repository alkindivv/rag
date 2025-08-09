Prioritas P0 (harus duluan)
	•	DB & Migrations
	•	Tabel: legal_documents, legal_units, document_vectors, subjects, document_subject, legal_relationships, uji_materi, vector_search_logs.
	•	content_vector (generated) = to_tsvector('indonesian', coalesce(bm25_body,'')) + GIN index.
	•	document_vectors.embedding (pgvector HNSW, cosine).
	•	Enums: DocForm, DocStatus, UnitType.
	•	Constraints/Indexes sesuai models.py.
	•	Utils
	•	text_cleaner.py (pipeline modular + registry).
	•	pattern_manager.py (regex + helper is_amendment_line).
	•	http.py (retry/backoff util + DI httpx client).
	•	logging.py (JSON structured, integrasi level dari env).
	•	Parser
	•	service/pdf/extractor.py (PyMuPDF minimal; context manager).
	•	service/pdf/orchestrator.py (build tree, aggregate pasal).
	•	Indexer
	•	pipeline/indexer.py (upsert → flatten → embed pasal → store vectors).
	•	Error path: jika document_tree kosong → skip embed.
	•	Embedding Service
	•	services/embedding/jina_embedder.py (batch, retry/backoff).
	•	Hash vector helper.
	•	Retriever
	•	retriever/hybrid_retriever.py (explicit regex + vector + FTS + optional rerank).
	•	Config rerank provider via env.
	•	LLM
	•	services/llm/base.py, factory.py, providers (gemini.py, openai_.py, anthropic_.py)—minimal working.
	•	Prompts
	•	prompts/system.py (aturan kutip/citation).
	•	prompts/answer_with_citations.py (build user prompt).
	•	API
	•	api/app.py (POST /ask, POST /index/document, CORS).
	•	Frontend
	•	Next.js 14 App Router: chat page + components (MessageList, SourceCard, ChatInput).
	•	BACKEND_URL env.
	•	CLI
	•	__main__.py (command: index-json, health, migrate).
	•	Testing
	•	Unit tests: parser (variasi Pasal 1A, ayat/huruf/angka), explicit retriever.

⸻

Prioritas P1
	•	Post‑index hooks: build materialized views (opsional).
	•	Reranker Jina: fallback otomatis bila gagal.
	•	Rate limit aware (async limiter) jika nanti pindah ke async.

Prioritas P2
	•	Neo4j adapter untuk knowledge graph (edges: mengubah, mencabut, uji materi ke pasal).
	•	Analytics ringkas: top query, latency, hit source (vector vs fts).

⸻

Acceptance Criteria (ringkas)
	•	Query eksplisit “UU 8 Tahun 1981 Pasal 5 ayat (1) huruf b” mengembalikan leaf benar (<150ms p95 di VPS NVMe lokal dataset kecil).
	•	Query tematik “wewenang penyidik prapenuntutan” memunculkan pasal/ayat relevan + citation valid.
	•	Indexer dapat ingest JSON contoh kamu (UU 6/2023 dsb) tanpa error.
	•	Frontend menampilkan jawaban + sumber clickable.
