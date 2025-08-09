# Indonesia Legal RAG

This repository contains a minimal Retrieval-Augmented Generation (RAG) pipeline for Indonesian legal documents.  The system is intentionally small but production oriented: each module is under 300 lines, uses structured logging, and relies on environment driven configuration.

## Features

* **Hybrid retrieval** – explicit regex lookup, full‑text search on leaf nodes, and semantic vector search on *pasal* level.
* **PostgreSQL ready schema** – FTS (`tsvector`) and pgvector indexes with Alembic migrations.
* **PDF pipeline** – PyMuPDF extractor and orchestrator that build a hierarchical document tree.
* **Embedding service** – Jina embeddings with batching and retry logic.
* **FastAPI API** – `/index/document` to ingest parsed JSON, `/ask` to query.
* **Next.js chat frontend** – minimal UI to query the backend and show citations.

## Repository Layout

```
backend/
  src/
    api/app.py              # FastAPI endpoints
    config/settings.py      # Pydantic settings
    db/models.py            # SQLAlchemy models
    db/session.py           # Session factory
    db/migrations/          # Alembic env and versions
    orchestrator/qa_orchestrator.py  # Build context & call LLM
    pipeline/indexer.py     # Upsert + embed pipeline
    prompts/                # System and answer prompts
    retriever/hybrid_retriever.py    # Regex + FTS + vector search
    service/pdf/            # PDF extraction and tree build
    services/embedding/jina_embedder.py  # Batch embedder
    services/llm/           # LLM providers & factory
    utils/                  # logging, http client, regex patterns, cleaner
frontend/
  app/chat/                 # Next.js chat page + actions
  components/               # Chat UI components
scripts/                    # DB setup helpers
tests/                      # Unit tests for parser and explicit retriever
```

## Configuration

Environment variables are defined in `backend/src/config/settings.py`:

* `DATABASE_URL` – database connection string.
* `JINA_API_KEY` – key for Jina embedding/rerank API.
* `LLM_PROVIDER` – `gemini`, `openai`, or `anthropic`.
* `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` – provider keys.
* `EMBED_BATCH_SIZE` – embedding batch size (default 16).
* `RERANK_PROVIDER` – optional Jina reranker.
* `ENABLE_STREAM` – enable LLM streaming responses.
* `LOG_LEVEL` – logging level.

Use a `.env` file or export variables before running the app.

## Development

### Install minimal dependencies

```bash
pip install sqlalchemy pgvector psycopg2-binary pydantic-settings fastapi alembic pytest
```

### Run migrations

```bash
PYTHONPATH=backend DATABASE_URL=sqlite:///app.db alembic -c backend/alembic.ini upgrade head
```

### Ingest a document

`pipeline/indexer.py` exposes `ingest_document_json`.

```bash
PYTHONPATH=backend DATABASE_URL=sqlite:///app.db python - <<'PY'
from src.pipeline.indexer import ingest_document_json
import json, sys
with open('data/json/undang_undang_37_2009.json') as f:
    ingest_document_json(json.load(f))
PY
```

### Ask a question

Run the API:

```bash
PYTHONPATH=backend DATABASE_URL=sqlite:///app.db uvicorn src.api.app:app --reload
```

Query using `curl`:

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{"query":"UU 8 Tahun 1981 Pasal 5 ayat (1) huruf b"}'
```

### Frontend

```
cd frontend
npm install
BACKEND_URL=http://localhost:8000 npm run dev
```

## Testing

Run unit tests and migrations:

```bash
PYTHONPATH=backend DATABASE_URL=sqlite:///test.db pytest
PYTHONPATH=backend DATABASE_URL=sqlite:///test_mig.db alembic -c backend/alembic.ini upgrade head
```

## Notes

This project is a starting point.  TODO/PLAN files describe additional features such as CLI helpers, rerankers, and knowledge‑graph adapters that are not yet implemented.
