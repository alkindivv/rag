from __future__ import annotations
"""FastAPI application exposing RAG endpoints."""

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.db.session import SessionLocal
from src.orchestrator.qa_orchestrator import answer
from src.pipeline.indexer import ingest_document_json

app = FastAPI(title="Legal RAG API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"]
)


@app.post("/ask")
def ask(payload: dict = Body(...)):
    """Answer a user query."""

    q = payload.get("query", "").strip()
    if not q:
        return {"answer": "", "candidates": []}
    with SessionLocal() as db:
        return answer(db, q)


@app.post("/index/document")
def index_document(doc: dict = Body(...)):
    """Ingest parsed document JSON."""

    ingest_document_json(doc)
    return {"status": "ok"}
