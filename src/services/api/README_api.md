BELOW IS JUST A TEMPLATE AND YOU SHOULD IMPROVE IT BASED ON MY CODE BASE STRUCTURE

<!-- from fastapi import FastAPI, Query
from pydantic import BaseModel
from src.db.session import SessionLocal
from src.retriever.hybrid_retriever import route_and_retrieve

app = FastAPI(title="RAG Hukum Indonesia")

class AskRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search")
def search(q: str = Query(...)):
    db = SessionLocal()
    try:
        res = route_and_retrieve(db, q)
        return res
    finally:
        db.close()

@app.post("/ask")
def ask(req: AskRequest):
    db = SessionLocal()
    try:
        res = route_and_retrieve(db, req.query)
        # TODO: rakit context → panggil LLM → jawaban + citations
        return {"mode": res["mode"], "results": res["results"], "answer": None}
    finally:
        db.close() -->