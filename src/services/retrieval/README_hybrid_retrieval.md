BELOW IS JUST A SOME TEMPLATE CODE, AND YOU SHOULD IMPROVE IT BASED ON MY CODE BASE STRUCTURE


<!-- import re, uuid    
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.db.models import LegalUnit
from src.services.embedding.jina_embedder import JinaEmbedder
from src.services.rerank.base import Reranker, RerankItem

EXPLICIT = re.compile(r"(?i)(UU|PP|Perpres|Permen|Perda)|Pasal\s*\d+[A-Z]*|ayat\s*\(\d+\)|huruf\s*[a-z]|angka\s*\d+|Tahun\s*\d{4}")

def fts_units(db: Session, query: str, limit=20) -> List[Dict[str,Any]]:
    sql = text("""
    SELECT unit_id, unit_type, citation_string, bm25_body, content,
           ts_rank(to_tsvector('indonesian', coalesce(bm25_body,'')),
                   plainto_tsquery('indonesian', :q)) AS rank
    FROM legal_units
    WHERE bm25_body IS NOT NULL
    ORDER BY rank DESC
    LIMIT :limit
    """)
    return [dict(r) for r in db.execute(sql, {"q": query, "limit": limit}).mappings().all()]

def vector_by_query(db: Session, query_emb: list[float], limit=10) -> List[Dict[str,Any]]:
    # asumsikan kolom embedding tipe pgvector dan index HNSW sudah dibuat → gunakan operator cosine <=> 
    sql = text("""
    SELECT hierarchy_path AS unit_id, content_text AS content, doc_type, doc_year, doc_number, pasal_number
    FROM document_vectors
    ORDER BY embedding <=> :q
    LIMIT :limit
    """)
    return [dict(r) for r in db.execute(sql, {"q": query_emb, "limit": limit}).mappings().all()]

async def hybrid_search(db: Session, query: str, embedder: JinaEmbedder, reranker: Reranker | None = None, limit=12):
    # 1) Candidate via FTS (granular leaf)
    fts = fts_units(db, query, limit=limit*2)

    # 2) Candidate via Vector (per pasal)
    q_vec = (await embedder.embed([query]))[0]
    vec = vector_by_query(db, q_vec, limit=limit)

    # 3) Canonicalize & merge
    # FTS candidates: gunakan unit leaf; Vector candidates: pasal (bisa dijelaskan dengan context)
    merged: List[RerankItem] = []
    seen = set()

    for r in fts:
        uid = r["unit_id"]
        if uid in seen: continue
        seen.add(uid)
        text = r.get("bm25_body") or r.get("content") or ""
        merged.append(RerankItem(uid, text, {"source":"fts", "citation": r["citation_string"], "unit_type": r["unit_type"]}))

    for r in vec:
        uid = r["unit_id"]
        if uid in seen: continue
        seen.add(uid)
        text = r.get("content") or ""
        citation = f"{r.get('doc_type','') } {r.get('doc_number','')}/{r.get('doc_year','')}, Pasal {r.get('pasal_number','')}"
        merged.append(RerankItem(uid, text, {"source":"vector", "citation": citation, "unit_type":"pasal"}))

    # 4) Optional rerank
    ranked = merged
    if reranker:
        ranked = await reranker.rerank(query, merged, top_k=limit)

    # 5) Top‑K
    return ranked[:limit]

def is_explicit(query: str) -> bool:
    return bool(EXPLICIT.search(query)) -->