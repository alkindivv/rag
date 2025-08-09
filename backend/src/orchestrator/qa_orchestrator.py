from __future__ import annotations
"""QA orchestrator combining retrieval and LLM."""

from typing import Any, Dict, List

from sqlalchemy.orm import Session

from src.prompts.answer_with_citations import build_prompt, postprocess_answer
from src.retriever.hybrid_retriever import hybrid_search
from src.services.llm.factory import get_llm


def answer(db: Session, query: str, max_ctx_chars: int = 12000) -> Dict[str, Any]:
    """Answer query using hybrid retrieval and LLM."""

    candidates = hybrid_search(db, query, topk=12)
    ctx: List[Dict[str, Any]] = []
    total = 0
    for c in candidates:
        t = c.get("pasal_text") or c.get("display_text") or c.get("text") or ""
        if total + len(t) > max_ctx_chars:
            break
        ctx.append(c)
        total += len(t)
    llm = get_llm()
    system_prompt, user_prompt = build_prompt(query, ctx)
    out = llm.complete(system_prompt, user_prompt, stream=False)
    citations = [cand.get("citation_string") or cand.get("citation", "") for cand in ctx]
    ans = postprocess_answer(out.get("text", ""), citations)
    return {"answer": ans, "candidates": ctx}
