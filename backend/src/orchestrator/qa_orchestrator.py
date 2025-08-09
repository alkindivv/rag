from __future__ import annotations
"""QA orchestrator combining retrieval and LLM."""

from typing import Any, Dict

from sqlalchemy.orm import Session

from src.prompts.answer_with_citations import build_user_prompt
from src.prompts.system import SYSTEM_PROMPT
from src.retriever.hybrid_retriever import hybrid_search
from src.services.llm.factory import get_llm


def answer(db: Session, query: str, max_ctx_chars: int = 12000) -> Dict[str, Any]:
    """Answer query using hybrid retrieval and LLM."""

    candidates = hybrid_search(db, query, topk=12)
    ctx = []
    total = 0
    for c in candidates:
        t = c.get("text") or ""
        if total + len(t) > max_ctx_chars:
            break
        ctx.append({"citation": c["citation"], "text": t})
        total += len(t)
    llm = get_llm()
    up = build_user_prompt(query, ctx)
    out = llm.complete(SYSTEM_PROMPT, up, stream=False)
    return {"answer": out.get("text", ""), "candidates": candidates[: len(ctx)]}
