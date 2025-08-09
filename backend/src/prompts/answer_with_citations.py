from __future__ import annotations
"""Build user prompt with context citations."""

from typing import Dict, List


def build_user_prompt(query: str, contexts: List[Dict[str, str]]) -> str:
    """Compose prompt with numbered citations."""

    blocks = []
    for i, c in enumerate(contexts, 1):
        blocks.append(f"[{i}] {c['citation']}\n{c['text']}\n")
    ctx = "\n".join(blocks)
    return f"Pertanyaan: {query}\n\nKonteks:\n{ctx}\nJawab dengan menyertakan [citation]."
