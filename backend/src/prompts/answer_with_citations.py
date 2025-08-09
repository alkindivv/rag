from __future__ import annotations
"""Prompt builder ensuring citations in answers."""

from typing import Any, Dict, Iterable, List, Tuple


def build_prompt(user_query: str, candidates: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Return system and user prompts for answering with citations.

    Each candidate may contain keys:
        * citation_string or citation
        * display_text: text for leaf units
        * pasal_text: full article text for top candidate
        * text: fallback content
    The first candidate is treated as the top article; subsequent ones as leaf
    units. """

    system_prompt = (
        "Anda asisten hukum. Jangan mengarang nomor pasal dan hanya gunakan "
        "citation_string yang diberikan ketika menyebut sumber."  # noqa: E501
    )

    blocks: List[str] = []
    for idx, cand in enumerate(candidates, 1):
        citation = cand.get("citation_string") or cand.get("citation") or ""
        if idx == 1 and cand.get("pasal_text"):
            text = cand["pasal_text"]
        else:
            text = cand.get("display_text") or cand.get("text") or ""
        blocks.append(f"[{idx}] {citation}\n{text}")
    context = "\n\n".join(blocks)

    user_prompt = (
        f"Pertanyaan: {user_query}\n\n"  # noqa: E501
        f"Konteks:\n{context}\n\n"
        "Jawab dalam poin-poin singkat. Akhiri dengan blok 'Sumber:' yang "
        "berisi daftar citation_string sesuai nomor."  # noqa: E501
    )
    return system_prompt, user_prompt


def postprocess_answer(answer: str, citations: Iterable[str]) -> str:
    """Ensure answer contains a 'Sumber' block.

    If missing, append the citations provided.
    """

    if "Sumber:" not in answer:
        src = "\n".join(f"- {c}" for c in citations)
        answer = answer.rstrip() + f"\n\nSumber:\n{src}"
    return answer
