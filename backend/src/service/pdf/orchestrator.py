from __future__ import annotations
"""Build simple legal document tree from extracted pages."""

from typing import Any, Dict, List

from src.utils.pattern_manager import get_hierarchy_patterns
# optional cleaner import kept for future extensions
from src.utils.text_cleaner import build_default_cleaner  # pragma: no cover


def build_tree(pages: List[str]) -> Dict[str, Any]:
    """Return minimal document tree from pages."""

    # preserve newlines for structural parsing
    text = "\n".join(pages)
    patterns = get_hierarchy_patterns()
    lines = text.splitlines()
    root: Dict[str, Any] = {"type": "document", "children": []}
    current_pasal: Dict[str, Any] | None = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m_pasal = patterns["pasal"][0].match(line)
        if m_pasal:
            num = m_pasal.group(1)
            current_pasal = {"type": "pasal", "number": num, "children": []}
            root["children"].append(current_pasal)
            continue
        if not current_pasal:
            continue
        m_ayat = patterns["ayat"][0].match(line)
        if m_ayat:
            ay_num, text_ayat = m_ayat.groups()
            current_pasal["children"].append(
                {"type": "ayat", "number": ay_num, "text": text_ayat.strip(), "children": []}
            )
            continue
        m_huruf = patterns["huruf"][0].match(line)
        if m_huruf and current_pasal["children"]:
            huruf_char, h_text = m_huruf.groups()
            current_pasal["children"][-1].setdefault("children", []).append(
                {"type": "huruf", "number": huruf_char, "text": h_text.strip()}
            )
            continue
        # append to last node text
        if current_pasal["children"]:
            target = current_pasal["children"][-1]
            if target.get("children"):
                target = target["children"][-1]
            target["text"] = (target.get("text", "") + " " + line).strip()
    return root
