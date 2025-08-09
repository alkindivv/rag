"""Simple PDF orchestrator for building legal document trees.

This module provides a minimal implementation required for the unit
tests. It recognises *Pasal*, *ayat*, and hierarchical letters.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


def build_tree(pages: List[str]) -> Dict[str, Any]:
    """Build a hierarchical tree from a list of page texts.

    Parameters
    ----------
    pages:
        Pages of text extracted from a PDF.

    Returns
    -------
    dict
        A tree with parsed ``Pasal`` nodes containing ``ayat`` and
        ``huruf`` children. The structure is purposely minimal to
        satisfy the unit tests.
    """
    text = "\n".join(pages)
    lines = text.splitlines()
    root: Dict[str, Any] = {"type": "document", "children": []}
    current_article: Dict[str, Any] | None = None
    current_verse: Dict[str, Any] | None = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        pasal_match = re.match(r"Pasal\s+(\d+[A-Z]?)", line, re.IGNORECASE)
        if pasal_match:
            current_article = {
                "type": "pasal",
                "number": pasal_match.group(1),
                "children": [],
            }
            root["children"].append(current_article)
            current_verse = None
            continue

        ayat_match = re.match(r"\((\d+)\)\s*(.*)", line)
        if ayat_match and current_article is not None:
            current_verse = {
                "type": "ayat",
                "number": ayat_match.group(1),
                "text": ayat_match.group(2).strip(),
                "children": [],
            }
            current_article["children"].append(current_verse)
            continue

        huruf_match = re.match(r"([a-z])\.\s*(.*)", line)
        if huruf_match and current_verse is not None:
            current_verse["children"].append(
                {
                    "type": "huruf",
                    "number": huruf_match.group(1),
                    "text": huruf_match.group(2).strip(),
                }
            )
            continue

    return root
