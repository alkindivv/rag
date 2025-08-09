from __future__ import annotations
"""PDF text extractor using PyMuPDF."""

from typing import List

import fitz


def extract_text(path: str) -> List[str]:
    """Extract text per page from a PDF file."""

    pages: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            pages.append(page.get_text())
    return pages
