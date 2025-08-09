from __future__ import annotations
"""Text cleaning pipeline."""

import re
from typing import Callable, List


CleanFunc = Callable[[str], str]


class TextCleaner:
    """Composable text cleaner."""

    def __init__(self) -> None:
        self._steps: List[CleanFunc] = []

    def register(self, func: CleanFunc) -> None:
        """Register a cleaning step."""

        self._steps.append(func)

    def clean(self, text: str) -> str:
        """Apply all cleaning steps sequentially."""

        for step in self._steps:
            text = step(text)
        return text


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace."""

    return re.sub(r"\s+", " ", text).strip()


def build_default_cleaner() -> TextCleaner:
    """Return a cleaner with default steps."""

    cleaner = TextCleaner()
    cleaner.register(normalize_whitespace)
    return cleaner
