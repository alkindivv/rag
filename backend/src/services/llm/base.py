from __future__ import annotations
"""LLM service base classes."""

from abc import ABC, abstractmethod
from typing import Dict


class BaseLLM(ABC):
    """Abstract LLM interface."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, stream: bool = False) -> Dict[str, str]:
        """Return completion text."""
