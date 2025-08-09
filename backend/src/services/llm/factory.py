from __future__ import annotations
"""LLM factory."""

from src.config.settings import settings
from src.services.llm.base import BaseLLM
from src.services.llm.providers.anthropic_ import AnthropicLLM
from src.services.llm.providers.gemini import GeminiLLM
from src.services.llm.providers.openai_ import OpenAILLM


def get_llm() -> BaseLLM:
    """Return LLM provider based on settings."""

    provider = settings.llm_provider.lower()
    if provider == "openai":
        return OpenAILLM()
    if provider == "anthropic":
        return AnthropicLLM()
    return GeminiLLM()
