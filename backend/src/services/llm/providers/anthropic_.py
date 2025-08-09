from __future__ import annotations
"""Anthropic LLM provider."""

from typing import Dict

from src.config.settings import settings
from src.services.llm.base import BaseLLM
from src.utils.http import HttpClient


class AnthropicLLM(BaseLLM):
    """Minimal Anthropic Claude completion."""

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()
        self.url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": settings.anthropic_api_key or "",
            "anthropic-version": "2023-06-01",
        }

    def complete(self, system_prompt: str, user_prompt: str, stream: bool = False) -> Dict[str, str]:
        payload = {
            "model": "claude-2",
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "stream": stream,
        }
        data = self.client.post_json(self.url, payload, headers=self.headers)
        text = data.get("content", [{}])[0].get("text", "")
        return {"text": text}
