from __future__ import annotations
"""Gemini LLM provider."""

from typing import Dict

from src.config.settings import settings
from src.services.llm.base import BaseLLM
from src.utils.http import HttpClient


class GeminiLLM(BaseLLM):
    """Minimal Gemini API wrapper."""

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-1.5-flash:generateContent?key=" + (settings.gemini_api_key or "")
        )

    def complete(self, system_prompt: str, user_prompt: str, stream: bool = False) -> Dict[str, str]:
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{system_prompt}\n{user_prompt}"}]}
            ]
        }
        data = self.client.post_json(self.url, payload)
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return {"text": text}
