from __future__ import annotations
"""OpenAI LLM provider."""

from typing import Dict

from src.config.settings import settings
from src.services.llm.base import BaseLLM
from src.utils.http import HttpClient


class OpenAILLM(BaseLLM):
    """Minimal OpenAI chat completion."""

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {settings.openai_api_key}"}

    def complete(self, system_prompt: str, user_prompt: str, stream: bool = False) -> Dict[str, str]:
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": stream,
        }
        data = self.client.post_json(self.url, payload, headers=self.headers)
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"text": text}
