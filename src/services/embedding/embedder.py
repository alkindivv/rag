from __future__ import annotations
"""Jina embedding client with batching and retry."""

from typing import List

from src.config.settings import settings
from src.utils.http import HttpClient


class JinaEmbedder:
    """Embed text using Jina API."""

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {"Authorization": f"Bearer {settings.jina_api_key}"}

    def embed(self, texts: List[str], batch_size: int | None = None) -> List[List[float]]:
        """Embed a list of texts."""

        bs = batch_size or settings.embed_batch_size
        all_vecs: List[List[float]] = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            payload = {"model": "jina-embeddings-v2", "input": batch}
            data = self.client.post_json(self.url, payload, headers=self.headers)
            vecs = [d["embedding"] for d in data.get("data", [])]
            all_vecs.extend(vecs)
        return all_vecs
