import os
import json
import sys
from pathlib import Path

import httpx

os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("JINA_API_KEY", "test")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.services.embedding.embedder import JinaEmbedder  # noqa: E402
from src.utils.http import HttpClient  # noqa: E402


def test_embed_single_sends_correct_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content.decode())
        data = {"data": [{"embedding": [0.0] * 1024}]}
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(handler)
    client = HttpClient(client=httpx.Client(transport=transport))

    embedder = JinaEmbedder(client=client)
    embedding = embedder.embed_single("hello")

    assert len(embedding) == 1024
    assert captured["url"].endswith("/embeddings")
    assert captured["json"]["model"] == embedder.model_name
    assert captured["json"]["input"] == ["hello"]
    assert "encoding_format" not in captured["json"]

