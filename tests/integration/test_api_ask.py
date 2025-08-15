import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.config.settings import settings


@pytest.fixture()
def client():
    return TestClient(api_main.app)


def test_post_ask_forces_hybrid_and_reranker_flag(client, monkeypatch):
    # Arrange flags
    monkeypatch.setattr(settings, "NEW_PG_RETRIEVAL", True)
    monkeypatch.setattr(settings, "USE_RERANKER", True)

    # Track called params
    called = {}

    async def fake_search_async(query, k=5, filters=None, strategy="auto", use_reranking=False):
        called["strategy"] = strategy
        called["use_reranking"] = use_reranking
        return {"results": [], "metadata": {"strategy": strategy}}

    async def fake_generate_answer(query, context, temperature, max_tokens):
        return {
            "answer": "dummy",
            "sources": [],
            "confidence": 0.5,
            "duration_ms": 1.0,
        }

    monkeypatch.setattr(api_main, "search_service", api_main.search_service)
    monkeypatch.setattr(api_main.search_service, "search_async", fake_search_async)
    monkeypatch.setattr(api_main, "llm_service", api_main.llm_service)
    monkeypatch.setattr(api_main.llm_service, "generate_answer", fake_generate_answer)

    # Act
    resp = client.post(
        "/ask",
        json={
            "query": "Apa isi Pasal 149 ayat (2) UU 4/2009?",
            "context_limit": 3,
            "temperature": 0.2,
            "max_tokens": 256,
        },
    )

    # Assert
    assert resp.status_code == 200
    assert called["strategy"] == "hybrid"
    assert called["use_reranking"] is True


def test_post_ask_auto_when_feature_off(client, monkeypatch):
    monkeypatch.setattr(settings, "NEW_PG_RETRIEVAL", False)
    monkeypatch.setattr(settings, "USE_RERANKER", False)

    called = {}

    async def fake_search_async(query, k=5, filters=None, strategy="auto", use_reranking=False):
        called["strategy"] = strategy
        called["use_reranking"] = use_reranking
        return {"results": [], "metadata": {"strategy": strategy}}

    async def fake_generate_answer(query, context, temperature, max_tokens):
        return {
            "answer": "dummy",
            "sources": [],
            "confidence": 0.5,
            "duration_ms": 1.0,
        }

    monkeypatch.setattr(api_main, "search_service", api_main.search_service)
    monkeypatch.setattr(api_main.search_service, "search_async", fake_search_async)
    monkeypatch.setattr(api_main, "llm_service", api_main.llm_service)
    monkeypatch.setattr(api_main.llm_service, "generate_answer", fake_generate_answer)

    resp = client.post(
        "/ask",
        json={
            "query": "Definisi pertambangan",
            "context_limit": 3,
            "temperature": 0.2,
            "max_tokens": 256,
        },
    )

    assert resp.status_code == 200
    assert called["strategy"] == "auto"
    assert called["use_reranking"] is False
