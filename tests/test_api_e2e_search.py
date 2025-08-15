import pytest
from fastapi.testclient import TestClient

from src.api.main import app, search_service
from src.config.settings import settings


@pytest.fixture
def client():
    return TestClient(app)


def test_api_e2e_explicit_query_routes_and_shapes(client, monkeypatch):
    # Enable explicit routing
    monkeypatch.setattr(settings, "NEW_PG_RETRIEVAL", True)
    monkeypatch.setattr(settings, "USE_SQL_FUSION", False)

    async def fake_resolve(query: str, k: int = 10, filters=None):
        return {
            "results": [
                {"id": 10, "unit_id": 10, "content": "ayat 2 huruf b", "citation_string": "UU 4/2009 Pasal 149 ayat (2) huruf b"}
            ],
            "metadata": {
                "query": query,
                "strategy": "explicit_pg",
                "search_type": "explicit",
                "total_results": 1,
                "feature_flags": {
                    "NEW_PG_RETRIEVAL": settings.NEW_PG_RETRIEVAL,
                    "USE_SQL_FUSION": settings.USE_SQL_FUSION,
                    "USE_RERANKER": settings.USE_RERANKER,
                },
            },
        }

    monkeypatch.setattr(search_service.explicit_service, "resolve_by_citation", fake_resolve)

    payload = {"query": "UU 4/2009 Pasal 149 ayat (2) huruf b", "limit": 5, "strategy": "auto"}
    resp = client.post("/search", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "results" in data and isinstance(data["results"], list)
    assert "metadata" in data and data["metadata"]["strategy"] in ("auto", "explicit_pg", "hybrid")


def test_api_e2e_hybrid_sql_fusion_path(client, monkeypatch):
    # Enable SQL fusion
    monkeypatch.setattr(settings, "USE_SQL_FUSION", True)

    async def fake_hybrid(query: str, k: int, filters=None):
        return [
            {"id": 1, "unit_id": 1, "content": "vector a"},
            {"id": 2, "unit_id": 2, "content": "bm25 b"},
        ][:k]

    monkeypatch.setattr(search_service, "_hybrid_search_async", fake_hybrid)

    payload = {"query": "ekonomi kreatif", "limit": 2, "strategy": "hybrid"}
    resp = client.post("/search", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    assert data["metadata"]["feature_flags"]["USE_SQL_FUSION"] is True
