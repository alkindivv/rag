import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main


@pytest.fixture()
def client():
    return TestClient(api_main.app)


def test_post_search_normalizes_and_preserves_metadata(client, monkeypatch):
    # Arrange: monkeypatch search_service to return dict with results+metadata
    async def fake_search_async(query, k=15, filters=None, strategy="auto", use_reranking=False):
        return {
            "results": [
                {
                    "unit_id": 1,
                    "content": "contoh konten",
                    "citation_string": "UU 4/2009 Pasal 149 ayat (2) huruf b",
                    "score": 0.9,
                    "unit_type": "ayat",
                    "title": None,
                }
            ],
            "metadata": {
                "query": query,
                "strategy": strategy,
                "total_results": 1,
                "limit": k,
                "feature_flags": {
                    "NEW_PG_RETRIEVAL": True,
                    "USE_SQL_FUSION": False,
                    "USE_RERANKER": False,
                },
            },
        }

    monkeypatch.setattr(api_main, "search_service", api_main.search_service)
    monkeypatch.setattr(api_main.search_service, "search_async", fake_search_async)

    # Act
    resp = client.post(
        "/search",
        json={
            "query": "UU 4/2009 Pasal 149 ayat (2) huruf b",
            "limit": 5,
            "strategy": "auto",
        },
    )

    # Assert
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("results"), list)
    assert data["metadata"]["strategy"] == "auto"
    assert data["metadata"]["feature_flags"]["NEW_PG_RETRIEVAL"] in (True, False)
    assert data["results"][0]["citation_string"]


def test_get_search_normalizes_list_response(client, monkeypatch):
    # Arrange: monkeypatch search_service to return list of dicts
    async def fake_search_async(query, k=15, filters=None, strategy="auto", use_reranking=False):
        return [
            {
                "unit_id": 2,
                "content": "hasil list",
                "citation_string": "UU 1/2023 Pasal 1",
                "score": 0.8,
                "unit_type": "pasal",
                "title": None,
            }
        ]

    monkeypatch.setattr(api_main, "search_service", api_main.search_service)
    monkeypatch.setattr(api_main.search_service, "search_async", fake_search_async)

    # Act
    resp = client.get(
        "/search",
        params={"query": "Pasal 1", "limit": 3, "use_reranking": False},
    )

    # Assert
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("results"), list)
    assert data["metadata"]["strategy"] == "auto"
    assert any("citation_string" in hit for hit in data["results"])  
