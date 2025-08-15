import asyncio
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(autouse=True)
def patch_search_service(monkeypatch):
    """Patch search_service.search_async to avoid DB/network and return controlled data."""

    async def fake_search_async(query: str, k: int = 10, **kwargs) -> Dict[str, Any]:
        # Return dict shape that services commonly return
        results = [
            {
                "id": 1,
                "unit_id": 1,
                "content": "Contoh isi pasal",
                "citation_string": "UU Nomor 1 Tahun 2020, Pasal 1",
                "score": 0.9,
                "unit_type": "PASAL",
                "doc_form": "UU",
                "doc_year": 2020,
                "doc_number": "1",
                "hierarchy_path": "1.1",
                "metadata": {"search_type": "vector"},
            },
            {
                "id": 2,
                "unit_id": 2,
                "content": "Contoh isi ayat",
                "citation_string": "UU Nomor 1 Tahun 2020, ayat 1",
                "score": 0.8,
                "unit_type": "AYAT",
                "doc_form": "UU",
                "doc_year": 2020,
                "doc_number": "1",
                "hierarchy_path": "1.1.1",
                "metadata": {"search_type": "bm25_fts"},
            },
        ][:k]
        return {"results": results, "metadata": {"search_type": "hybrid", "total_results": len(results)}}

    # Apply monkeypatch
    from src import api as api_pkg
    from src.api import main as main_mod

    monkeypatch.setattr(main_mod.search_service, "search_async", fake_search_async)

    yield


def test_post_search_returns_unified_shape():
    client = TestClient(app)
    payload = {
        "query": "ekonomi kreatif",
        "limit": 2,
        "use_reranking": False,
        "filters": None,
        "strategy": "auto",
    }
    resp = client.post("/search", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Shape assertions
    assert "results" in data and isinstance(data["results"], list)
    assert "metadata" in data and isinstance(data["metadata"], dict)

    # Field presence from unified schema
    hit = data["results"][0]
    assert set(["id", "unit_id", "content", "citation_string"]).issubset(hit.keys())
    assert "feature_flags" in data["metadata"]


def test_get_search_returns_unified_shape():
    client = TestClient(app)
    resp = client.get("/search", params={"query": "ekonomi", "limit": 2, "use_reranking": False})
    assert resp.status_code == 200
    data = resp.json()

    # Shape assertions
    assert "results" in data and isinstance(data["results"], list)
    assert "metadata" in data and isinstance(data["metadata"], dict)

    # Field presence from unified schema
    hit = data["results"][0]
    assert set(["id", "unit_id", "content", "citation_string"]).issubset(hit.keys())
    assert "feature_flags" in data["metadata"]
