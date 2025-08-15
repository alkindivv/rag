import asyncio
import pytest

from src.services.search.hybrid_search import HybridSearchService
from src.config.settings import settings


class FakeVectorService:
    async def search_async(self, query, k=10, filters=None):
        return {
            "results": [
                {"id": 1, "unit_id": 1, "content": "vector a", "citation_string": "V-1"},
                {"id": 2, "unit_id": 2, "content": "vector b", "citation_string": "V-2"},
            ][:k],
            "metadata": {"search_type": "vector"},
        }

    def search(self, query, k=10, filters=None):
        return asyncio.get_event_loop().run_until_complete(self.search_async(query, k, filters))


class FakeBM25Service:
    async def search_async(self, query, k=10, filters=None):
        return {
            "results": [
                {"id": 2, "unit_id": 2, "content": "bm25 b", "citation_string": "B-2"},
                {"id": 3, "unit_id": 3, "content": "bm25 c", "citation_string": "B-3"},
            ][:k],
            "metadata": {"search_type": "bm25_fts"},
        }

    def search(self, query, k=10, filters=None):
        return asyncio.get_event_loop().run_until_complete(self.search_async(query, k, filters))


@pytest.mark.asyncio
async def test_hybrid_returns_unified_schema(monkeypatch):
    vec = FakeVectorService()
    bm25 = FakeBM25Service()
    svc = HybridSearchService(vector_service=vec, bm25_service=bm25)

    data = await svc.search_async("ekonomi kreatif", k=3, strategy="hybrid")

    assert isinstance(data, dict)
    assert "results" in data and isinstance(data["results"], list)
    assert "metadata" in data and isinstance(data["metadata"], dict)
    assert "feature_flags" in data["metadata"]


@pytest.mark.asyncio
async def test_use_sql_fusion_gates_rrf_and_merges(monkeypatch):
    # Enable SQL fusion
    monkeypatch.setattr(settings, "USE_SQL_FUSION", True)

    vec = FakeVectorService()
    bm25 = FakeBM25Service()
    svc = HybridSearchService(vector_service=vec, bm25_service=bm25)

    data = await svc.search_async("abc", k=10, strategy="hybrid")

    # Should merge and dedupe unit_id=2
    unit_ids = [h.get("unit_id") for h in data["results"]]
    assert unit_ids == [1, 2, 3]
    assert data["metadata"]["feature_flags"]["USE_SQL_FUSION"] is True


@pytest.mark.asyncio
async def test_new_pg_retrieval_routes_explicit_to_explicit_pg(monkeypatch):
    # Ensure SQL fusion off to avoid masking
    monkeypatch.setattr(settings, "USE_SQL_FUSION", False)
    # Turn on new retrieval flag
    monkeypatch.setattr(settings, "NEW_PG_RETRIEVAL", True)

    class SpyBM25(FakeBM25Service):
        def __init__(self):
            self.called = False
        async def search_async(self, query, k=10, filters=None):
            self.called = True
            return await super().search_async(query, k, filters)

    vec = FakeVectorService()
    bm25 = SpyBM25()
    svc = HybridSearchService(vector_service=vec, bm25_service=bm25)

    # Explicit query
    q = "UU 4/2009 Pasal 149 ayat (2) huruf b"
    data = await svc.search_async(q, k=5, strategy="auto")

    # Strategy metadata should reflect explicit_pg routing
    assert data["metadata"]["strategy"] == "explicit_pg"
    # BM25 should not be called when explicit_pg is used
    assert bm25.called is False

    # Reset flag side-effects for other tests
    monkeypatch.setattr(settings, "NEW_PG_RETRIEVAL", False)
