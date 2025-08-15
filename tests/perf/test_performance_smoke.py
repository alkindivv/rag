import asyncio
import time
from statistics import quantiles

import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main


@pytest.mark.asyncio
async def test_hybrid_performance_smoke(monkeypatch):
    # Monkeypatch hybrid search to simulate vector + bm25 work with bounded latency
    async def fake_search_async(query: str, k: int = 10, filters=None, strategy: str = "auto", use_reranking: bool = False):
        # Simulate bounded latency path representing FTS and vector
        # Keep < 120ms typical for smoke, but add minor jitter
        await asyncio.sleep(0.02)
        return {
            "results": [
                {
                    "unit_id": 1,
                    "content": "konten",
                    "citation_string": "UU 1/2020 Pasal 1",
                    "score": 0.9,
                    "unit_type": "PASAL",
                }
            ] * min(k, 3),
            "metadata": {
                "query": query,
                "strategy": strategy,
                "total_results": min(k, 3),
                "limit": k,
            },
        }

    monkeypatch.setattr(api_main, "search_service", api_main.search_service)
    monkeypatch.setattr(api_main.search_service, "search_async", fake_search_async)

    # Execute multiple requests to estimate p95 under mocked conditions
    client = TestClient(api_main.app)

    durations = []
    for _ in range(20):
        t0 = time.time()
        r = client.post("/search", json={"query": "ekonomi kreatif", "limit": 5, "strategy": "hybrid"})
        assert r.status_code == 200
        durations.append((time.time() - t0) * 1000.0)

    # Compute approximate p95 (discrete via statistics.quantiles)
    durations.sort()
    # p95 as the 95th percentile position
    p95 = durations[int(0.95 * (len(durations) - 1))]

    # Smoke threshold: API overhead + mocked 20ms sleep should be << 200ms
    # Keep generous budget to avoid flakiness on CI
    assert p95 < 200.0, f"p95 too high: {p95:.1f}ms"
