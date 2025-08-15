import pytest

from src.services.search.explicit_pg import ExplicitPGService


@pytest.mark.asyncio
async def test_explicit_service_parses_and_returns_unified_schema():
    svc = ExplicitPGService()
    q = "UU 4/2009 Pasal 149 ayat (2) huruf b"
    data = await svc.resolve_by_citation(q, k=5)

    assert isinstance(data, dict)
    assert "results" in data and isinstance(data["results"], list)
    assert "metadata" in data and isinstance(data["metadata"], dict)
    assert data["metadata"]["strategy"] == "explicit_pg"
    assert data["metadata"]["search_type"] == "explicit"
    assert "feature_flags" in data["metadata"]


@pytest.mark.asyncio
async def test_explicit_service_when_not_parseable_returns_empty_unified():
    svc = ExplicitPGService()
    q = "apa itu keadilan?"  # not explicit
    data = await svc.resolve_by_citation(q, k=5)

    assert data["results"] == []
    assert data["metadata"]["strategy"] == "explicit_pg"
