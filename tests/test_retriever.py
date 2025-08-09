import types
from unittest.mock import MagicMock
import os
import sys
from pathlib import Path
import warnings
import pytest

from sqlalchemy.sql.elements import TextClause

warnings.filterwarnings("ignore")
os.environ.setdefault("DATABASE_URL", "sqlite://")
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.services.retriever.hybrid_retriever import FTSSearcher, VectorSearcher


class DummyEmbedder:
    def embed_single(self, text: str):
        # Return 1024-dim zero vector
        return [0.0] * 1024


def test_fts_search_executes_without_format_error():
    mock_session = MagicMock()
    mock_session.execute.return_value.fetchall.return_value = []

    searcher = FTSSearcher(mock_session)
    results = searcher.search("test", limit=5)

    assert results == []
    executed_query = mock_session.execute.call_args[0][0]
    assert isinstance(executed_query, TextClause)


def test_vector_search_executes_without_format_error():
    mock_session = MagicMock()
    mock_session.execute.return_value.fetchall.return_value = []

    searcher = VectorSearcher(mock_session, embedder=DummyEmbedder())
    results = searcher.search("test", limit=5)

    assert results == []
    executed_query = mock_session.execute.call_args[0][0]
    assert isinstance(executed_query, TextClause)
    assert "content_type" not in executed_query.text
    assert "lu.unit_type" in executed_query.text


def test_hybrid_retriever_default_embedder_initialization(monkeypatch):
    dummy_module = types.ModuleType("embedder")
    dummy_module.JinaEmbedder = DummyEmbedder
    monkeypatch.setitem(sys.modules, "src.services.embedding.embedder", dummy_module)

    from src.services.retriever.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    assert isinstance(retriever.embedder, DummyEmbedder)
