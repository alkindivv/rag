import os
import sys
import types
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import MagicMock
import warnings

from sqlalchemy.sql.elements import TextClause

warnings.filterwarnings("ignore")
os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("JINA_API_KEY", "test")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.services.retriever.hybrid_retriever import (  # noqa: E402
    FTSSearcher,
    VectorSearcher,
    HybridRetriever,
)


class DummyEmbedder:
    def embed_single(self, text: str):
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

    retriever = HybridRetriever()
    assert isinstance(retriever.embedder, DummyEmbedder)


def test_hybrid_retriever_fts_strategy(monkeypatch):
    mock_session = MagicMock()
    mock_session.execute.return_value.fetchall.return_value = []

    @contextmanager
    def dummy_session():
        yield mock_session

    monkeypatch.setattr("src.db.session.get_db_session", dummy_session)
    retriever = HybridRetriever(embedder=DummyEmbedder())
    results = retriever.search("test", strategy="fts")

    assert results == []
    executed_query = mock_session.execute.call_args[0][0]
    assert "legal_units" in executed_query.text
    assert "document_vectors" not in executed_query.text


def test_hybrid_retriever_vector_strategy(monkeypatch):
    mock_session = MagicMock()
    mock_session.execute.return_value.fetchall.return_value = []

    @contextmanager
    def dummy_session():
        yield mock_session

    monkeypatch.setattr("src.db.session.get_db_session", dummy_session)
    retriever = HybridRetriever(embedder=DummyEmbedder())
    results = retriever.search("test", strategy="vector")

    assert results == []
    executed_query = mock_session.execute.call_args[0][0]
    assert "document_vectors" in executed_query.text

