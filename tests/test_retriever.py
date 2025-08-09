import types
from unittest.mock import MagicMock
import os
import sys
from pathlib import Path
import warnings

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
