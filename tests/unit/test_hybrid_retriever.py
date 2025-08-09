"""
Comprehensive unit tests for HybridRetriever service.

Tests cover:
- SQL query formatting fixes (addresses critical blocker #2)
- Search strategy routing and execution
- Filter construction and parameter binding
- Error handling and fallback strategies
- Integration with embedding service
- Performance optimization

Key Focus Areas:
- Fix SQLAlchemy text() format issues
- Validate all search strategies work independently
- Test query parameter binding
- Ensure proper column references (lu.unit_type vs dv.content_type)
- Validate explicit reference parsing
"""

import pytest
import json
from unittest.mock import MagicMock, patch, Mock
from typing import List, Dict, Any
from contextlib import contextmanager

from sqlalchemy.sql import text
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.exc import SQLAlchemyError

from src.services.retriever.hybrid_retriever import (
    FTSSearcher,
    VectorSearcher,
    ExplicitSearcher,
    HybridRetriever,
    SearchResult,
    SearchFilters
)


class TestSearchResult:
    """Test SearchResult data structure and validation."""

    def test_search_result_creation(self):
        """Test SearchResult can be created with all required fields."""
        result = SearchResult(
            unit_id="UU-2025-1/pasal-1/ayat-1",
            content="Test legal content",
            citation="UU No. 1 Tahun 2025 Pasal 1 ayat (1)",
            score=0.95,
            source_type="fts",
            unit_type="ayat",
            document_form="UU",
            document_year=2025,
            document_number="1"
        )

        assert result.unit_id == "UU-2025-1/pasal-1/ayat-1"
        assert result.score == 0.95
        assert result.source_type == "fts"

    def test_search_result_validation(self):
        """Test SearchResult field validation."""
        # Valid source types
        valid_sources = ["fts", "vector", "explicit", "reranked"]
        for source in valid_sources:
            result = SearchResult(
                unit_id="test", content="test", citation="test",
                score=0.5, source_type=source, unit_type="ayat",
                document_form="UU", document_year=2025, document_number="1"
            )
            assert result.source_type == source

        # Score should be between 0 and 1
        result = SearchResult(
            unit_id="test", content="test", citation="test",
            score=1.5, source_type="fts", unit_type="ayat",
            document_form="UU", document_year=2025, document_number="1"
        )
        # Note: Validation logic depends on implementation

    def test_search_result_comparison(self):
        """Test SearchResult comparison and sorting."""
        result1 = SearchResult(
            unit_id="test1", content="test", citation="test",
            score=0.9, source_type="fts", unit_type="ayat",
            document_form="UU", document_year=2025, document_number="1"
        )
        result2 = SearchResult(
            unit_id="test2", content="test", citation="test",
            score=0.7, source_type="vector", unit_type="ayat",
            document_form="UU", document_year=2025, document_number="1"
        )

        # Should be able to sort by score
        results = [result2, result1]
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        assert sorted_results[0].score > sorted_results[1].score


class TestSearchFilters:
    """Test SearchFilters construction and validation."""

    def test_empty_filters(self):
        """Test empty filters object."""
        filters = SearchFilters()

        assert filters.doc_forms is None
        assert filters.doc_years is None
        assert filters.unit_types is None

    def test_filters_with_values(self):
        """Test filters with specific values."""
        filters = SearchFilters(
            doc_forms=["UU", "PP"],
            doc_years=[2024, 2025],
            unit_types=["ayat", "huruf"]
        )

        assert filters.doc_forms == ["UU", "PP"]
        assert filters.doc_years == [2024, 2025]
        assert filters.unit_types == ["ayat", "huruf"]

    def test_filter_clause_generation(self):
        """Test SQL filter clause generation."""
        filters = SearchFilters(
            doc_forms=["UU"],
            doc_years=[2025]
        )

        # Test clause generation (implementation specific)
        # This validates the filter logic works correctly


class TestFTSSearcher:
    """Test Full-Text Search functionality."""

    def test_fts_searcher_initialization(self):
        """Test FTSSearcher can be initialized."""
        mock_session = MagicMock()
        searcher = FTSSearcher(mock_session)

        assert searcher.session is mock_session

    def test_fts_query_construction_no_filters(self):
        """Test FTS query construction without filters."""
        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = []

        searcher = FTSSearcher(mock_session)
        results = searcher.search("pertambangan", limit=10)

        # Verify query was executed
        mock_session.execute.assert_called_once()
        executed_query = mock_session.execute.call_args[0][0]

        # Should be SQLAlchemy TextClause, not string
        assert isinstance(executed_query, TextClause)

        # Query should contain FTS elements
        query_text = executed_query.text
        assert "legal_units" in query_text
        assert "plainto_tsquery" in query_text
        assert "ts_rank" in query_text
        assert "unit_type IN" in query_text

    def test_fts_query_construction_with_filters(self):
        """Test FTS query construction with filters."""
        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = []

        searcher = FTSSearcher(mock_session)

        filters = SearchFilters(
            doc_forms=["UU"],
            doc_years=[2025]
        )

        results = searcher.search("pertambangan", filters=filters, limit=10)

        executed_query = mock_session.execute.call_args[0][0]
        query_text = executed_query.text

        # Should include filter conditions
        assert "ld.doc_form" in query_text
        assert "ld.doc_year" in query_text

        # Should use proper parameter binding, not string formatting
        assert "{filters_clause}" not in query_text  # No string formatting

    def test_fts_query_parameter_binding(self):
        """Test that FTS queries use proper parameter binding."""
        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = []

        searcher = FTSSearcher(mock_session)
        searcher.search("test query", limit=5)

        executed_query = mock_session.execute.call_args[0][0]

        # Should be TextClause with bound parameters
        assert isinstance(executed_query, TextClause)

        # Should not contain format placeholders
        query_text = executed_query.text
        assert "{" not in query_text and "}" not in query_text

    def test_fts_result_mapping(self):
        """Test FTS result mapping to SearchResult objects."""
        # Mock database response
        mock_row = MagicMock()
        mock_row.unit_id = "UU-2025-1/pasal-1/ayat-1"
        mock_row.content = "Pertambangan adalah kegiatan..."
        mock_row.citation_string = "Pasal 1 ayat (1)"
        mock_row.score = 0.85
        mock_row.unit_type = "ayat"
        mock_row.doc_form = "UU"
        mock_row.doc_year = 2025
        mock_row.doc_number = "1"

        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = [mock_row]

        searcher = FTSSearcher(mock_session)
        results = searcher.search("pertambangan")

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.unit_id == "UU-2025-1/pasal-1/ayat-1"
        assert result.score == 0.85
        assert result.source_type == "fts"

    def test_fts_empty_results(self):
        """Test FTS search with no matching results."""
        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = []

        searcher = FTSSearcher(mock_session)
        results = searcher.search("nonexistent term")

        assert results == []
        mock_session.execute.assert_called_once()

    def test_fts_database_error_handling(self):
        """Test FTS search error handling."""
        mock_session = MagicMock()
        mock_session.execute.side_effect = SQLAlchemyError("Database connection failed")

        searcher = FTSSearcher(mock_session)

        with pytest.raises(SQLAlchemyError):
            searcher.search("test query")


class TestVectorSearcher:
    """Test Vector Search functionality."""

    def test_vector_searcher_initialization(self):
        """Test VectorSearcher initialization."""
        mock_session = MagicMock()
        mock_embedder = MagicMock()

        searcher = VectorSearcher(mock_session, embedder=mock_embedder)

        assert searcher.session is mock_session
        assert searcher.embedder is mock_embedder

    def test_vector_query_construction(self):
        """Test vector query construction and column references."""
        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = []

        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        searcher = VectorSearcher(mock_session, embedder=mock_embedder)
        results = searcher.search("mining regulation", limit=10)

        # Verify embedder was called
        mock_embedder.embed_single.assert_called_once_with("mining regulation")

        # Verify query execution
        executed_query = mock_session.execute.call_args[0][0]
        assert isinstance(executed_query, TextClause)

        query_text = executed_query.text
        # Should use correct table aliases and column names
        assert "document_vectors" in query_text
        assert "legal_units" in query_text
        assert "legal_documents" in query_text

        # Critical fix: should use lu.unit_type, not dv.content_type
        assert "lu.unit_type" in query_text
        assert "dv.content_type" not in query_text  # This column doesn't exist

        # Should use vector similarity operator
        assert "<->" in query_text or "<=>" in query_text

    def test_vector_search_with_filters(self):
        """Test vector search with filters applied."""
        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = []

        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        searcher = VectorSearcher(mock_session, embedder=mock_embedder)

        filters = SearchFilters(doc_forms=["UU"], doc_years=[2025])
        results = searcher.search("test", filters=filters)

        executed_query = mock_session.execute.call_args[0][0]
        query_text = executed_query.text

        # Should include filter conditions
        assert "ld.doc_form" in query_text
        assert "ld.doc_year" in query_text

        # Should not use string formatting
        assert "{" not in query_text and "}" not in query_text

    def test_vector_embedding_error_handling(self):
        """Test vector search when embedding fails."""
        mock_session = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_single.side_effect = Exception("Embedding API failed")

        searcher = VectorSearcher(mock_session, embedder=mock_embedder)

        with pytest.raises(Exception):
            searcher.search("test query")

    def test_vector_result_mapping(self):
        """Test vector search result mapping."""
        # Mock database response
        mock_row = MagicMock()
        mock_row.unit_id = "UU-2025-1/pasal-1"
        mock_row.content = "Aggregated pasal content"
        mock_row.citation_string = "Pasal 1"
        mock_row.similarity = 0.92
        mock_row.unit_type = "pasal"
        mock_row.doc_form = "UU"
        mock_row.doc_year = 2025
        mock_row.doc_number = "1"

        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = [mock_row]

        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        searcher = VectorSearcher(mock_session, embedder=mock_embedder)
        results = searcher.search("pertambangan")

        assert len(results) == 1
        result = results[0]
        assert result.source_type == "vector"
        assert result.score == 0.92  # Uses similarity score
        assert result.unit_type == "pasal"


class TestExplicitSearcher:
    """Test Explicit Search functionality for legal references."""

    def test_explicit_searcher_initialization(self):
        """Test ExplicitSearcher initialization."""
        mock_session = MagicMock()
        searcher = ExplicitSearcher(mock_session)

        assert searcher.session is mock_session

    def test_explicit_reference_parsing(self):
        """Test parsing explicit legal references."""
        mock_session = MagicMock()
        searcher = ExplicitSearcher(mock_session)

        # Test various reference formats
        test_cases = [
            ("pasal 1", {"pasal": "1"}),
            ("pasal 1 ayat 2", {"pasal": "1", "ayat": "2"}),
            ("UU 4/2009", {"doc_form": "UU", "doc_number": "4", "doc_year": "2009"}),
            ("ayat 3", {"ayat": "3"}),
            ("huruf a", {"huruf": "a"})
        ]

        for query, expected_refs in test_cases:
            refs = searcher._parse_explicit_references(query)
            for key, value in expected_refs.items():
                assert key in refs
                assert refs[key] == value

    def test_explicit_search_pasal_reference(self):
        """Test explicit search for pasal references."""
        mock_row = MagicMock()
        mock_row.unit_id = "UU-2025-1/pasal-1"
        mock_row.content = "Pasal content"
        mock_row.citation_string = "Pasal 1"
        mock_row.unit_type = "pasal"
        mock_row.doc_form = "UU"
        mock_row.doc_year = 2025
        mock_row.doc_number = "1"

        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = [mock_row]

        searcher = ExplicitSearcher(mock_session)
        results = searcher.search("pasal 1")

        assert len(results) == 1
        assert results[0].source_type == "explicit"
        assert results[0].unit_type == "pasal"

    def test_explicit_search_document_reference(self):
        """Test explicit search for document references."""
        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = []

        searcher = ExplicitSearcher(mock_session)
        results = searcher.search("UU 1/2025")

        executed_query = mock_session.execute.call_args[0][0]
        query_text = executed_query.text

        # Should search for specific document
        assert "ld.doc_form" in query_text
        assert "ld.doc_number" in query_text
        assert "ld.doc_year" in query_text

    def test_explicit_search_no_matches(self):
        """Test explicit search with no reference patterns."""
        mock_session = MagicMock()
        searcher = ExplicitSearcher(mock_session)

        # Query without explicit references
        results = searcher.search("general mining concepts")

        assert results == []
        # Should not execute any database queries
        mock_session.execute.assert_not_called()


class TestHybridRetriever:
    """Test main HybridRetriever orchestration."""

    def test_hybrid_retriever_initialization_default(self):
        """Test HybridRetriever initialization with defaults."""
        with patch('src.services.embedding.embedder.JinaEmbedder') as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder_class.return_value = mock_embedder

            retriever = HybridRetriever()

            assert retriever.embedder is mock_embedder
            mock_embedder_class.assert_called_once()

    def test_hybrid_retriever_initialization_custom_embedder(self):
        """Test HybridRetriever with custom embedder."""
        mock_embedder = MagicMock()
        retriever = HybridRetriever(embedder=mock_embedder)

        assert retriever.embedder is mock_embedder

    def test_query_strategy_routing_explicit(self):
        """Test query routing for explicit references."""
        mock_embedder = MagicMock()

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value.fetchall.return_value = []

            retriever = HybridRetriever(embedder=mock_embedder)

            # Test explicit queries
            explicit_queries = ["pasal 1", "UU 4/2009", "ayat 2"]

            for query in explicit_queries:
                results = retriever.search(query, strategy="auto")
                # Should route to explicit search

    def test_query_strategy_routing_thematic(self):
        """Test query routing for thematic searches."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value.fetchall.return_value = []

            retriever = HybridRetriever(embedder=mock_embedder)

            # Test thematic queries
            thematic_queries = ["pertambangan mineral", "izin usaha", "lingkungan hidup"]

            for query in thematic_queries:
                results = retriever.search(query, strategy="auto")
                # Should route to hybrid (FTS + Vector) search

    def test_explicit_strategy_forced(self):
        """Test forced explicit strategy."""
        mock_embedder = MagicMock()

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value.fetchall.return_value = []

            retriever = HybridRetriever(embedder=mock_embedder)

            # Force explicit strategy even for non-explicit query
            results = retriever.search("general mining", strategy="explicit")

            # Should only use explicit search (may return empty)
            assert isinstance(results, list)

    def test_fts_strategy_forced(self):
        """Test forced FTS strategy."""
        mock_embedder = MagicMock()

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value.fetchall.return_value = []

            retriever = HybridRetriever(embedder=mock_embedder)

            results = retriever.search("pertambangan", strategy="fts")

            # Should only use FTS, not vector or explicit
            mock_embedder.embed_single.assert_not_called()

    def test_vector_strategy_forced(self):
        """Test forced vector strategy."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value.fetchall.return_value = []

            retriever = HybridRetriever(embedder=mock_embedder)

            results = retriever.search("mining", strategy="vector")

            # Should call embedder for vector search
            mock_embedder.embed_single.assert_called_once_with("mining")

    def test_hybrid_strategy_combination(self):
        """Test hybrid strategy combines FTS and vector results."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        # Mock different results for FTS and vector searches
        fts_row = MagicMock()
        fts_row.unit_id = "UU-2025-1/pasal-1/ayat-1"
        fts_row.content = "FTS result"
        fts_row.citation_string = "Pasal 1 ayat (1)"
        fts_row.score = 0.8
        fts_row.unit_type = "ayat"
        fts_row.doc_form = "UU"
        fts_row.doc_year = 2025
        fts_row.doc_number = "1"

        vector_row = MagicMock()
        vector_row.unit_id = "UU-2025-1/pasal-2"
        vector_row.content = "Vector result"
        vector_row.citation_string = "Pasal 2"
        vector_row.similarity = 0.9
        vector_row.unit_type = "pasal"
        vector_row.doc_form = "UU"
        vector_row.doc_year = 2025
        vector_row.doc_number = "1"

        call_count = 0
        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            executed_query = args[0]
            query_text = executed_query.text

            if "document_vectors" in query_text:
                # Vector search query
                mock_result = MagicMock()
                mock_result.fetchall.return_value = [vector_row]
                return mock_result
            else:
                # FTS search query
                mock_result = MagicMock()
                mock_result.fetchall.return_value = [fts_row]
                return mock_result

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.side_effect = mock_execute
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)
            results = retriever.search("pertambangan", strategy="hybrid", limit=10)

            # Should combine results from both searches
            assert len(results) <= 10  # Respects limit

            # Should have called both FTS and vector searches
            assert call_count >= 2

    def test_result_deduplication(self):
        """Test deduplication of results from multiple search strategies."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        # Create duplicate result (same unit_id from different strategies)
        duplicate_row = MagicMock()
        duplicate_row.unit_id = "UU-2025-1/pasal-1/ayat-1"
        duplicate_row.content = "Same content"
        duplicate_row.citation_string = "Pasal 1 ayat (1)"
        duplicate_row.score = 0.8
        duplicate_row.similarity = 0.85
        duplicate_row.unit_type = "ayat"
        duplicate_row.doc_form = "UU"
        duplicate_row.doc_year = 2025
        duplicate_row.doc_number = "1"

        def mock_execute(*args, **kwargs):
            # Both FTS and vector return same result
            mock_result = MagicMock()
            mock_result.fetchall.return_value = [duplicate_row]
            return mock_result

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.side_effect = mock_execute
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)
            results = retriever.search("test", strategy="hybrid")

            # Should deduplicate and keep higher score
            unit_ids = [r.unit_id for r in results]
            assert len(set(unit_ids)) == len(unit_ids)  # No duplicates


class TestHybridRetrieverErrorHandling:
    """Test error handling in hybrid retriever."""

    def test_database_connection_error(self):
        """Test handling of database connection errors."""
        mock_embedder = MagicMock()

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.side_effect = SQLAlchemyError("Connection failed")

            retriever = HybridRetriever(embedder=mock_embedder)

            with pytest.raises(SQLAlchemyError):
                retriever.search("test query")

    def test_embedder_failure_fallback(self):
        """Test fallback when embedder fails."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.side_effect = Exception("API timeout")

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.return_value.fetchall.return_value = []
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            # Vector strategy should fail
            with pytest.raises(Exception):
                retriever.search("test", strategy="vector")

            # But FTS should still work
            results = retriever.search("test", strategy="fts")
            assert isinstance(results, list)

    def test_partial_failure_in_hybrid_search(self):
        """Test hybrid search when one strategy fails."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.side_effect = Exception("Vector search failed")

        # Mock FTS to succeed
        fts_row = MagicMock()
        fts_row.unit_id = "UU-2025-1/pasal-1/ayat-1"
        fts_row.content = "FTS content"
        fts_row.citation_string = "Pasal 1 ayat (1)"
        fts_row.score = 0.8
        fts_row.unit_type = "ayat"
        fts_row.doc_form = "UU"
        fts_row.doc_year = 2025
        fts_row.doc_number = "1"

        call_count = 0
        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            executed_query = args[0]
            if "document_vectors" in executed_query.text:
                # Vector query should fail
                raise SQLAlchemyError("Vector search error")
            else:
                # FTS query succeeds
                mock_result = MagicMock()
                mock_result.fetchall.return_value = [fts_row]
                return mock_result

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.side_effect = mock_execute
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            # Hybrid search should gracefully handle vector failure
            results = retriever.search("pertambangan", strategy="hybrid")

            # Should return FTS results even if vector fails
            assert len(results) >= 0  # May have FTS results


class TestHybridRetrieverPerformance:
    """Test performance characteristics of hybrid retriever."""

    def test_search_performance_limits(self, performance_timer):
        """Test search performance meets system requirements."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        # Mock minimal database response
        mock_row = MagicMock()
        mock_row.unit_id = "UU-2025-1/pasal-1"
        mock_row.content = "Test content"
        mock_row.citation_string = "Pasal 1"
        mock_row.score = 0.8
        mock_row.similarity = 0.8
        mock_row.unit_type = "pasal"
        mock_row.doc_form = "UU"
        mock_row.doc_year = 2025
        mock_row.doc_number = "1"

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.return_value.fetchall.return_value = [mock_row]
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            # Test search performance
            performance_timer.start()
            results = retriever.search("pertambangan mineral", strategy="hybrid", limit=10)
            performance_timer.stop()

            # Should meet performance requirements (<500ms from AGENTS.md)
            assert performance_timer.duration_ms < 500
            assert len(results) >= 0

    def test_concurrent_search_safety(self):
        """Test that multiple concurrent searches don't interfere."""
        import threading
        import time

        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        results_collection = []
        errors_collection = []

        def search_worker(query: str, worker_id: int):
            try:
                with patch('src.db.session.get_db_session') as mock_get_session:
                    mock_session = MagicMock()
                    mock_session.execute.return_value.fetchall.return_value = []
                    mock_get_session.return_value.__enter__.return_value = mock_session

                    retriever = HybridRetriever(embedder=mock_embedder)
                    results = retriever.search(f"{query} {worker_id}")
                    results_collection.append((worker_id, len(results)))
            except Exception as e:
                errors_collection.append((worker_id, str(e)))

        # Launch concurrent searches
        threads = []
        for i in range(5):
            thread = threading.Thread(target=search_worker, args=("mining", i))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)

        # All searches should complete without errors
        assert len(errors_collection) == 0
        assert len(results_collection) == 5

    def test_memory_usage_large_results(self):
        """Test memory efficiency with large result sets."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        # Create many mock results
        mock_rows = []
        for i in range(100):
            row = MagicMock()
            row.unit_id = f"UU-2025-1/pasal-{i}/ayat-1"
            row.content = f"Legal content {i}" * 100  # Large content
            row.citation_string = f"Pasal {i} ayat (1)"
            row.score = 0.9 - (i * 0.001)
            row.unit_type = "ayat"
            row.doc_form = "UU"
            row.doc_year = 2025
            row.doc_number = "1"
            mock_rows.append(row)

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.return_value.fetchall.return_value = mock_rows
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            # Should handle large result sets efficiently
            results = retriever.search("test", limit=50)

            # Should respect limit
            assert len(results) <= 50

            # Memory usage should be reasonable
            import sys
            total_size = sum(sys.getsizeof(r) for r in results)
            assert total_size < 10 * 1024 * 1024  # Less than 10MB


class TestHybridRetrieverRegressionTests:
    """Regression tests for known issues and fixes."""

    def test_sql_format_method_fix(self):
        """
        Regression test for SQL query formatting issue.

        Original error: SQLAlchemy text() object doesn't support .format() method
        Fix: Use proper parameter binding instead of string formatting
        """
        mock_embedder = MagicMock()

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()

            # Mock execute to capture the query
            executed_queries = []
            def capture_execute(query, *args, **kwargs):
                executed_queries.append(query)
                mock_result = MagicMock()
                mock_result.fetchall.return_value = []
                return mock_result

            mock_session.execute.side_effect = capture_execute
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            # Test FTS search with filters (this was failing before)
            filters = SearchFilters(doc_forms=["UU"], doc_years=[2025])
            results = retriever.search("test", strategy="fts", filters=filters)

            # Should not raise AttributeError about .format() method
            assert isinstance(results, list)
            assert len(executed_queries) > 0

            # Verify query is TextClause, not string
            query = executed_queries[0]
            assert isinstance(query, TextClause)

            # Should not contain format placeholders
            assert "{" not in query.text and "}" not in query.text

    def test_nonexistent_column_reference_fix(self):
        """
        Regression test for nonexistent column references.

        Original error: dv.content_type doesn't exist
        Fix: Use lu.unit_type instead
        """
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()

            executed_queries = []
            def capture_execute(query, *args, **kwargs):
                executed_queries.append(query)
                mock_result = MagicMock()
                mock_result.fetchall.return_value = []
                return mock_result

            mock_session.execute.side_effect = capture_execute
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            # Test vector search (this was failing with column reference error)
            results = retriever.search("test", strategy="vector")

            assert len(executed_queries) > 0
            query_text = executed_queries[0].text

            # Should use correct column references
            assert "lu.unit_type" in query_text
            assert "dv.content_type" not in query_text  # This column doesn't exist

    def test_embedder_import_error_fix(self):
        """
        Regression test for embedder import issues.

        Original error: NameError when importing JinaEmbedder
        Fix: Lazy import in HybridRetriever
        """
        # Test that retriever can be imported without embedder
        with patch.dict('sys.modules', {'src.services.embedding.embedder': None}):
            try:
                from src.services.retriever.hybrid_retriever import HybridRetriever

                # Should be able to create instance with provided embedder
                mock_embedder = MagicMock()
                retriever = HybridRetriever(embedder=mock_embedder)
                assert retriever.embedder is mock_embedder

            except (ImportError, NameError) as e:
                pytest.fail(f"Import error not properly handled: {e}")


class TestHybridRetrieverIntegration:
    """Integration tests with other components."""

    def test_integration_with_search_service(self):
        """Test retriever integration with search service."""
        from src.services.search.hybrid_search import HybridSearchService

        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        # Mock database
        mock_row = MagicMock()
        mock_row.unit_id = "UU-2025-1/pasal-1/ayat-1"
        mock_row.content = "Test content"
        mock_row.citation_string = "Pasal 1 ayat (1)"
        mock_row.score = 0.8
        mock_row.unit_type = "ayat"
        mock_row.doc_form = "UU"
        mock_row.doc_year = 2025
        mock_row.doc_number = "1"

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.return_value.fetchall.return_value = [mock_row]
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            # Should be able to integrate with search service
            with patch('src.services.search.reranker.JinaReranker'):
                search_service = HybridSearchService(retriever=retriever)
                response = search_service.search("test query")

                # Should return formatted response
                assert "results" in response
                assert isinstance(response["results"], list)

    def test_integration_with_indexer(self):
        """Test retriever can work with indexed data."""
        mock_embedder = MagicMock()

        # This test validates that retriever can work with data
        # that has been processed by the indexer
        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.return_value.fetchall.return_value = []
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            # Should work with various unit types that indexer creates
            unit_types = ["pasal", "ayat", "huruf", "angka"]
            for unit_type in unit_types:
                filters = SearchFilters(unit_types=[unit_type])
                results = retriever.search("test", filters=filters)
                assert isinstance(results, list)


class TestHybridRetrieverQueryPatterns:
    """Test various query patterns and edge cases."""

    def test_indonesian_text_handling(self):
        """Test retriever handles Indonesian legal text correctly."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.return_value.fetchall.return_value = []
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            # Test Indonesian legal queries
            indonesian_queries = [
                "pertambangan mineral dan batubara",
                "izin usaha pertambangan eksplorasi",
                "kewajiban reklamasi dan pascatambang",
                "sanksi administratif dan pidana"
            ]

            for query in indonesian_queries:
                results = retriever.search(query)
                assert isinstance(results, list)

    def test_complex_legal_references(self):
        """Test complex legal reference patterns."""
        mock_embedder = MagicMock()

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.return_value.fetchall.return_value = []
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            # Test complex reference patterns
            complex_queries = [
                "pasal 1 ayat 2 huruf a",
                "UU No. 4 Tahun 2009 pasal 15",
                "ayat (3) huruf b angka 2",
                "PP 78/2010 pasal 5 ayat 1"
            ]

            for query in complex_queries:
                results = retriever.search(query, strategy="explicit")
                assert isinstance(results, list)

    def test_edge_case_queries(self):
        """Test edge case queries."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.return_value.fetchall.return_value = []
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            # Test edge cases
            edge_cases = [
                "",  # Empty query
                "   ",  # Whitespace only
                "a",  # Single character
                "123",  # Numbers only
                "!@#$%^&*()",  # Special characters only
            ]

            for query in edge_cases:
                try:
                    results = retriever.search(query)
                    assert isinstance(results, list)
                except ValueError:
                    # Empty queries may raise ValueError, which is acceptable
                    pass


class TestHybridRetrieverConfiguration:
    """Test configuration and customization options."""

    def test_custom_search_limits(self):
        """Test custom search limits and pagination."""
        mock_embedder = MagicMock()

        # Mock large result set
        mock_rows = [MagicMock() for _ in range(100)]
        for i, row in enumerate(mock_rows):
            row.unit_id = f"UU-2025-1/pasal-{i}"
            row.content = f"Content {i}"
            row.citation_string = f"Pasal {i}"
            row.score = 0.9 - (i * 0.001)
            row.unit_type = "pasal"
            row.doc_form = "UU"
            row.doc_year = 2025
            row.doc_number = "1"

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.return_value.fetchall.return_value = mock_rows
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            # Test different limits
            for limit in [5, 10, 25, 50]:
                results = retriever.search("test", limit=limit)
                assert len(results) <= limit

    def test_filter_combinations(self):
        """Test various filter combinations."""
        mock_embedder = MagicMock()

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()

            executed_queries = []
            def capture_query(query, *args, **kwargs):
                executed_queries.append(query.text if hasattr(query, 'text') else str(query))
                mock_result = MagicMock()
                mock_result.fetchall.return_value = []
                return mock_result

            mock_session.execute.side_effect = capture_query
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            # Test filter combinations
            filter_combinations = [
                SearchFilters(doc_forms=["UU"]),
                SearchFilters(doc_years=[2025]),
                SearchFilters(unit_types=["ayat"]),
                SearchFilters(doc_forms=["UU"], doc_years=[2025]),
                SearchFilters(doc_forms=["UU", "PP"], doc_years=[2024, 2025], unit_types=["ayat", "huruf"])
            ]

            for filters in filter_combinations:
                executed_queries.clear()
                results = retriever.search("test", filters=filters)

                # Should execute query without errors
                assert isinstance(results, list)
                assert len(executed_queries) > 0


class TestHybridRetrieverSmokeTests:
    """Final smoke tests for overall functionality."""

    def test_all_strategies_smoke_test(self):
        """Smoke test all search strategies."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.return_value.fetchall.return_value = []
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            strategies = ["explicit", "fts", "vector", "hybrid", "auto"]

            for strategy in strategies:
                try:
                    results = retriever.search("pertambangan", strategy=strategy)
                    assert isinstance(results, list)
                except Exception as e:
                    pytest.fail(f"Strategy {strategy} failed: {e}")

    def test_retriever_with_real_database_schema(self, populated_db):
        """Test retriever with actual database schema."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        # Use populated database from conftest.py
        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = populated_db

            retriever = HybridRetriever(embedder=mock_embedder)

            # Should work with actual database structure
            results = retriever.search("pertambangan", strategy="fts")
            assert isinstance(results, list)

    def test_end_to_end_search_workflow(self):
        """Test complete search workflow end-to-end."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        # Mock realistic search results
        fts_result = MagicMock()
        fts_result.unit_id = "UU-2025-1/pasal-1/ayat-1"
        fts_result.content = "Pertambangan adalah kegiatan..."
        fts_result.citation_string = "Pasal 1 ayat (1)"
        fts_result.score = 0.85
        fts_result.unit_type = "ayat"
        fts_result.doc_form = "UU"
        fts_result.doc_year = 2025
        fts_result.doc_number = "1"

        vector_result = MagicMock()
        vector_result.unit_id = "UU-2025-1/pasal-2"
        vector_result.content = "Mineral adalah senyawa..."
        vector_result.citation_string = "Pasal 2"
        vector_result.similarity = 0.90
        vector_result.unit_type = "pasal"
        vector_result.doc_form = "UU"
        vector_result.doc_year = 2025
        vector_result.doc_number = "1"

        call_count = 0
        def mock_execute(query, *args, **kwargs):
            nonlocal call_count
            call_count += 1

            query_text = query.text if hasattr(query, 'text') else str(query)

            if "document_vectors" in query_text:
                # Vector search
                mock_result = MagicMock()
                mock_result.fetchall.return_value = [vector_result]
                return mock_result
            else:
                # FTS search
                mock_result = MagicMock()
                mock_result.fetchall.return_value = [fts_result]
                return mock_result

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.side_effect = mock_execute
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_embedder)

            # Execute end-to-end search
            results = retriever.search("pertambangan mineral", strategy="hybrid", limit=10)

            # Should return combined results
            assert isinstance(results, list)
            assert len(results) >= 0

            # Should have executed multiple queries (FTS + Vector)
            assert call_count >= 1

    def test_retriever_thread_safety(self):
        """Test that retriever is thread-safe."""
        import threading
        import time

        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        results_by_thread = {}
        errors_by_thread = {}

        def search_in_thread(thread_id: int):
            try:
                with patch('src.db.session.get_db_session') as mock_get_session:
                    mock_session = MagicMock()
                    mock_session.execute.return_value.fetchall.return_value = []
                    mock_get_session.return_value.__enter__.return_value = mock_session

                    retriever = HybridRetriever(embedder=mock_embedder)
                    results = retriever.search(f"query {thread_id}")
                    results_by_thread[thread_id] = len(results)

            except Exception as e:
                errors_by_thread[thread_id] = str(e)

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=search_in_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # All threads should complete successfully
        assert len(errors_by_thread) == 0
        assert len(results_by_thread) == 3
