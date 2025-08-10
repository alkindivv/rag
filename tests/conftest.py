"""
Pytest configuration and shared fixtures for Legal RAG System.

This module provides comprehensive testing setup including:
- Database fixtures with proper isolation
- Mock external services (Jina, LLM providers)
- Sample legal document data
- Environment configuration for tests
- Common testing utilities

Key Design Principles:
- Test isolation: Each test gets clean database state
- Mock external dependencies for reliability
- Realistic test data that mirrors production
- Fast execution without network calls
"""

import os
import sys
import json
import tempfile
import warnings
from pathlib import Path
from typing import Generator, Dict, Any, List
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

import pytest
import httpx
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment variables before any imports
os.environ.update({
    "DATABASE_URL": "sqlite:///:memory:",
    "JINA_API_KEY": "test-key",
    "GEMINI_API_KEY": "test-key",
    "OPENAI_API_KEY": "test-key",
    "ANTHROPIC_API_KEY": "test-key",
    "LOG_LEVEL": "WARNING",  # Reduce noise during tests
    "EMBED_BATCH_SIZE": "2",  # Small batches for testing
    "RERANK_PROVIDER": "noop",  # Use noop reranker for tests
})

# Import after environment setup
from src.config.settings import settings
from src.db.models import Base, LegalDocument, LegalUnit, DocumentVector, Subject
from src.db.session import get_db_session
from src.utils.http import HttpClient


# ================================
# DATABASE FIXTURES
# ================================

@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine with in-memory SQLite."""
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False  # Set to True for SQL debugging
    )

    # Create all tables
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture(scope="session")
def test_session_factory(test_engine):
    """Create session factory for test database."""
    return sessionmaker(bind=test_engine)


@pytest.fixture
def db_session(test_session_factory) -> Generator[Session, None, None]:
    """
    Provide clean database session for each test.

    Features:
    - Fresh session per test
    - Automatic rollback after test
    - Proper cleanup
    """
    session = test_session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def mock_db_session(db_session):
    """Mock the global database session getter."""
    @contextmanager
    def _get_session():
        yield db_session

    with patch('src.db.session.get_db_session', _get_session):
        yield db_session


# ================================
# MOCK EXTERNAL SERVICES
# ================================

@pytest.fixture
def mock_jina_embedder():
    """Mock Jina embedding service."""
    embedder = MagicMock()
    embedder.embed_single.return_value = [0.1] * 1024
    embedder.embed_batch.return_value = [[0.1] * 1024, [0.2] * 1024]
    embedder.model_name = "test-model"
    return embedder


@pytest.fixture
def mock_jina_reranker():
    """Mock Jina reranking service."""
    reranker = MagicMock()
    # Mock rerank to return results with modified scores
    def mock_rerank(query: str, results: List, limit: int = None):
        limited_results = results[:limit] if limit else results
        for i, result in enumerate(limited_results):
            result.score = 0.9 - (i * 0.1)  # Decreasing scores
        return limited_results

    reranker.rerank.side_effect = mock_rerank
    return reranker


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for external API calls."""
    def create_mock_response(status_code: int = 200, json_data: Dict = None):
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = json_data or {}
        response.raise_for_status.return_value = None
        return response

    client = MagicMock(spec=HttpClient)
    client.get.return_value = create_mock_response()
    client.post.return_value = create_mock_response()
    return client


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing Q&A functionality."""
    provider = MagicMock()
    provider.generate.return_value = {
        "answer": "Test answer based on provided context.",
        "citations": ["UU-2025-2/pasal-1/ayat-1"],
        "confidence": 0.85
    }
    return provider


# ================================
# TEST DATA FIXTURES
# ================================

@pytest.fixture
def sample_legal_document():
    """Sample legal document for testing."""
    return {
        "id": "test-uuid-1",
        "doc_id": "UU-2025-1",
        "doc_form": "UU",
        "doc_number": "1",
        "doc_year": "2025",
        "doc_title": "Test Undang-Undang Pertambangan",
        "doc_status": "BERLAKU",
        "doc_subject": ["PERTAMBANGAN", "MINERAL"]
    }


@pytest.fixture
def sample_legal_units():
    """Sample legal units with hierarchical structure."""
    return [
        {
            "id": "test-unit-1",
            "document_id": "test-uuid-1",
            "unit_type": "pasal",
            "unit_id": "UU-2025-1/pasal-1",
            "content": "Dalam Undang-Undang ini yang dimaksud dengan pertambangan adalah...",
            "local_content": "pertambangan adalah sebagian atau seluruh tahapan kegiatan...",
            "path": ["pasal", "1"],
            "citation_string": "Pasal 1"
        },
        {
            "id": "test-unit-2",
            "document_id": "test-uuid-1",
            "unit_type": "ayat",
            "unit_id": "UU-2025-1/pasal-1/ayat-1",
            "parent_pasal_id": "UU-2025-1/pasal-1",
            "content": "Pertambangan mineral adalah pertambangan kekayaan alam berupa mineral...",
            "local_content": "pertambangan kekayaan alam berupa mineral dalam bentuk aslinya",
            "path": ["pasal", "1", "ayat", "1"],
            "citation_string": "Pasal 1 ayat (1)"
        }
    ]


@pytest.fixture
def sample_document_vectors():
    """Sample document vectors for testing vector search."""
    return [
        {
            "id": "test-vector-1",
            "document_id": "test-uuid-1",
            "unit_id": "UU-2025-1/pasal-1",
            "embedding": [0.1] * 1024,
            "doc_form": "UU",
            "doc_year": 2025,
            "doc_number": "1",
            "pasal_number": 1
        }
    ]


@pytest.fixture
def sample_json_document():
    """Sample JSON document structure from crawler output."""
    return {
        "doc_id": "UU-2025-1",
        "doc_form": "UU",
        "doc_number": "1",
        "doc_year": "2025",
        "doc_title": "Test Undang-Undang",
        "doc_subject": ["PERTAMBANGAN"],
        "relationships": {"mengubah": [], "diubah": []},
        "uji_materi": [],
        "document_tree": {
            "doc_type": "document",
            "children": [
                {
                    "type": "pasal",
                    "unit_id": "UU-2025-1/pasal-1",
                    "number_label": "1",
                    "local_content": "Dalam Undang-Undang ini yang dimaksud dengan:",
                    "citation_string": "Pasal 1",
                    "path": ["pasal", "1"],
                    "children": [
                        {
                            "type": "ayat",
                            "unit_id": "UU-2025-1/pasal-1/ayat-1",
                            "number_label": "1",
                            "local_content": "pertambangan adalah...",
                            "citation_string": "Pasal 1 ayat (1)",
                            "path": ["pasal", "1", "ayat", "1"],
                            "parent_pasal_id": "UU-2025-1/pasal-1"
                        }
                    ]
                }
            ]
        }
    }


@pytest.fixture
def populated_db(db_session, sample_legal_document, sample_legal_units, sample_document_vectors):
    """Database session populated with test data."""
    # Add document
    doc = LegalDocument(**sample_legal_document)
    db_session.add(doc)

    # Add units
    for unit_data in sample_legal_units:
        unit = LegalUnit(**unit_data)
        db_session.add(unit)

    # Add vectors
    for vector_data in sample_document_vectors:
        vector = DocumentVector(**vector_data)
        db_session.add(vector)

    # Add sample subjects
    subject = Subject(name="PERTAMBANGAN", description="Mining related laws")
    db_session.add(subject)

    db_session.commit()
    return db_session


# ================================
# UTILITY FIXTURES
# ================================

@pytest.fixture
def temp_json_file(sample_json_document):
    """Create temporary JSON file for testing indexer."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_json_document, f, ensure_ascii=False, indent=2)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def mock_settings():
    """Override settings for testing."""
    test_settings = {
        "database_url": "sqlite:///:memory:",
        "jina_api_key": "test-key",
        "log_level": "WARNING",
        "embed_batch_size": 2,
        "rerank_provider": "noop"
    }

    with patch.multiple(settings, **test_settings):
        yield settings


@pytest.fixture
def capture_logs():
    """Capture logs during test execution."""
    import logging
    from io import StringIO

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)

    # Add to all loggers
    loggers = [
        logging.getLogger("src.services"),
        logging.getLogger("src.pipeline"),
        logging.getLogger("src.utils")
    ]

    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    yield log_capture

    # Cleanup
    for logger in loggers:
        logger.removeHandler(handler)


# ================================
# PERFORMANCE TESTING UTILITIES
# ================================

@pytest.fixture
def performance_timer():
    """Context manager for measuring test performance."""
    import time
    from dataclasses import dataclass

    @dataclass
    class Timer:
        start_time: float = 0
        end_time: float = 0

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def duration_ms(self) -> float:
            return (self.end_time - self.start_time) * 1000

    return Timer()


# ================================
# EXTERNAL API MOCKS
# ================================

@pytest.fixture
def mock_jina_api_success():
    """Mock successful Jina API responses."""
    def handler(request: httpx.Request) -> httpx.Response:
        if "/embeddings" in str(request.url):
            data = {
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024}]
            }
            return httpx.Response(200, json=data)
        elif "/rerank" in str(request.url):
            data = {
                "model": "jina-reranker-v1",
                "results": [{"index": 0, "relevance_score": 0.95}]
            }
            return httpx.Response(200, json=data)
        else:
            return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.fixture
def mock_jina_api_error():
    """Mock Jina API error responses for testing error handling."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "Unprocessable Entity"})

    return httpx.MockTransport(handler)


# ================================
# SEARCH TESTING UTILITIES
# ================================

@pytest.fixture
def sample_search_queries():
    """Sample search queries for testing different search strategies."""
    return {
        "explicit": [
            "pasal 1",
            "pasal 1 ayat 2",
            "UU 1/2025",
            "ayat 1"
        ],
        "thematic": [
            "pertambangan mineral",
            "izin usaha pertambangan",
            "lingkungan hidup",
            "batubara"
        ],
        "contextual": [
            "bagaimana cara mengajukan izin pertambangan?",
            "apa saja kewajiban perusahaan tambang?",
            "sanksi pelanggaran undang-undang pertambangan"
        ]
    }


@pytest.fixture
def expected_search_results():
    """Expected search results for validation."""
    return [
        {
            "unit_id": "UU-2025-1/pasal-1/ayat-1",
            "content": "pertambangan adalah...",
            "citation": "UU No. 1 Tahun 2025 Pasal 1 ayat (1)",
            "score": 0.95,
            "source_type": "fts",
            "unit_type": "ayat"
        }
    ]


# ================================
# CONFIGURATION HELPERS
# ================================

@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment and configuration before each test."""
    # Store original environment
    original_env = dict(os.environ)

    yield

    # Restore environment (though not strictly necessary for most tests)
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return PROJECT_ROOT / "tests" / "data"


@pytest.fixture
def ensure_test_data_dir(test_data_dir):
    """Ensure test data directory exists."""
    test_data_dir.mkdir(exist_ok=True)
    return test_data_dir


# ================================
# INTEGRATION TESTING HELPERS
# ================================

@pytest.fixture
def indexer_with_mocks(mock_db_session, mock_jina_embedder):
    """Indexer with mocked dependencies for integration testing."""
    from src.pipeline.indexer import LegalDocumentIndexer as DocumentIndexer

    with patch('src.services.embedding.embedder.JinaEmbedder', return_value=mock_jina_embedder):
        indexer = DocumentIndexer(skip_embeddings=True)  # Skip embeddings for speed
        yield indexer


@pytest.fixture
def hybrid_retriever_with_mocks(mock_db_session, mock_jina_embedder):
    """Hybrid retriever with mocked dependencies."""
    from src.services.retriever.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(embedder=mock_jina_embedder)
    yield retriever


@pytest.fixture
def search_service_with_mocks(hybrid_retriever_with_mocks, mock_jina_reranker):
    """Search service with mocked dependencies."""
    from src.services.search.hybrid_search import HybridSearchService

    with patch('src.services.search.reranker.JinaReranker', return_value=mock_jina_reranker):
        service = HybridSearchService(retriever=hybrid_retriever_with_mocks)
        yield service


# ================================
# MARKERS FOR TEST CATEGORIES
# ================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "external: Tests requiring external APIs")


# ================================
# SKIP CONDITIONS
# ================================

@pytest.fixture
def skip_if_no_db():
    """Skip test if database is not available."""
    try:
        from src.db.session import get_db_session
        with get_db_session() as db:
            db.execute("SELECT 1")
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


@pytest.fixture
def skip_if_no_jina_api():
    """Skip test if Jina API is not available."""
    if not os.getenv("JINA_API_KEY") or os.getenv("JINA_API_KEY") == "test-key":
        pytest.skip("Jina API key not configured for integration testing")


# ================================
# TEST UTILITIES
# ================================

class TestHelper:
    """Helper utilities for testing."""

    @staticmethod
    def assert_search_result_structure(result: Dict[str, Any]):
        """Validate search result has required structure."""
        required_fields = ["unit_id", "content", "citation", "score", "source_type"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        assert isinstance(result["score"], (int, float))
        assert 0 <= result["score"] <= 1
        assert result["source_type"] in ["fts", "vector", "explicit", "reranked"]

    @staticmethod
    def assert_valid_unit_id(unit_id: str):
        """Validate unit ID follows expected format."""
        assert "/" in unit_id, "Unit ID should contain hierarchy separator"
        parts = unit_id.split("/")
        assert len(parts) >= 2, "Unit ID should have at least document and unit parts"

        # Check document part format (e.g., "UU-2025-1")
        doc_part = parts[0]
        assert "-" in doc_part, "Document part should contain form-year-number"

    @staticmethod
    def create_mock_search_result(unit_id: str, score: float = 0.8, source_type: str = "fts"):
        """Create mock search result for testing."""
        return {
            "unit_id": unit_id,
            "content": f"Sample content for {unit_id}",
            "citation": f"Citation for {unit_id}",
            "score": score,
            "source_type": source_type,
            "unit_type": "ayat",
            "document": {
                "form": "UU",
                "year": 2025,
                "number": "1"
            }
        }


@pytest.fixture
def test_helper():
    """Provide test helper utilities."""
    return TestHelper


# ================================
# PERFORMANCE TEST FIXTURES
# ================================

@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks for validation."""
    return {
        "search_latency_ms": 500,      # Max 500ms for search
        "indexing_rate_docs_min": 100, # Min 100 docs/minute
        "embedding_batch_size": 16,    # Optimal batch size
        "memory_limit_mb": 512         # Max memory usage
    }


# ================================
# CLEANUP FIXTURES
# ================================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Clean up temporary files after each test."""
    temp_files = []

    yield temp_files

    # Cleanup any temporary files created during test
    for file_path in temp_files:
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass  # Ignore cleanup errors


# ================================
# SESSION SCOPED SETUP
# ================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """One-time setup for entire test session."""
    # Ensure test directories exist
    test_dirs = [
        PROJECT_ROOT / "tests" / "data",
        PROJECT_ROOT / "tests" / "fixtures",
        PROJECT_ROOT / "tests" / "output"
    ]

    for test_dir in test_dirs:
        test_dir.mkdir(exist_ok=True)

    yield

    # Session cleanup if needed
    pass


# ================================
# ERROR TESTING UTILITIES
# ================================

@pytest.fixture
def mock_database_error():
    """Mock database errors for testing error handling."""
    from sqlalchemy.exc import SQLAlchemyError

    def raise_db_error(*args, **kwargs):
        raise SQLAlchemyError("Mocked database error")

    return raise_db_error


@pytest.fixture
def mock_api_timeout():
    """Mock API timeout for testing retry logic."""
    def raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("Request timeout")

    return raise_timeout


# ================================
# ASYNC TESTING SUPPORT
# ================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
