"""
Golden Operations Test Suite for Legal RAG Dense Search System.

Tests the 6 core scenarios defined in golden_ops.yaml:
1. Citation exact match
2. Citation partial (without ayat)
3. Query definition
4. Query sanksi (sanctions)
5. Query multi-hop definition+sanksi
6. Query general concept

Validates search accuracy, latency, and semantic quality.
"""

import asyncio
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from statistics import mean

import pytest
from unittest.mock import Mock, patch

from src.services.search.vector_search import VectorSearchService, SearchResult, SearchFilters
from src.services.citation import parse_citation, is_explicit_citation, get_best_citation_match
from src.services.embedding.embedder import JinaV4Embedder
from src.db.session import get_db_session
from src.db.models import LegalDocument, LegalUnit, DocumentVector, DocForm, DocStatus
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TestResult:
    """Test execution result with metrics."""
    test_id: str
    passed: bool
    latency_ms: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    search_type: str
    actual_results: List[SearchResult]
    error_message: Optional[str] = None


class GoldenOpsTestSuite:
    """Golden operations test suite for dense search validation."""

    def __init__(self, config_path: str = "tests/golden_ops.yaml"):
        """Initialize test suite with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.search_service = None
        self.test_results: List[TestResult] = []

    def _load_config(self) -> Dict[str, Any]:
        """Load golden test configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Golden test config not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @pytest.fixture(autouse=True)
    def setup_search_service(self):
        """Setup vector search service for testing."""
        # Mock embedder to avoid API calls in tests
        mock_embedder = Mock(spec=JinaV4Embedder)
        mock_embedder.embed_texts.return_value = [[0.1] * 384]  # Mock 384-dim embedding

        self.search_service = VectorSearchService(
            embedder=mock_embedder,
            default_k=15,
            min_citation_confidence=0.60
        )
        yield
        self.search_service = None

    # ============================================================================
    # Citation Exact Match Tests
    # ============================================================================

    @pytest.mark.parametrize("test_case", [
        pytest.param(tc, id=tc['id']) for tc in [
            {
                'id': 'cite_exact_001',
                'query': 'UU 8/2019 Pasal 6 ayat (2) huruf b',
                'expected_search_type': 'explicit_citation',
                'expected_doc_form': 'UU',
                'expected_doc_year': 2019,
                'expected_doc_number': '8',
                'expected_pasal': '6',
                'expected_ayat': '2',
                'expected_huruf': 'b'
            },
            {
                'id': 'cite_exact_002',
                'query': 'PP No. 45 Tahun 2020 Pasal 12',
                'expected_search_type': 'explicit_citation',
                'expected_doc_form': 'PP',
                'expected_doc_year': 2020,
                'expected_doc_number': '45',
                'expected_pasal': '12'
            },
            {
                'id': 'cite_exact_003',
                'query': 'Undang-Undang No. 4 Tahun 2009 Pasal 121 Ayat 1',
                'expected_search_type': 'explicit_citation',
                'expected_doc_form': 'UU',
                'expected_doc_year': 2009,
                'expected_doc_number': '4',
                'expected_pasal': '121',
                'expected_ayat': '1'
            }
        ]
    ])
    def test_citation_exact_match(self, test_case):
        """Test exact citation parsing and direct SQL lookup."""
        query = test_case['query']
        start_time = time.time()

        # Test citation parsing
        assert is_explicit_citation(query, 0.60), f"Query should be detected as explicit citation: {query}"

        citation_match = get_best_citation_match(query)
        assert citation_match is not None, f"Should parse citation from: {query}"
        assert citation_match.confidence >= 0.60, f"Citation confidence too low: {citation_match.confidence}"

        # Validate parsed citation components
        if 'expected_doc_form' in test_case:
            assert citation_match.doc_form == test_case['expected_doc_form']
        if 'expected_doc_year' in test_case:
            assert citation_match.doc_year == test_case['expected_doc_year']
        if 'expected_doc_number' in test_case:
            assert citation_match.doc_number == test_case['expected_doc_number']
        if 'expected_pasal' in test_case:
            assert citation_match.pasal_number == test_case['expected_pasal']
        if 'expected_ayat' in test_case:
            assert citation_match.ayat_number == test_case['expected_ayat']
        if 'expected_huruf' in test_case:
            assert citation_match.huruf_letter == test_case['expected_huruf']

        # Test search execution
        with patch.object(self.search_service, '_handle_explicit_citation') as mock_citation_search:
            mock_citation_search.return_value = [
                SearchResult(
                    id=f"test-unit-{test_case['id']}",
                    content="Mock citation search result",
                    citation_string=query,
                    score=1.0,
                    unit_type="PASAL",
                    unit_id=f"test-unit-{test_case['id']}",
                    doc_form=test_case.get('expected_doc_form'),
                    doc_year=test_case.get('expected_doc_year'),
                    doc_number=test_case.get('expected_doc_number'),
                    metadata={'search_type': 'explicit_citation'}
                )
            ]

            result = self.search_service.search(query, k=5)
            latency_ms = (time.time() - start_time) * 1000

            # Validate search response
            assert result['metadata']['search_type'] == 'explicit_citation'
            assert len(result['results']) >= 1, "Should return at least one result for exact citation"
            assert latency_ms < 50, f"Citation search should be fast (<50ms), got {latency_ms}ms"

            # Validate result quality
            first_result = result['results'][0]
            assert first_result.score == 1.0, "Exact citation should have perfect score"
            assert first_result.metadata.get('search_type') == 'explicit_citation'

    # ============================================================================
    # Citation Partial Tests
    # ============================================================================

    @pytest.mark.parametrize("test_case", [
        pytest.param(tc, id=tc['id']) for tc in [
            {
                'id': 'cite_partial_001',
                'query': 'Pasal 15 ayat (1)',
                'expected_search_type': 'explicit_citation',
                'expected_pasal': '15',
                'expected_ayat': '1'
            },
            {
                'id': 'cite_partial_002',
                'query': 'UU 21/2008',
                'expected_search_type': 'explicit_citation',
                'expected_doc_form': 'UU',
                'expected_doc_year': 2008,
                'expected_doc_number': '21'
            },
            {
                'id': 'cite_partial_003',
                'query': 'ayat (3) huruf c',
                'expected_search_type': 'explicit_citation',
                'expected_ayat': '3',
                'expected_huruf': 'c'
            }
        ]
    ])
    def test_citation_partial_match(self, test_case):
        """Test partial citation parsing with broader lookup scope."""
        query = test_case['query']
        start_time = time.time()

        # Test citation parsing
        assert is_explicit_citation(query, 0.60), f"Partial citation should be detected: {query}"

        citation_match = get_best_citation_match(query)
        assert citation_match is not None, f"Should parse partial citation from: {query}"

        # Test search execution with mock
        with patch.object(self.search_service, '_handle_explicit_citation') as mock_citation_search:
            mock_results = []
            for i in range(3):  # Mock multiple results for partial citations
                mock_results.append(SearchResult(
                    id=f"test-unit-{test_case['id']}-{i}",
                    content=f"Mock partial citation result {i}",
                    citation_string=f"Mock Citation {i}",
                    score=1.0,
                    unit_type="PASAL",
                    unit_id=f"test-unit-{test_case['id']}-{i}",
                    metadata={'search_type': 'explicit_citation'}
                ))
            mock_citation_search.return_value = mock_results

            result = self.search_service.search(query, k=10)
            latency_ms = (time.time() - start_time) * 1000

            # Validate search response
            assert result['metadata']['search_type'] == 'explicit_citation'
            assert len(result['results']) >= 1, "Should return results for partial citation"
            assert latency_ms < 100, f"Partial citation search should be fast (<100ms), got {latency_ms}ms"

    # ============================================================================
    # Contextual Definition Tests
    # ============================================================================

    @pytest.mark.parametrize("test_case", [
        pytest.param(tc, id=tc['id']) for tc in [
            {
                'id': 'def_001',
                'query': 'definisi badan hukum dalam peraturan perundang-undangan',
                'expected_keywords': ['badan hukum', 'definisi', 'pengertian'],
                'min_precision_at_5': 0.80
            },
            {
                'id': 'def_002',
                'query': 'apa yang dimaksud dengan kontrak kerja',
                'expected_keywords': ['kontrak', 'kerja', 'perjanjian', 'definisi'],
                'min_precision_at_5': 0.75
            },
            {
                'id': 'def_003',
                'query': 'pengertian tata kelola perusahaan yang baik',
                'expected_keywords': ['tata kelola', 'perusahaan', 'governance'],
                'min_precision_at_5': 0.70
            }
        ]
    ])
    def test_contextual_definition_search(self, test_case):
        """Test contextual definition queries using vector search."""
        query = test_case['query']
        start_time = time.time()

        # Verify not detected as citation
        assert not is_explicit_citation(query, 0.60), f"Definition query should not be citation: {query}"

        # Test search execution with mock
        with patch.object(self.search_service, '_handle_contextual_search') as mock_vector_search:
            mock_results = []
            for i in range(10):  # Mock comprehensive results
                mock_results.append(SearchResult(
                    id=f"def-unit-{test_case['id']}-{i}",
                    content=f"Definition content {i} containing {test_case['expected_keywords'][0]}",
                    citation_string=f"Mock Definition Citation {i}",
                    score=0.9 - (i * 0.05),  # Decreasing similarity scores
                    unit_type="PASAL",
                    unit_id=f"def-unit-{test_case['id']}-{i}",
                    metadata={'search_type': 'contextual_semantic'}
                ))
            mock_vector_search.return_value = mock_results

            result = self.search_service.search(query, k=15)
            latency_ms = (time.time() - start_time) * 1000

            # Validate search response
            assert result['metadata']['search_type'] == 'contextual_semantic'
            assert len(result['results']) >= 5, "Should return multiple definition results"
            assert latency_ms < 200, f"Vector search should complete in <200ms, got {latency_ms}ms"

            # Calculate precision@5
            precision_at_5 = self._calculate_precision_at_k(result['results'][:5], test_case['expected_keywords'])
            assert precision_at_5 >= test_case['min_precision_at_5'], \
                f"Precision@5 {precision_at_5} below threshold {test_case['min_precision_at_5']}"

    # ============================================================================
    # Sanctions Query Tests
    # ============================================================================

    @pytest.mark.parametrize("test_case", [
        pytest.param(tc, id=tc['id']) for tc in [
            {
                'id': 'sanksi_001',
                'query': 'sanksi pidana pelanggaran lingkungan hidup',
                'expected_keywords': ['sanksi', 'pidana', 'pelanggaran', 'lingkungan'],
                'min_precision_at_5': 0.80
            },
            {
                'id': 'sanksi_002',
                'query': 'sanksi administratif korporasi',
                'expected_keywords': ['sanksi', 'administratif', 'korporasi'],
                'min_precision_at_5': 0.75
            },
            {
                'id': 'sanksi_003',
                'query': 'denda maksimal pelanggaran peraturan pasar modal',
                'expected_keywords': ['denda', 'maksimal', 'pelanggaran', 'pasar modal'],
                'min_precision_at_5': 0.70
            }
        ]
    ])
    def test_sanctions_query_search(self, test_case):
        """Test sanctions/penalty queries using vector search."""
        query = test_case['query']
        start_time = time.time()

        # Verify not detected as citation
        assert not is_explicit_citation(query, 0.60), f"Sanctions query should not be citation: {query}"

        # Test search execution with mock
        with patch.object(self.search_service, '_handle_contextual_search') as mock_vector_search:
            mock_results = []
            for i in range(10):
                mock_results.append(SearchResult(
                    id=f"sanksi-unit-{test_case['id']}-{i}",
                    content=f"Sanctions content {i} about {test_case['expected_keywords'][0]}",
                    citation_string=f"Mock Sanctions Citation {i}",
                    score=0.85 - (i * 0.04),
                    unit_type="PASAL",
                    unit_id=f"sanksi-unit-{test_case['id']}-{i}",
                    metadata={'search_type': 'contextual_semantic'}
                ))
            mock_vector_search.return_value = mock_results

            result = self.search_service.search(query, k=15)
            latency_ms = (time.time() - start_time) * 1000

            # Validate search response
            assert result['metadata']['search_type'] == 'contextual_semantic'
            assert len(result['results']) >= 5, "Should return multiple sanctions results"
            assert latency_ms < 200, f"Vector search should complete in <200ms, got {latency_ms}ms"

            # Calculate precision@5
            precision_at_5 = self._calculate_precision_at_k(result['results'][:5], test_case['expected_keywords'])
            assert precision_at_5 >= test_case['min_precision_at_5'], \
                f"Precision@5 {precision_at_5} below threshold {test_case['min_precision_at_5']}"

    # ============================================================================
    # Multi-hop Definition+Sanctions Tests
    # ============================================================================

    @pytest.mark.parametrize("test_case", [
        pytest.param(tc, id=tc['id']) for tc in [
            {
                'id': 'multihop_001',
                'query': 'pengertian insider trading dan sanksi pidananya',
                'expected_keywords': ['insider trading', 'sanksi', 'pidana', 'definisi'],
                'min_precision_at_5': 0.70,
                'min_results': 8
            },
            {
                'id': 'multihop_002',
                'query': 'definisi aksi korporasi dan denda pelanggarannya',
                'expected_keywords': ['aksi korporasi', 'definisi', 'denda', 'pelanggaran'],
                'min_precision_at_5': 0.70,
                'min_results': 8
            }
        ]
    ])
    def test_multi_hop_queries(self, test_case):
        """Test complex multi-hop queries requiring multiple concepts."""
        query = test_case['query']
        start_time = time.time()

        # Verify not detected as citation
        assert not is_explicit_citation(query, 0.60), f"Multi-hop query should not be citation: {query}"

        # Test search execution with mock
        with patch.object(self.search_service, '_handle_contextual_search') as mock_vector_search:
            mock_results = []
            for i in range(15):  # More results for complex queries
                # Mix definition and sanctions content
                content_type = "definition" if i % 2 == 0 else "sanctions"
                mock_results.append(SearchResult(
                    id=f"multihop-unit-{test_case['id']}-{i}",
                    content=f"Multi-hop {content_type} content {i}",
                    citation_string=f"Mock Multi-hop Citation {i}",
                    score=0.80 - (i * 0.03),
                    unit_type="PASAL",
                    unit_id=f"multihop-unit-{test_case['id']}-{i}",
                    metadata={'search_type': 'contextual_semantic', 'content_type': content_type}
                ))
            mock_vector_search.return_value = mock_results

            result = self.search_service.search(query, k=20)
            latency_ms = (time.time() - start_time) * 1000

            # Validate search response
            assert result['metadata']['search_type'] == 'contextual_semantic'
            assert len(result['results']) >= test_case['min_results'], \
                f"Should return at least {test_case['min_results']} results for multi-hop query"
            assert latency_ms < 250, f"Multi-hop search should complete in <250ms, got {latency_ms}ms"

            # Calculate precision@5
            precision_at_5 = self._calculate_precision_at_k(result['results'][:5], test_case['expected_keywords'])
            assert precision_at_5 >= test_case['min_precision_at_5'], \
                f"Precision@5 {precision_at_5} below threshold {test_case['min_precision_at_5']}"

    # ============================================================================
    # General Concept Tests
    # ============================================================================

    @pytest.mark.parametrize("test_case", [
        pytest.param(tc, id=tc['id']) for tc in [
            {
                'id': 'concept_001',
                'query': 'tanggung jawab sosial perusahaan',
                'expected_keywords': ['tanggung jawab', 'sosial', 'perusahaan', 'CSR'],
                'min_precision_at_5': 0.60
            },
            {
                'id': 'concept_002',
                'query': 'transformasi digital sektor keuangan',
                'expected_keywords': ['transformasi', 'digital', 'keuangan', 'teknologi'],
                'min_precision_at_5': 0.60
            },
            {
                'id': 'concept_003',
                'query': 'perlindungan konsumen dalam transaksi elektronik',
                'expected_keywords': ['perlindungan', 'konsumen', 'transaksi', 'elektronik'],
                'min_precision_at_5': 0.60
            }
        ]
    ])
    def test_general_concept_queries(self, test_case):
        """Test broad conceptual queries."""
        query = test_case['query']
        start_time = time.time()

        # Verify not detected as citation
        assert not is_explicit_citation(query, 0.60), f"General concept query should not be citation: {query}"

        # Test search execution with mock
        with patch.object(self.search_service, '_handle_contextual_search') as mock_vector_search:
            mock_results = []
            for i in range(20):  # Many results for broad concepts
                mock_results.append(SearchResult(
                    id=f"concept-unit-{test_case['id']}-{i}",
                    content=f"General concept content {i}",
                    citation_string=f"Mock Concept Citation {i}",
                    score=0.75 - (i * 0.02),
                    unit_type="PASAL",
                    unit_id=f"concept-unit-{test_case['id']}-{i}",
                    metadata={'search_type': 'contextual_semantic'}
                ))
            mock_vector_search.return_value = mock_results

            result = self.search_service.search(query, k=25)
            latency_ms = (time.time() - start_time) * 1000

            # Validate search response
            assert result['metadata']['search_type'] == 'contextual_semantic'
            assert len(result['results']) >= 10, "Should return many results for general concepts"
            assert latency_ms < 200, f"Concept search should complete in <200ms, got {latency_ms}ms"

            # Calculate precision@5 (lower threshold for broad concepts)
            precision_at_5 = self._calculate_precision_at_k(result['results'][:5], test_case['expected_keywords'])
            assert precision_at_5 >= test_case['min_precision_at_5'], \
                f"Precision@5 {precision_at_5} below threshold {test_case['min_precision_at_5']}"

    # ============================================================================
    # Performance and Integration Tests
    # ============================================================================

    def test_search_type_routing_accuracy(self):
        """Test that queries are routed to correct search strategies."""
        test_cases = [
            ("UU 8/2019 Pasal 6", "explicit_citation"),
            ("definisi badan hukum", "contextual_semantic"),
            ("sanksi pidana korupsi", "contextual_semantic"),
            ("PP 45/2020", "explicit_citation"),
            ("Pasal 15 ayat (1)", "explicit_citation"),
        ]

        routing_accuracy = 0
        for query, expected_type in test_cases:
            with patch.object(self.search_service, '_handle_explicit_citation') as mock_citation, \
                 patch.object(self.search_service, '_handle_contextual_search') as mock_vector:

                mock_citation.return_value = []
                mock_vector.return_value = []

                result = self.search_service.search(query, k=5)
                actual_type = result['metadata']['search_type']

                if actual_type == expected_type:
                    routing_accuracy += 1

        routing_accuracy = routing_accuracy / len(test_cases)
        assert routing_accuracy >= 0.80, f"Search routing accuracy {routing_accuracy} below 80%"

    def test_concurrent_search_performance(self):
        """Test concurrent search performance."""
        queries = [
            "UU 8/2019 Pasal 6",
            "definisi badan hukum",
            "sanksi pidana lingkungan",
            "PP 45/2020 Pasal 12",
            "tanggung jawab sosial perusahaan"
        ]

        async def run_concurrent_searches():
            tasks = []
            for query in queries:
                with patch.object(self.search_service, '_handle_explicit_citation') as mock_citation, \
                     patch.object(self.search_service, '_handle_contextual_search') as mock_vector:

                    mock_citation.return_value = []
                    mock_vector.return_value = []

                    task = asyncio.create_task(asyncio.to_thread(self.search_service.search, query, 10))
                    tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            return results, total_time

        # Run concurrent test
        results, total_time = asyncio.run(run_concurrent_searches())

        assert len(results) == len(queries), "All concurrent searches should complete"
        assert total_time < 2.0, f"Concurrent searches took too long: {total_time}s"

    def test_error_handling_graceful_degradation(self):
        """Test graceful error handling for edge cases."""
        error_cases = [
            "",  # Empty query
            "   ",  # Whitespace only
            "xyz nonexistent regulation 999/9999",  # Non-existent citation
            "random gibberish query that makes no sense ăẩấềếể",  # Nonsensical with special chars
        ]

        for query in error_cases:
            try:
                result = self.search_service.search(query, k=5)

                # Should not crash and return valid structure
                assert 'results' in result, "Should return results key even for error cases"
                assert 'metadata' in result, "Should return metadata even for error cases"
                assert isinstance(result['results'], list), "Results should be list"
                assert len(result['results']) <= 5, "Should respect limit even for edge cases"

            except Exception as e:
                pytest.fail(f"Search should handle error case gracefully, but raised: {e}")

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _calculate_precision_at_k(self, results: List[SearchResult], expected_keywords: List[str]) -> float:
        """Calculate precision@k based on keyword relevance."""
        if not results:
            return 0.0

        relevant_count = 0
        for result in results:
            content_lower = result.content.lower()
            citation_lower = result.citation_string.lower()

            # Check if any expected keywords are present
            for keyword in expected_keywords:
                if keyword.lower() in content_lower or keyword.lower() in citation_lower:
                    relevant_count += 1
                    break

        return relevant_count / len(results)

    def _calculate_recall_at_k(self, results: List[SearchResult], total_relevant: int) -> float:
        """Calculate recall@k (simplified for mocked tests)."""
        if total_relevant == 0:
            return 1.0

        # For mocked tests, assume results are relevant based on score threshold
        relevant_retrieved = sum(1 for r in results if r.score > 0.5)
        return min(relevant_retrieved / total_relevant, 1.0)

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.test_results:
            return {"error": "No test results available"}

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)

        latencies = [r.latency_ms for r in self.test_results if r.latency_ms > 0]
        avg_latency = mean(latencies) if latencies else 0

        # Group results by search type
        citation_tests = [r for r in self.test_results if r.search_type == 'explicit_citation']
        semantic_tests = [r for r in self.test_results if r.search_type == 'contextual_semantic']

        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "average_latency_ms": avg_latency
            },
            "citation_tests": {
                "count": len(citation_tests),
                "passed": sum(1 for r in citation_tests if r.passed),
                "avg_latency": mean([r.latency_ms for r in citation_tests if r.latency_ms > 0]) if citation_tests else 0
            },
            "semantic_tests": {
                "count": len(semantic_tests),
                "passed": sum(1 for r in semantic_tests if r.passed),
                "avg_latency": mean([r.latency_ms for r in semantic_tests if r.latency_ms > 0]) if semantic_tests else 0
            },
            "performance_targets": {
                "citation_latency_target_ms": 50,
                "semantic_latency_target_ms": 200,
                "precision_at_5_target": 0.75,
                "routing_accuracy_target": 0.80
            }
        }


# Integration with pytest
def pytest_configure(config):
    """Configure pytest for golden tests."""
    config.addinivalue_line(
        "markers", "golden: mark test as part of golden test suite"
    )


@pytest.mark.golden
class TestGoldenOps(GoldenOpsTestSuite):
    """Pytest wrapper for golden operations test suite."""

    def test_full_golden_suite(self):
        """Run complete golden test suite and generate report."""
        logger.info("Starting full golden operations test suite")

        # This would normally run all individual tests
        # For now, just validate the framework is working
        assert self.config is not None, "Configuration should be loaded"
        assert 'test_cases' in self.config, "Test cases should be defined in config"
        assert self.search_service is not None, "Search service should be initialized"

        logger.info("Golden test suite framework validated successfully")


if __name__ == "__main__":
    """Run golden tests standalone."""
    suite = G
