#!/usr/bin/env python3
"""
Simplified Test Runner for Legal RAG System

Focuses on:
- Critical blocker validation (Jina API, SQL formatting)
- Basic functionality testing
- Simple execution without complex dependencies
- Clear error reporting

Usage:
    python tests/run_tests.py --quick    # Quick validation
    python tests/run_tests.py --unit     # Unit tests
    python tests/run_tests.py --all      # All tests
"""

import os
import sys
import argparse
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment
os.environ.update({
    "DATABASE_URL": "sqlite:///:memory:",
    "JINA_API_KEY": "test-key",
    "LOG_LEVEL": "WARNING",
    "EMBED_BATCH_SIZE": "2",
    "RERANK_PROVIDER": "noop",
})


@dataclass
class TestResult:
    """Test execution result."""
    name: str
    status: str  # "PASS", "FAIL", "SKIP", "ERROR"
    duration_ms: float
    message: str = ""


class SimpleTestRunner:
    """Simplified test runner."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
        self.setup_logging()

    def setup_logging(self):
        """Setup simple logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        self.logger = logging.getLogger("test_runner")

    def run_smoke_tests(self) -> List[TestResult]:
        """Run basic smoke tests."""
        results = []

        self.logger.info("Running smoke tests...")

        # Test 1: Import validation
        start = time.time()
        try:
            from src.config.settings import settings
            from src.db.models import LegalDocument

            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="Import Test",
                status="PASS",
                duration_ms=duration,
                message="All imports successful"
            ))
        except Exception as e:
            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="Import Test",
                status="FAIL",
                duration_ms=duration,
                message=f"Import error: {str(e)}"
            ))

        # Test 2: Configuration validation
        start = time.time()
        try:
            from src.config.settings import settings

            # Check basic settings exist
            assert hasattr(settings, 'database_url')
            assert hasattr(settings, 'jina_api_key')

            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="Configuration Test",
                status="PASS",
                duration_ms=duration,
                message="Configuration loaded successfully"
            ))
        except Exception as e:
            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="Configuration Test",
                status="FAIL",
                duration_ms=duration,
                message=f"Configuration error: {str(e)}"
            ))

        return results

    def run_basic_functionality_tests(self) -> List[TestResult]:
        """Run basic functionality tests."""
        results = []

        self.logger.info("Running basic functionality tests...")

        # Test 3: Database model creation
        start = time.time()
        try:
            from src.db.models import LegalDocument, LegalUnit

            # Try to create model instances
            doc = LegalDocument(
                id="test-id",
                doc_id="TEST-2025-1",
                doc_form="UU",
                doc_number="1",
                doc_year="2025",
                doc_title="Test Document",
                doc_status="BERLAKU"
            )

            unit = LegalUnit(
                id="test-unit-id",
                document_id="test-id",
                unit_type="ayat",
                unit_id="TEST-2025-1/pasal-1/ayat-1",
                content="Test content",
                local_content="test",
                citation_string="Pasal 1 ayat (1)"
            )

            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="Database Models",
                status="PASS",
                duration_ms=duration,
                message="Models created successfully"
            ))
        except Exception as e:
            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="Database Models",
                status="FAIL",
                duration_ms=duration,
                message=f"Model creation error: {str(e)}"
            ))

        # Test 4: Embedder initialization
        start = time.time()
        try:
            from src.services.embedding.embedder import JinaV4Embedder
            from src.utils.http import HttpClient

            # Test embedder can be created
            embedder = JinaV4Embedder()
            assert embedder.model
            assert embedder.default_dims > 0

            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="Embedder Initialization",
                status="PASS",
                duration_ms=duration,
                message="Embedder created successfully"
            ))
        except Exception as e:
            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="Embedder Initialization",
                status="FAIL",
                duration_ms=duration,
                message=f"Embedder error: {str(e)}"
            ))

        # Test 5: Retriever initialization
        start = time.time()
        try:
            from src.services.retriever.hybrid_retriever import HybridRetriever
            from unittest.mock import MagicMock

            # Test with mock embedder
            mock_embedder = MagicMock()
            mock_embedder.embed_single.return_value = [0.1] * 1024

            retriever = HybridRetriever(embedder=mock_embedder)
            assert retriever.embedder is mock_embedder

            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="Retriever Initialization",
                status="PASS",
                duration_ms=duration,
                message="Retriever created successfully"
            ))
        except Exception as e:
            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="Retriever Initialization",
                status="FAIL",
                duration_ms=duration,
                message=f"Retriever error: {str(e)}"
            ))

        return results

    def run_critical_fixes_validation(self) -> List[TestResult]:
        """Validate critical fixes from TODO_NEXT.md."""
        results = []

        self.logger.info("Validating critical fixes...")

        # Test 6: SQL Query Format Fix
        start = time.time()
        try:
            from src.services.retriever.hybrid_retriever import FTSSearcher
            from sqlalchemy.sql import text
            from unittest.mock import MagicMock

            # Test that FTS searcher doesn't use .format() on text() objects
            mock_session = MagicMock()
            mock_session.execute.return_value.fetchall.return_value = []

            searcher = FTSSearcher(mock_session)

            # This should not raise AttributeError about .format() method
            results_list = searcher.search("test query", limit=5)

            # Verify the query was executed properly
            assert mock_session.execute.called
            executed_query = mock_session.execute.call_args[0][0]

            # Should be TextClause, not string with format placeholders
            from sqlalchemy.sql.elements import TextClause
            assert isinstance(executed_query, TextClause)

            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="SQL Format Fix",
                status="PASS",
                duration_ms=duration,
                message="SQL queries use proper parameter binding"
            ))
        except Exception as e:
            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="SQL Format Fix",
                status="FAIL",
                duration_ms=duration,
                message=f"SQL format error: {str(e)}"
            ))

        # Test 7: Jina API Format Validation
        start = time.time()
        try:
            from src.services.embedding.embedder import JinaV4Embedder
            from src.utils.http import HttpClient
            import httpx
            import json
            from unittest.mock import MagicMock

            # Test correct API payload format
            captured_request = {}

            def mock_handler(request: httpx.Request) -> httpx.Response:
                captured_request["json"] = json.loads(request.content.decode())
                return httpx.Response(200, json={
                    "model": "jina-embeddings-v4",
                    "data": [{"embedding": [0.1] * 1024, "index": 0}]
                })

            transport = httpx.MockTransport(mock_handler)
            client = HttpClient(client=httpx.Client(transport=transport))
            embedder = JinaV4Embedder(client=client)

            # This should not cause 422 errors
            result = embedder.embed_single("test text")

            # Validate correct v4 API format
            payload = captured_request["json"]
            assert payload["model"] == "jina-embeddings-v4"
            assert payload["input"] == ["test text"]
            assert payload["task"] == "retrieval.passage"
            assert payload["dimensions"] == 1024
            assert payload["return_multivector"] == False
            assert payload["late_chunking"] == False
            assert payload["truncate"] == True
            assert "encoding_format" not in payload  # v4 doesn't use this

            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="Jina API Format Fix",
                status="PASS",
                duration_ms=duration,
                message="Jina API requests use correct v4 format"
            ))
        except Exception as e:
            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="Jina API Format Fix",
                status="FAIL",
                duration_ms=duration,
                message=f"Jina API format error: {str(e)}"
            ))

        # Test 8: Reranker Smoke Test
        start = time.time()
        try:
            from src.services.search.reranker import JinaReranker
            from src.services.retriever.hybrid_retriever import SearchResult
            from src.utils.http import HttpClient
            import httpx
            import json

            # Mock reranker response with two scores
            def mock_handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.content.decode())

                # Validate reranker request format
                assert payload["model"] == "jina-reranker-v2-base-multilingual"
                assert "query" in payload
                assert "documents" in payload
                assert "top_n" in payload
                assert "return_documents" in payload

                # Return mock scores in descending order
                return httpx.Response(200, json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "usage": {"total_tokens": 100},
                    "results": [
                        {"index": 1, "relevance_score": 0.95},  # Higher score
                        {"index": 0, "relevance_score": 0.85}   # Lower score
                    ]
                })

            transport = httpx.MockTransport(mock_handler)
            client = HttpClient(client=httpx.Client(transport=transport))
            reranker = JinaReranker(client=client)

            # Create sample search results
            search_results = [
                SearchResult(
                    id="1",
                    text="First document",
                    citation_string="Doc 1",
                    score=0.5,
                    source_type="fts",
                    unit_type="ayat",
                    unit_id="test-1"
                ),
                SearchResult(
                    id="2",
                    text="Second document",
                    citation_string="Doc 2",
                    score=0.6,
                    source_type="vector",
                    unit_type="pasal",
                    unit_id="test-2"
                )
            ]

            # Test reranking
            reranked = reranker.rerank("test query", search_results, top_k=2)

            # Verify sort order: results should be sorted by score descending
            assert len(reranked) == 2
            assert reranked[0].score >= reranked[1].score, f"Results not sorted by score: {reranked[0].score} < {reranked[1].score}"

            # Check that scores were updated (they should be 0.95 and 0.85)
            scores = [r.score for r in reranked]
            assert 0.95 in scores, f"Expected score 0.95 not found in {scores}"
            assert 0.85 in scores, f"Expected score 0.85 not found in {scores}"

            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="Reranker Smoke Test",
                status="PASS",
                duration_ms=duration,
                message="Reranker correctly sorts by relevance score"
            ))
        except Exception as e:
            duration = (time.time() - start) * 1000
            results.append(TestResult(
                name="Reranker Smoke Test",
                status="FAIL",
                duration_ms=duration,
                message=f"Reranker test error: {type(e).__name__}: {str(e)}"
            ))

        return results

    def run_pytest_suite(self, test_path: str, test_name: str) -> TestResult:
        """Run pytest suite with simplified approach."""
        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "-v", "--tb=short", "--disable-warnings",
            "--maxfail=5"  # Stop after 5 failures
        ]

        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            duration = (time.time() - start) * 1000

            if result.returncode == 0:
                return TestResult(
                    name=test_name,
                    status="PASS",
                    duration_ms=duration,
                    message="All tests passed"
                )
            elif result.returncode == 5:  # No tests found
                return TestResult(
                    name=test_name,
                    status="SKIP",
                    duration_ms=duration,
                    message="No tests found"
                )
            else:
                # Extract failure info from output
                lines = result.stdout.split('\n')
                failure_lines = [line for line in lines if 'FAILED' in line]
                message = f"Tests failed: {len(failure_lines)} failures"

                return TestResult(
                    name=test_name,
                    status="FAIL",
                    duration_ms=duration,
                    message=message
                )

        except subprocess.TimeoutExpired:
            duration = (time.time() - start) * 1000
            return TestResult(
                name=test_name,
                status="ERROR",
                duration_ms=duration,
                message="Test suite timed out"
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                name=test_name,
                status="ERROR",
                duration_ms=duration,
                message=f"Execution error: {str(e)}"
            )

    def print_results(self, results: List[TestResult]):
        """Print test results in a clear format."""
        print("\n" + "="*60)
        print("🧪 LEGAL RAG SYSTEM - TEST RESULTS")
        print("="*60)

        total_tests = len(results)
        passed = sum(1 for r in results if r.status == "PASS")
        failed = sum(1 for r in results if r.status == "FAIL")
        skipped = sum(1 for r in results if r.status == "SKIP")
        errors = sum(1 for r in results if r.status == "ERROR")

        print(f"\n📊 SUMMARY:")
        print(f"   Total: {total_tests}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏭️  Skipped: {skipped}")
        print(f"   🚨 Errors: {errors}")

        if total_tests > 0:
            success_rate = passed / total_tests
            print(f"   📈 Success Rate: {success_rate:.1%}")

        total_duration = (time.time() - self.start_time) * 1000
        print(f"   ⏱️  Total Duration: {total_duration:.0f}ms")

        # Status indicator
        if failed == 0 and errors == 0:
            print(f"\n🟢 STATUS: ALL TESTS PASSED!")
        elif failed + errors <= 2:
            print(f"\n🟡 STATUS: MOSTLY WORKING (Minor issues)")
        else:
            print(f"\n🔴 STATUS: NEEDS ATTENTION")

        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for result in results:
            status_icon = {
                "PASS": "✅",
                "FAIL": "❌",
                "SKIP": "⏭️",
                "ERROR": "🚨"
            }.get(result.status, "❓")

            print(f"   {status_icon} {result.name}: {result.message} ({result.duration_ms:.0f}ms)")

        print("\n" + "="*60)

    def run_quick_validation(self) -> List[TestResult]:
        """Run quick validation tests."""
        all_results = []

        self.logger.info("⚡ Quick Validation Started")

        # Basic smoke tests
        smoke_results = self.run_smoke_tests()
        all_results.extend(smoke_results)

        # Critical fixes validation
        critical_results = self.run_critical_fixes_validation()
        all_results.extend(critical_results)

        return all_results

    def run_unit_tests(self) -> List[TestResult]:
        """Run unit tests if available."""
        results = []

        unit_test_files = [
            ("tests/test_embedding.py", "Legacy Embedding Tests"),
            ("tests/test_retriever.py", "Legacy Retriever Tests"),
            ("tests/test_pdf_orchestrator.py", "PDF Orchestrator Tests")
        ]

        for test_file, test_name in unit_test_files:
            test_path = PROJECT_ROOT / test_file
            if test_path.exists():
                result = self.run_pytest_suite(str(test_path), test_name)
                results.append(result)

        return results

    def run_all_available_tests(self) -> List[TestResult]:
        """Run all available tests."""
        all_results = []

        self.logger.info("🧪 Running All Available Tests")

        # Quick validation first
        quick_results = self.run_quick_validation()
        all_results.extend(quick_results)

        # Unit tests
        unit_results = self.run_unit_tests()
        all_results.extend(unit_results)

        # Try to run any pytest files in tests directory
        test_files = list(PROJECT_ROOT.glob("tests/test_*.py"))
        for test_file in test_files:
            if test_file.name not in ["test_embedding.py", "test_retriever.py", "test_pdf_orchestrator.py"]:
                result = self.run_pytest_suite(str(test_file), f"Additional Tests ({test_file.name})")
                all_results.append(result)

        return all_results

    def validate_environment(self) -> TestResult:
        """Validate test environment setup."""
        start = time.time()

        try:
            issues = []

            # Check Python version
            if sys.version_info < (3, 8):
                issues.append("Python 3.8+ required")

            # Check critical modules
            critical_modules = ["sqlalchemy", "pydantic", "httpx"]
            for module in critical_modules:
                try:
                    __import__(module)
                except ImportError:
                    issues.append(f"Missing module: {module}")

            # Check project structure
            critical_dirs = ["src", "src/config", "src/db", "src/services"]
            for dir_path in critical_dirs:
                if not (PROJECT_ROOT / dir_path).exists():
                    issues.append(f"Missing directory: {dir_path}")

            duration = (time.time() - start) * 1000

            if issues:
                return TestResult(
                    name="Environment Validation",
                    status="FAIL",
                    duration_ms=duration,
                    message=f"Issues found: {'; '.join(issues)}"
                )
            else:
                return TestResult(
                    name="Environment Validation",
                    status="PASS",
                    duration_ms=duration,
                    message="Environment setup correctly"
                )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                name="Environment Validation",
                status="ERROR",
                duration_ms=duration,
                message=f"Validation error: {str(e)}"
            )

    def run_database_test(self) -> TestResult:
        """Test basic database functionality."""
        start = time.time()

        try:
            from src.db.session import get_db_session
            from sqlalchemy import text

            # Test session creation with actual configured database
            # Skip table creation for now to avoid PostgreSQL/SQLite compatibility issues
            with get_db_session() as db:
                # Basic query test that works on both PostgreSQL and SQLite
                result = db.execute(text("SELECT 1 as test")).fetchone()
                assert result[0] == 1

                # Test database type detection
                db_name = str(db.bind.url).split('://')[0] if db.bind else 'unknown'
                self.logger.debug(f"Database type: {db_name}")

            duration = (time.time() - start) * 1000
            return TestResult(
                name="Database Test",
                status="PASS",
                duration_ms=duration,
                message=f"Database connectivity working"
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                name="Database Test",
                status="FAIL",
                duration_ms=duration,
                message=f"Database error: {str(e)}"
            )

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Legal RAG System Test Runner")
    parser.add_argument("--quick", action="store_true", help="Quick validation tests")
    parser.add_argument("--unit", action="store_true", help="Unit tests")
    parser.add_argument("--all", action="store_true", help="All available tests")
    parser.add_argument("--db", action="store_true", help="Database tests")
    parser.add_argument("--env", action="store_true", help="Environment validation")

    args = parser.parse_args()

    runner = SimpleTestRunner()
    all_results = []

    try:
        # Environment validation if requested
        if args.env:
            env_result = runner.validate_environment()
            all_results.append(env_result)

        # Database test if requested
        if args.db:
            db_result = runner.run_database_test()
            all_results.append(db_result)

        # Quick validation
        if args.quick or not any([args.unit, args.all, args.db, args.env]):
            quick_results = runner.run_quick_validation()
            all_results.extend(quick_results)

        # Unit tests
        elif args.unit:
            unit_results = runner.run_unit_tests()
            all_results.extend(unit_results)

        # All tests
        elif args.all:
            all_test_results = runner.run_all_available_tests()
            all_results.extend(all_test_results)

        # Print results
        runner.print_results(all_results)

        # Exit with appropriate code
        failed_count = sum(1 for r in all_results if r.status in ["FAIL", "ERROR"])

        if failed_count == 0:
            print("🎉 All tests completed successfully!")
            sys.exit(0)
        else:
            print(f"⚠️  {failed_count} test(s) failed")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Test execution interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {str(e)}")
        sys.exit(2)


if __name__ == "__main__":
    main()
