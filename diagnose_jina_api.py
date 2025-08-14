#!/usr/bin/env python3
"""
Jina API Diagnostic Script
=========================

This script diagnoses connectivity and performance issues with the Jina AI embedding service.
Run this when experiencing timeout errors or embedding failures.

Usage:
    python diagnose_jina_api.py
    python diagnose_jina_api.py --verbose
    python diagnose_jina_api.py --quick-test
"""

import os
import sys
import time
import json
import asyncio
import argparse
from typing import Dict, List, Optional, Tuple
import httpx
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.config.settings import settings
    from src.services.embedding.embedder import JinaV4Embedder
    from src.utils.http import HttpClient
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

@dataclass
class DiagnosticResult:
    """Result of a diagnostic test."""
    test_name: str
    success: bool
    duration_ms: float
    error: Optional[str] = None
    details: Optional[Dict] = None

class JinaAPIDiagnostics:
    """Comprehensive Jina API diagnostics."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[DiagnosticResult] = []

        # Test configurations
        self.test_queries = [
            "definisi badan hukum",
            "sanksi pidana korupsi",
            "tanggung jawab sosial perusahaan",
            "UU 8/2019 Pasal 6"  # This should trigger citation parsing
        ]

        self.timeout_tests = [5, 15, 30, 60, 120]  # Different timeout values to test

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔍"
        }.get(level, "📝")

        print(f"[{timestamp}] {prefix} {message}")

    def measure_time(self, func, *args, **kwargs) -> Tuple[any, float]:
        """Measure execution time of a function."""
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = (time.time() - start) * 1000
            return result, duration
        except Exception as e:
            duration = (time.time() - start) * 1000
            raise e

    def test_environment_setup(self) -> DiagnosticResult:
        """Test basic environment configuration."""
        start_time = time.time()

        try:
            # Check API key
            api_key = settings.jina_api_key
            if not api_key:
                return DiagnosticResult(
                    "Environment Setup",
                    False,
                    (time.time() - start_time) * 1000,
                    "JINA_API_KEY not set in environment"
                )

            if api_key == "test-key":
                return DiagnosticResult(
                    "Environment Setup",
                    False,
                    (time.time() - start_time) * 1000,
                    "JINA_API_KEY is set to test value, not real API key"
                )

            # Check other settings
            details = {
                "jina_api_key": f"{api_key[:8]}..." if len(api_key) > 8 else "too_short",
                "jina_embed_base": settings.jina_embed_base,
                "jina_embed_model": settings.jina_embed_model,
                "embedding_model": settings.embedding_model,
                "embedding_dims": settings.embedding_dims
            }

            return DiagnosticResult(
                "Environment Setup",
                True,
                (time.time() - start_time) * 1000,
                details=details
            )

        except Exception as e:
            return DiagnosticResult(
                "Environment Setup",
                False,
                (time.time() - start_time) * 1000,
                str(e)
            )

    def test_network_connectivity(self) -> DiagnosticResult:
        """Test basic network connectivity to Jina API."""
        start_time = time.time()

        try:
            # Test basic HTTP connectivity
            with httpx.Client(timeout=10.0) as client:
                response = client.get("https://api.jina.ai/", timeout=10.0)

            details = {
                "status_code": response.status_code,
                "response_time_ms": (time.time() - start_time) * 1000,
                "headers": dict(response.headers) if self.verbose else None
            }

            success = response.status_code in [200, 404, 405]  # 404/405 acceptable for base URL

            return DiagnosticResult(
                "Network Connectivity",
                success,
                (time.time() - start_time) * 1000,
                None if success else f"Unexpected status code: {response.status_code}",
                details
            )

        except Exception as e:
            return DiagnosticResult(
                "Network Connectivity",
                False,
                (time.time() - start_time) * 1000,
                f"Network error: {str(e)}"
            )

    def test_api_authentication(self) -> DiagnosticResult:
        """Test API key authentication."""
        start_time = time.time()

        try:
            # Test with minimal embedding request
            headers = {
                "Authorization": f"Bearer {settings.jina_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": settings.jina_embed_model,
                "task": "retrieval.query",
                "dimensions": 384,
                "input": ["test"]
            }

            with httpx.Client(timeout=15.0) as client:
                response = client.post(
                    settings.jina_embed_base,
                    headers=headers,
                    json=payload,
                    timeout=15.0
                )

            details = {
                "status_code": response.status_code,
                "response_time_ms": (time.time() - start_time) * 1000
            }

            if response.status_code == 200:
                data = response.json()
                details["embedding_length"] = len(data.get("data", [{}])[0].get("embedding", []))
                return DiagnosticResult("API Authentication", True, (time.time() - start_time) * 1000, details=details)
            elif response.status_code == 401:
                return DiagnosticResult("API Authentication", False, (time.time() - start_time) * 1000, "Invalid API key")
            elif response.status_code == 429:
                return DiagnosticResult("API Authentication", False, (time.time() - start_time) * 1000, "Rate limit exceeded")
            else:
                return DiagnosticResult("API Authentication", False, (time.time() - start_time) * 1000, f"HTTP {response.status_code}: {response.text[:200]}")

        except httpx.TimeoutException:
            return DiagnosticResult("API Authentication", False, (time.time() - start_time) * 1000, "Request timeout during authentication test")
        except Exception as e:
            return DiagnosticResult("API Authentication", False, (time.time() - start_time) * 1000, str(e))

    def test_embedding_service_basic(self) -> DiagnosticResult:
        """Test basic embedding service functionality."""
        start_time = time.time()

        try:
            # Initialize embedder
            embedder = JinaV4Embedder()

            # Test single embedding
            test_text = "test embedding"
            embedding, duration = self.measure_time(
                embedder.embed_single,
                test_text,
                task="retrieval.query",
                dims=384
            )

            details = {
                "embedding_length": len(embedding) if embedding else 0,
                "expected_length": 384,
                "embedding_duration_ms": duration
            }

            success = embedding is not None and len(embedding) == 384
            error = None if success else f"Expected 384-dim vector, got {len(embedding) if embedding else 0}"

            return DiagnosticResult(
                "Embedding Service Basic",
                success,
                (time.time() - start_time) * 1000,
                error,
                details
            )

        except Exception as e:
            return DiagnosticResult(
                "Embedding Service Basic",
                False,
                (time.time() - start_time) * 1000,
                str(e)
            )

    def test_timeout_resilience(self) -> List[DiagnosticResult]:
        """Test embedding service with different timeout configurations."""
        results = []

        for timeout in self.timeout_tests:
            start_time = time.time()

            try:
                # Create HTTP client with specific timeout
                http_client = HttpClient(timeout=timeout, max_retries=1)
                embedder = JinaV4Embedder(client=http_client)

                # Test with medium-length text
                test_text = "definisi badan hukum dalam peraturan perundang-undangan"

                embedding, duration = self.measure_time(
                    embedder.embed_single,
                    test_text,
                    task="retrieval.query",
                    dims=384
                )

                details = {
                    "timeout_setting": timeout,
                    "actual_duration_ms": duration,
                    "embedding_success": embedding is not None,
                    "embedding_length": len(embedding) if embedding else 0
                }

                success = embedding is not None and len(embedding) == 384

                results.append(DiagnosticResult(
                    f"Timeout Test ({timeout}s)",
                    success,
                    (time.time() - start_time) * 1000,
                    None if success else "Embedding failed",
                    details
                ))

            except Exception as e:
                results.append(DiagnosticResult(
                    f"Timeout Test ({timeout}s)",
                    False,
                    (time.time() - start_time) * 1000,
                    str(e)
                ))

        return results

    def test_batch_embedding(self) -> DiagnosticResult:
        """Test batch embedding functionality."""
        start_time = time.time()

        try:
            embedder = JinaV4Embedder()

            # Test batch of queries
            batch_texts = self.test_queries[:3]  # Use first 3 queries

            embeddings, duration = self.measure_time(
                embedder.embed_texts,
                batch_texts,
                task="retrieval.query",
                dims=384
            )

            details = {
                "batch_size": len(batch_texts),
                "embeddings_returned": len(embeddings) if embeddings else 0,
                "embedding_duration_ms": duration,
                "avg_time_per_text": duration / len(batch_texts) if batch_texts else 0
            }

            success = (embeddings is not None and
                      len(embeddings) == len(batch_texts) and
                      all(len(emb) == 384 for emb in embeddings))

            error = None if success else "Batch embedding failed or returned wrong dimensions"

            return DiagnosticResult(
                "Batch Embedding",
                success,
                (time.time() - start_time) * 1000,
                error,
                details
            )

        except Exception as e:
            return DiagnosticResult(
                "Batch Embedding",
                False,
                (time.time() - start_time) * 1000,
                str(e)
            )

    def test_search_integration(self) -> DiagnosticResult:
        """Test integration with vector search service."""
        start_time = time.time()

        try:
            from src.services.search.vector_search import VectorSearchService

            # Initialize search service (this should work without DB for embedding test)
            search_service = VectorSearchService()

            # Test query embedding only (not full search)
            test_query = "definisi badan hukum"

            # Test the internal embedding function
            embedding, duration = self.measure_time(
                search_service._embed_query,
                test_query
            )

            details = {
                "query": test_query,
                "embedding_duration_ms": duration,
                "embedding_length": len(embedding) if embedding else 0
            }

            success = embedding is not None and len(embedding) == 384
            error = None if success else "Query embedding failed in search service"

            return DiagnosticResult(
                "Search Integration",
                success,
                (time.time() - start_time) * 1000,
                error,
                details
            )

        except Exception as e:
            return DiagnosticResult(
                "Search Integration",
                False,
                (time.time() - start_time) * 1000,
                str(e)
            )

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Check for common issues
        auth_test = next((r for r in self.results if r.test_name == "API Authentication"), None)
        network_test = next((r for r in self.results if r.test_name == "Network Connectivity"), None)
        basic_test = next((r for r in self.results if r.test_name == "Embedding Service Basic"), None)

        if not auth_test or not auth_test.success:
            recommendations.append("🔑 Check your JINA_API_KEY - get a free key at: https://jina.ai/?sui=apikey")

        if not network_test or not network_test.success:
            recommendations.append("🌐 Check your internet connection and firewall settings")
            recommendations.append("🌐 Try using a VPN if behind corporate firewall")

        if basic_test and not basic_test.success and "timeout" in basic_test.error.lower():
            recommendations.append("⏱️ Increase timeout: Set EMBEDDING_REQUEST_TIMEOUT=60 in environment")
            recommendations.append("⏱️ Reduce batch size: Set EMBEDDING_BATCH_SIZE=10 in environment")

        # Check timeout test results
        timeout_results = [r for r in self.results if "Timeout Test" in r.test_name]
        successful_timeouts = [r for r in timeout_results if r.success]

        if successful_timeouts:
            min_working_timeout = min(r.details["timeout_setting"] for r in successful_timeouts)
            recommendations.append(f"⏱️ Use timeout of at least {min_working_timeout}s: EMBEDDING_REQUEST_TIMEOUT={min_working_timeout}")

        if not successful_timeouts and timeout_results:
            recommendations.append("🚨 All timeout tests failed - check Jina API status: https://status.jina.ai/")
            recommendations.append("🔄 Try again later - API might be experiencing issues")

        # Performance recommendations
        batch_test = next((r for r in self.results if r.test_name == "Batch Embedding"), None)
        if batch_test and batch_test.success and batch_test.details:
            avg_time = batch_test.details.get("avg_time_per_text", 0)
            if avg_time > 1000:  # > 1 second per text
                recommendations.append("🚀 Consider using smaller batch sizes for better responsiveness")

        return recommendations

    def run_all_tests(self, quick: bool = False) -> None:
        """Run all diagnostic tests."""
        self.log("🔍 Starting Jina API Diagnostics", "INFO")
        self.log("=" * 50, "DEBUG")

        # Environment setup
        self.log("Testing environment setup...", "INFO")
        result = self.test_environment_setup()
        self.results.append(result)
        self.log(f"Environment: {'✅ PASS' if result.success else '❌ FAIL'} ({result.duration_ms:.1f}ms)",
                "SUCCESS" if result.success else "ERROR")
        if result.error:
            self.log(f"  Error: {result.error}", "ERROR")
        if self.verbose and result.details:
            self.log(f"  Details: {json.dumps(result.details, indent=2)}", "DEBUG")

        # Network connectivity
        self.log("Testing network connectivity...", "INFO")
        result = self.test_network_connectivity()
        self.results.append(result)
        self.log(f"Network: {'✅ PASS' if result.success else '❌ FAIL'} ({result.duration_ms:.1f}ms)",
                "SUCCESS" if result.success else "ERROR")
        if result.error:
            self.log(f"  Error: {result.error}", "ERROR")

        # API authentication
        self.log("Testing API authentication...", "INFO")
        result = self.test_api_authentication()
        self.results.append(result)
        self.log(f"Authentication: {'✅ PASS' if result.success else '❌ FAIL'} ({result.duration_ms:.1f}ms)",
                "SUCCESS" if result.success else "ERROR")
        if result.error:
            self.log(f"  Error: {result.error}", "ERROR")

        # Basic embedding
        self.log("Testing basic embedding service...", "INFO")
        result = self.test_embedding_service_basic()
        self.results.append(result)
        self.log(f"Basic Embedding: {'✅ PASS' if result.success else '❌ FAIL'} ({result.duration_ms:.1f}ms)",
                "SUCCESS" if result.success else "ERROR")
        if result.error:
            self.log(f"  Error: {result.error}", "ERROR")

        if not quick:
            # Timeout resilience tests
            self.log("Testing timeout resilience...", "INFO")
            timeout_results = self.test_timeout_resilience()
            self.results.extend(timeout_results)
            for result in timeout_results:
                self.log(f"  {result.test_name}: {'✅ PASS' if result.success else '❌ FAIL'} ({result.duration_ms:.1f}ms)",
                        "SUCCESS" if result.success else "ERROR")

            # Batch embedding
            self.log("Testing batch embedding...", "INFO")
            result = self.test_batch_embedding()
            self.results.append(result)
            self.log(f"Batch Embedding: {'✅ PASS' if result.success else '❌ FAIL'} ({result.duration_ms:.1f}ms)",
                    "SUCCESS" if result.success else "ERROR")
            if result.error:
                self.log(f"  Error: {result.error}", "ERROR")

            # Search integration
            self.log("Testing search service integration...", "INFO")
            result = self.test_search_integration()
            self.results.append(result)
            self.log(f"Search Integration: {'✅ PASS' if result.success else '❌ FAIL'} ({result.duration_ms:.1f}ms)",
                    "SUCCESS" if result.success else "ERROR")
            if result.error:
                self.log(f"  Error: {result.error}", "ERROR")

    def print_summary(self) -> None:
        """Print diagnostic summary and recommendations."""
        self.log("=" * 50, "DEBUG")
        self.log("📊 DIAGNOSTIC SUMMARY", "INFO")
        self.log("=" * 50, "DEBUG")

        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.success])

        self.log(f"Tests Passed: {passed_tests}/{total_tests}", "SUCCESS" if passed_tests == total_tests else "WARNING")

        if passed_tests < total_tests:
            self.log("Failed Tests:", "ERROR")
            for result in self.results:
                if not result.success:
                    self.log(f"  ❌ {result.test_name}: {result.error}", "ERROR")

        # Generate and display recommendations
        recommendations = self.generate_recommendations()
        if recommendations:
            self.log("", "DEBUG")
            self.log("🔧 RECOMMENDATIONS", "INFO")
            self.log("-" * 30, "DEBUG")
            for rec in recommendations:
                self.log(rec, "WARNING")

        # Overall status
        if passed_tests == total_tests:
            self.log("", "DEBUG")
            self.log("🎉 All tests passed! Jina API is working correctly.", "SUCCESS")
        else:
            self.log("", "DEBUG")
            self.log("⚠️ Some tests failed. Follow recommendations above to resolve issues.", "WARNING")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Diagnose Jina API connectivity and performance")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quick-test", "-q", action="store_true", help="Run only basic tests")

    args = parser.parse_args()

    diagnostics = JinaAPIDiagnostics(verbose=args.verbose)

    try:
        diagnostics.run_all_tests(quick=args.quick_test)
        diagnostics.print_summary()

        # Exit with appropriate code
        failed_tests = len([r for r in diagnostics.results if not r.success])
        sys.exit(0 if failed_tests == 0 else 1)

    except KeyboardInterrupt:
        diagnostics.log("Diagnostics interrupted by user", "WARNING")
        sys.exit(130)
    except Exception as e:
        diagnostics.log(f"Unexpected error during diagnostics: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
