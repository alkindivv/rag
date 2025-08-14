#!/usr/bin/env python3
"""
Simple API test script for Legal RAG Dense Search System.

Tests the FastAPI endpoints to ensure the vector search and citation parsing
are working correctly through the API layer.

Run with: python test_api.py
"""

import json
import time
import requests
from typing import Dict, Any, List
import sys


class APITester:
    """Simple API tester for Legal RAG endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API tester."""
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []

    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint."""
        print("🏥 Testing Health Endpoint")
        print("-" * 30)

        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed")
                print(f"   Status: {data.get('status')}")
                print(f"   Service: {data.get('service')}")
                return True
            else:
                print(f"❌ Health check failed with status {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"❌ Health check failed: {e}")
            print("   Make sure the API server is running on http://localhost:8000")
            return False

    def test_search_endpoint(self) -> bool:
        """Test search endpoint with various query types."""
        print("\n🔍 Testing Search Endpoint")
        print("-" * 30)

        test_cases = [
            {
                "name": "Citation Query - UU Format",
                "query": "UU 8/2019 Pasal 6 ayat (2)",
                "expected_type": "explicit_citation",
                "min_results": 1
            },
            {
                "name": "Citation Query - PP Format",
                "query": "PP 45/2020 Pasal 12",
                "expected_type": "explicit_citation",
                "min_results": 1
            },
            {
                "name": "Contextual Query - Definition",
                "query": "definisi badan hukum",
                "expected_type": "contextual_semantic",
                "min_results": 0  # May not have data in test environment
            },
            {
                "name": "Contextual Query - Sanctions",
                "query": "sanksi pidana korupsi",
                "expected_type": "contextual_semantic",
                "min_results": 0
            },
            {
                "name": "Empty Query",
                "query": "",
                "expected_type": "error",
                "min_results": 0
            }
        ]

        passed_tests = 0
        total_tests = len(test_cases)

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['name']}")
            print(f"   Query: '{test_case['query']}'")

            start_time = time.time()

            try:
                # Test POST endpoint
                payload = {
                    "query": test_case['query'],
                    "limit": 10,
                    "use_reranking": False
                }

                response = self.session.post(
                    f"{self.base_url}/search",
                    json=payload,
                    timeout=10
                )

                duration_ms = int((time.time() - start_time) * 1000)
                print(f"   Response time: {duration_ms}ms")

                if response.status_code == 200:
                    data = response.json()

                    # Validate response structure
                    if 'results' in data and 'metadata' in data:
                        results = data['results']
                        metadata = data['metadata']

                        print(f"   Results count: {len(results)}")
                        print(f"   Search type: {metadata.get('search_type', 'unknown')}")

                        # Check search type if not error case
                        if test_case['expected_type'] != 'error':
                            expected_type = test_case['expected_type']
                            actual_type = metadata.get('search_type')

                            type_correct = actual_type == expected_type
                            results_sufficient = len(results) >= test_case['min_results']

                            if type_correct and results_sufficient:
                                print("   Result: ✅ PASS")
                                passed_tests += 1
                            else:
                                print(f"   Result: ❌ FAIL")
                                if not type_correct:
                                    print(f"     Expected search type: {expected_type}, got: {actual_type}")
                                if not results_sufficient:
                                    print(f"     Expected min results: {test_case['min_results']}, got: {len(results)}")
                        else:
                            # Error case - should have handled gracefully
                            if len(results) == 0:
                                print("   Result: ✅ PASS (Error handled gracefully)")
                                passed_tests += 1
                            else:
                                print("   Result: ❌ FAIL (Should return no results for error case)")

                    else:
                        print("   Result: ❌ FAIL (Invalid response structure)")

                elif response.status_code == 422 and test_case['expected_type'] == 'error':
                    print("   Result: ✅ PASS (Validation error handled correctly)")
                    passed_tests += 1
                else:
                    print(f"   Result: ❌ FAIL (HTTP {response.status_code})")
                    try:
                        error_data = response.json()
                        print(f"     Error: {error_data.get('detail', 'Unknown error')}")
                    except:
                        print(f"     Raw response: {response.text[:100]}...")

            except requests.exceptions.Timeout:
                print(f"   Result: ❌ FAIL (Timeout after 10s)")
            except requests.exceptions.RequestException as e:
                print(f"   Result: ❌ FAIL (Request error: {e})")
            except Exception as e:
                print(f"   Result: ❌ FAIL (Unexpected error: {e})")

        print(f"\n📊 Search Test Results: {passed_tests}/{total_tests} passed")
        return passed_tests == total_tests

    def test_get_search_endpoint(self) -> bool:
        """Test GET search endpoint for browser compatibility."""
        print("\n🌐 Testing GET Search Endpoint")
        print("-" * 30)

        try:
            # Test GET endpoint with query parameters
            params = {
                "query": "UU 8/2019 Pasal 6",
                "limit": 5,
                "use_reranking": False
            }

            response = self.session.get(
                f"{self.base_url}/search",
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                print("✅ GET search endpoint working")
                print(f"   Results: {len(data.get('results', []))}")
                print(f"   Search type: {data.get('metadata', {}).get('search_type', 'unknown')}")
                return True
            else:
                print(f"❌ GET search failed with status {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ GET search test failed: {e}")
            return False

    def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks for different query types."""
        print("\n⚡ Testing Performance Benchmarks")
        print("-" * 30)

        # Performance test queries
        citation_queries = [
            "UU 8/2019 Pasal 6",
            "PP 45/2020 Pasal 12",
            "Pasal 15 ayat (1)"
        ]

        contextual_queries = [
            "definisi badan hukum",
            "sanksi pidana",
            "tanggung jawab sosial"
        ]

        citation_times = []
        contextual_times = []

        # Test citation query performance
        print("Citation Query Performance:")
        for query in citation_queries:
            start_time = time.time()
            try:
                response = self.session.post(
                    f"{self.base_url}/search",
                    json={"query": query, "limit": 5},
                    timeout=5
                )
                duration_ms = int((time.time() - start_time) * 1000)
                citation_times.append(duration_ms)
                print(f"  '{query}': {duration_ms}ms")
            except:
                print(f"  '{query}': FAILED")

        # Test contextual query performance
        print("\nContextual Query Performance:")
        for query in contextual_queries:
            start_time = time.time()
            try:
                response = self.session.post(
                    f"{self.base_url}/search",
                    json={"query": query, "limit": 10},
                    timeout=5
                )
                duration_ms = int((time.time() - start_time) * 1000)
                contextual_times.append(duration_ms)
                print(f"  '{query}': {duration_ms}ms")
            except:
                print(f"  '{query}': FAILED")

        # Analyze performance
        if citation_times:
            avg_citation = sum(citation_times) / len(citation_times)
            print(f"\nCitation queries average: {avg_citation:.0f}ms")
            citation_ok = avg_citation < 100  # Should be very fast
        else:
            citation_ok = False

        if contextual_times:
            avg_contextual = sum(contextual_times) / len(contextual_times)
            print(f"Contextual queries average: {avg_contextual:.0f}ms")
            contextual_ok = avg_contextual < 500  # Vector search can be slower
        else:
            contextual_ok = False

        performance_ok = citation_ok and contextual_ok
        if performance_ok:
            print("✅ Performance benchmarks passed")
        else:
            print("⚠️  Performance benchmarks did not meet targets")

        return performance_ok

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all API tests and return results."""
        print("🚀 Legal RAG API Test Suite")
        print("=" * 50)

        results = {}

        # Run individual tests
        results['health'] = self.test_health_endpoint()
        results['search_post'] = self.test_search_endpoint()
        results['search_get'] = self.test_get_search_endpoint()
        results['performance'] = self.test_performance_benchmarks()

        return results

    def print_summary(self, results: Dict[str, bool]) -> None:
        """Print test summary."""
        print("\n" + "=" * 50)
        print("📋 Test Summary")
        print("-" * 20)

        passed_count = sum(results.values())
        total_count = len(results)

        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")

        print(f"\nOverall: {passed_count}/{total_count} test categories passed")

        if passed_count == total_count:
            print("🎉 All API tests passed! System is ready for use.")
        else:
            print(f"⚠️  {total_count - passed_count} test categories failed.")


def main():
    """Main test execution."""
    print("Starting API tests...")
    print("Make sure the API server is running: python src/api/main.py\n")

    tester = APITester()

    try:
        results = tester.run_all_tests()
        tester.print_summary(results)

        # Exit with appropriate code
        if all(results.values()):
            print("\n✅ All tests passed - API is functioning correctly")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed - please check the API implementation")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
