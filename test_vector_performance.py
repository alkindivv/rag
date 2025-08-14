#!/usr/bin/env python3
"""
Comprehensive performance test for vector search HNSW index usage.

This test validates:
1. Vector search query format compatibility with pgvector
2. HNSW index usage verification
3. Performance measurements before/after optimizations
4. Legal keyword detection functionality
5. Search type routing optimization

Run this test to verify the performance fixes are working correctly.
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.search.vector_search import VectorSearchService
from src.services.embedding.embedder import JinaV4Embedder
from src.db.session import get_db_session
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorPerformanceTest:
    """Comprehensive test suite for vector search performance."""

    def __init__(self):
        """Initialize test components."""
        print("🔧 Initializing Vector Performance Test...")

        try:
            self.embedder = JinaV4Embedder()
            self.search_service = VectorSearchService(embedder=self.embedder, default_k=5)
            print("✅ Services initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize services: {e}")
            sys.exit(1)

    def test_legal_keyword_detection(self) -> Dict[str, bool]:
        """Test the legal keyword detection functionality."""
        print("\n📝 Testing Legal Keyword Detection...")

        test_cases = [
            # Should detect legal keywords
            ("Ekosistem Ekonomi Kreatif diatur dalam undang undang apa?", True),
            ("definisi badan hukum dalam peraturan", True),
            ("sanksi pidana korupsi", True),
            ("UU No 8 Tahun 2019 Pasal 6", True),
            ("tanggung jawab sosial perusahaan", True),
            ("ketentuan pidana dalam undang-undang", True),

            # Should NOT detect legal keywords
            ("what is the weather today", False),
            ("cara memasak nasi goreng", False),
            ("film terbaik tahun ini", False),
            ("teknologi artificial intelligence", False),
        ]

        results = {}
        passed = 0
        total = len(test_cases)

        for query, expected in test_cases:
            detected = self.search_service._detect_legal_keywords(query)
            status = "✅" if detected == expected else "❌"
            print(f"  {status} '{query[:40]}...' -> Expected: {expected}, Got: {detected}")
            results[query] = detected == expected
            if detected == expected:
                passed += 1

        print(f"\n📊 Legal Keyword Detection: {passed}/{total} tests passed")
        return results

    def test_vector_format_compatibility(self) -> bool:
        """Test different vector formats to find the one that works with pgvector."""
        print("\n🔧 Testing Vector Format Compatibility...")

        # Generate a test embedding
        try:
            test_embedding = self.embedder.embed_single("test query for vector format")
            print(f"✅ Generated test embedding: {len(test_embedding)} dimensions")
        except Exception as e:
            print(f"❌ Failed to generate test embedding: {e}")
            return False

        # Test different vector formats
        vector_formats = [
            ("String Format", f"[{','.join(map(str, test_embedding))}]"),
            ("Array Format", test_embedding),
            ("String with Spaces", f"[{', '.join(map(str, test_embedding))}]"),
        ]

        with get_db_session() as db:
            for format_name, vector_param in vector_formats:
                print(f"\n🧪 Testing {format_name}...")

                test_query = """
                SELECT COUNT(*) as count
                FROM document_vectors dv
                WHERE dv.embedding IS NOT NULL
                AND dv.embedding <=> :query_vector IS NOT NULL
                LIMIT 1
                """

                try:
                    result = db.execute(text(test_query), {'query_vector': vector_param})
                    count = result.fetchone()
                    print(f"  ✅ {format_name}: Query executed successfully")
                    return True
                except Exception as e:
                    print(f"  ❌ {format_name}: {str(e)[:100]}...")
                    continue

        print("❌ No compatible vector format found!")
        return False

    def test_hnsw_index_usage(self) -> bool:
        """Test if HNSW index is being used in vector queries."""
        print("\n📊 Testing HNSW Index Usage...")

        # Generate test embedding
        try:
            test_embedding = self.embedder.embed_single("test query for index usage")
        except Exception as e:
            print(f"❌ Failed to generate embedding: {e}")
            return False

        # Build the actual query used by the system
        test_query = """
        EXPLAIN ANALYZE
        SELECT
            dv.unit_id,
            dv.doc_form,
            dv.doc_year,
            (1 - (dv.embedding <=> :query_vector)) AS similarity_score
        FROM document_vectors dv
        JOIN legal_units lu ON lu.unit_id = dv.unit_id
        WHERE dv.content_type = 'pasal'
        AND lu.unit_type = 'PASAL'
        ORDER BY dv.embedding <=> :query_vector
        LIMIT 5
        """

        with get_db_session() as db:
            try:
                # Test with string format (current implementation)
                vector_string = f"[{','.join(map(str, test_embedding))}]"
                result = db.execute(text(test_query), {'query_vector': vector_string})
                query_plan = result.fetchall()

                # Analyze query plan
                plan_text = str(query_plan).lower()
                uses_hnsw = "index scan" in plan_text and "hnsw" in plan_text
                uses_seq_scan = "seq scan" in plan_text

                print("📋 Query Execution Plan Analysis:")
                for row in query_plan:
                    print(f"  {row[0]}")

                if uses_hnsw:
                    print("✅ HNSW index is being used!")
                    return True
                elif uses_seq_scan:
                    print("❌ Sequential scan detected - HNSW index NOT used!")
                    print("💡 This indicates a performance problem")
                    return False
                else:
                    print("⚠️  Unclear if HNSW index is used - check plan above")
                    return False

            except Exception as e:
                print(f"❌ Failed to analyze query plan: {e}")
                return False

    def test_search_performance(self) -> Dict[str, float]:
        """Test search performance with different query types."""
        print("\n⏱️  Testing Search Performance...")

        test_queries = [
            ("UU 8/2019 Pasal 6", "citation"),
            ("definisi badan hukum", "legal_semantic"),
            ("sanksi pidana korupsi", "legal_semantic"),
            ("tanggung jawab sosial perusahaan", "legal_semantic"),
            ("artificial intelligence technology", "general_semantic"),
        ]

        performance_results = {}

        for query, query_type in test_queries:
            print(f"\n🔍 Testing: '{query}' ({query_type})")

            start_time = time.time()
            try:
                results = self.search_service.search(query, k=5)
                duration = (time.time() - start_time) * 1000  # Convert to ms

                result_count = len(results.get('results', []))
                search_type = results.get('metadata', {}).get('search_type', 'unknown')

                print(f"  ⏱️  Duration: {duration:.1f}ms")
                print(f"  📊 Results: {result_count}")
                print(f"  🔍 Search Type: {search_type}")

                performance_results[query] = duration

                # Performance expectations
                if query_type == "citation" and duration > 500:
                    print(f"  ⚠️  Citation query slower than expected (>{500}ms)")
                elif query_type in ["legal_semantic", "general_semantic"] and duration > 5000:
                    print(f"  ⚠️  Semantic query slower than expected (>{5000}ms)")
                else:
                    print(f"  ✅ Performance within acceptable range")

            except Exception as e:
                print(f"  ❌ Query failed: {e}")
                performance_results[query] = -1

        return performance_results

    def test_database_connectivity(self) -> bool:
        """Test basic database connectivity and vector extension."""
        print("\n🔌 Testing Database Connectivity...")

        try:
            with get_db_session() as db:
                # Test basic connectivity
                result = db.execute(text("SELECT 1 as test"))
                assert result.fetchone()[0] == 1
                print("  ✅ Database connection working")

                # Test pgvector extension
                result = db.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
                if result.fetchone():
                    print("  ✅ pgvector extension installed")
                else:
                    print("  ❌ pgvector extension not found!")
                    return False

                # Test vector data exists
                result = db.execute(text("SELECT COUNT(*) FROM document_vectors"))
                count = result.fetchone()[0]
                print(f"  📊 Document vectors in database: {count}")

                if count == 0:
                    print("  ⚠️  No vectors found - database may need indexing")

                return True

        except Exception as e:
            print(f"  ❌ Database test failed: {e}")
            return False

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        print("🚀 Starting Comprehensive Vector Performance Test")
        print("=" * 60)

        results = {
            'timestamp': time.time(),
            'tests': {}
        }

        # Run all tests
        try:
            results['tests']['database_connectivity'] = self.test_database_connectivity()
            results['tests']['legal_keyword_detection'] = self.test_legal_keyword_detection()
            results['tests']['vector_format_compatibility'] = self.test_vector_format_compatibility()
            results['tests']['hnsw_index_usage'] = self.test_hnsw_index_usage()
            results['tests']['search_performance'] = self.test_search_performance()

        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
            return results
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            results['error'] = str(e)
            return results

        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        total_tests = 0
        passed_tests = 0

        for test_name, result in results['tests'].items():
            if isinstance(result, bool):
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{test_name}: {status}")
                total_tests += 1
                passed_tests += result
            elif isinstance(result, dict):
                passed = sum(1 for v in result.values() if v)
                total = len(result)
                print(f"{test_name}: {passed}/{total} subtests passed")
                total_tests += total
                passed_tests += passed

        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print("🎉 All tests PASSED! Vector search is optimized.")
        elif passed_tests > total_tests * 0.8:
            print("⚠️  Most tests passed, but some optimizations needed.")
        else:
            print("❌ Multiple tests failed - significant issues detected.")

        return results

def main():
    """Main test execution."""
    if not os.path.exists('src'):
        print("❌ Please run this test from the project root directory")
        sys.exit(1)

    # Create and run test
    test_suite = VectorPerformanceTest()
    results = test_suite.run_comprehensive_test()

    # Save results
    import json
    with open('vector_performance_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📁 Results saved to: vector_performance_test_results.json")

if __name__ == "__main__":
    main()
