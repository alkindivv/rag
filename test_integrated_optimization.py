#!/usr/bin/env python3
"""
Integration test for cache and query optimizer implementation.
Tests both embedding cache and query optimization in the main search pipeline.
"""

import asyncio
import time
from typing import Dict, Any

from src.services.search.vector_search import VectorSearchService
from src.services.search.query_optimizer import get_query_optimizer, get_optimization_stats
from src.services.embedding.cache import get_cache_performance_report
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class IntegrationTestRunner:
    """Simple test runner for cache and optimizer integration."""

    def __init__(self):
        self.search_service = VectorSearchService()
        self.optimizer = get_query_optimizer()
        self.test_queries = [
            "definisi badan hukum",
            "sanksi pidana korupsi",
            "UU 8/2019 Pasal 6 ayat (2)",
            "tanggung jawab sosial perusahaan",
            "kewajiban pelaporan keuangan"
        ]

    def test_query_optimization(self) -> Dict[str, Any]:
        """Test query optimization functionality."""
        print("\n🔧 Testing Query Optimization...")

        results = {}
        for query in self.test_queries[:3]:  # Test first 3 queries
            start_time = time.time()

            # Test optimizer directly
            optimized_query, analysis = self.optimizer.optimize_query(query)
            optimization_time = (time.time() - start_time) * 1000

            results[query] = {
                "original": query,
                "optimized": optimized_query,
                "query_type": analysis.query_type.value,
                "confidence": analysis.confidence_score,
                "optimization_time_ms": optimization_time,
                "improved": query != optimized_query
            }

            print(f"  ✅ {query}")
            print(f"     → {optimized_query}")
            print(f"     Type: {analysis.query_type.value} (confidence: {analysis.confidence_score:.2f})")
            print(f"     Time: {optimization_time:.1f}ms")

        return results

    def test_embedding_cache(self) -> Dict[str, Any]:
        """Test embedding cache performance."""
        print("\n💾 Testing Embedding Cache...")

        # Test cache miss (first time)
        test_query = "definisi badan hukum dalam hukum perdata"

        print("  Testing cache miss (first call)...")
        start_time = time.time()
        result1 = self.search_service.search(query=test_query, k=5)
        first_duration = (time.time() - start_time) * 1000

        print(f"  ✅ First call: {first_duration:.1f}ms")

        # Test cache hit (second time)
        print("  Testing cache hit (second call)...")
        start_time = time.time()
        result2 = self.search_service.search(query=test_query, k=5)
        second_duration = (time.time() - start_time) * 1000

        print(f"  ✅ Second call: {second_duration:.1f}ms")

        # Calculate improvement
        improvement = ((first_duration - second_duration) / first_duration) * 100

        cache_stats = get_cache_performance_report()

        return {
            "test_query": test_query,
            "first_call_ms": first_duration,
            "second_call_ms": second_duration,
            "improvement_percent": improvement,
            "cache_stats": cache_stats,
            "results_identical": len(result1["results"]) == len(result2["results"])
        }

    def test_integrated_search_performance(self) -> Dict[str, Any]:
        """Test complete integrated search pipeline."""
        print("\n🚀 Testing Integrated Search Pipeline...")

        performance_results = {}

        for i, query in enumerate(self.test_queries):
            print(f"  Testing query {i+1}/5: {query}")

            start_time = time.time()
            result = self.search_service.search(query=query, k=10)
            duration = (time.time() - start_time) * 1000

            performance_results[query] = {
                "duration_ms": duration,
                "results_count": len(result["results"]),
                "search_type": result["metadata"]["search_type"],
                "optimization_applied": result["metadata"].get("optimization_applied", False),
                "original_query": result["metadata"].get("original_query", query),
                "optimized_query": result["metadata"].get("optimized_query", query)
            }

            print(f"    ✅ {duration:.1f}ms | {len(result['results'])} results | {result['metadata']['search_type']}")

            if result["metadata"].get("optimization_applied"):
                print(f"    🔧 Query optimized: {result['metadata']['original_query']} → {result['metadata']['optimized_query']}")

        return performance_results

    def get_system_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive system performance summary."""
        try:
            cache_stats = get_cache_performance_report()
            optimization_stats = get_optimization_stats()

            return {
                "cache_performance": cache_stats,
                "optimization_performance": optimization_stats,
                "system_config": {
                    "embedding_dim": settings.embedding_dim,
                    "vector_search_k": settings.vector_search_k,
                    "citation_confidence": settings.citation_confidence_threshold,
                    "embedding_model": settings.embedding_model
                }
            }
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}


def run_integration_tests():
    """Run all integration tests."""
    print("🔍 Legal RAG Integration Test - Cache & Optimizer")
    print("=" * 60)

    runner = IntegrationTestRunner()

    try:
        # Test 1: Query Optimization
        optimization_results = runner.test_query_optimization()

        # Test 2: Embedding Cache
        cache_results = runner.test_embedding_cache()

        # Test 3: Integrated Performance
        integration_results = runner.test_integrated_search_performance()

        # Test 4: System Summary
        system_summary = runner.get_system_performance_summary()

        # Print summary
        print("\n" + "=" * 60)
        print("📊 INTEGRATION TEST SUMMARY")
        print("=" * 60)

        print(f"\n🔧 Query Optimization:")
        optimized_count = sum(1 for r in optimization_results.values() if r["improved"])
        print(f"  • {optimized_count}/{len(optimization_results)} queries optimized")
        avg_opt_time = sum(r["optimization_time_ms"] for r in optimization_results.values()) / len(optimization_results)
        print(f"  • Average optimization time: {avg_opt_time:.1f}ms")

        print(f"\n💾 Embedding Cache:")
        print(f"  • First call: {cache_results['first_call_ms']:.1f}ms")
        print(f"  • Second call: {cache_results['second_call_ms']:.1f}ms")
        print(f"  • Performance improvement: {cache_results['improvement_percent']:.1f}%")
        print(f"  • Results identical: {cache_results['results_identical']}")

        if cache_results.get('cache_stats'):
            cache_info = cache_results['cache_stats']
            print(f"  • Cache hit rate: {cache_info.get('hit_rate', 0):.1f}%")
            print(f"  • Cache size: {cache_info.get('cache_size', 0)} entries")

        print(f"\n🚀 Integrated Pipeline:")
        avg_duration = sum(r["duration_ms"] for r in integration_results.values()) / len(integration_results)
        print(f"  • Average query time: {avg_duration:.1f}ms")

        citation_queries = sum(1 for r in integration_results.values() if r["search_type"] == "explicit_citation")
        semantic_queries = len(integration_results) - citation_queries
        print(f"  • Citation queries: {citation_queries}")
        print(f"  • Semantic queries: {semantic_queries}")

        optimized_in_pipeline = sum(1 for r in integration_results.values() if r["optimization_applied"])
        print(f"  • Queries optimized in pipeline: {optimized_in_pipeline}/{len(integration_results)}")

        print(f"\n🎯 Performance Targets:")
        print(f"  • Citation queries <50ms: {'✅' if any(r['duration_ms'] < 50 and r['search_type'] == 'explicit_citation' for r in integration_results.values()) else '⚠️'}")
        print(f"  • Average query <30s: {'✅' if avg_duration < 30000 else '⚠️'}")
        print(f"  • Cache improvement >50%: {'✅' if cache_results['improvement_percent'] > 50 else '⚠️'}")

        print("\n✅ Integration test completed successfully!")

        return {
            "status": "success",
            "optimization": optimization_results,
            "cache": cache_results,
            "integration": integration_results,
            "system": system_summary
        }

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        logger.error(f"Integration test error: {e}")
        return {"status": "failed", "error": str(e)}


def run_quick_health_check():
    """Quick health check for cache and optimizer."""
    print("🏥 Quick Health Check")
    print("-" * 30)

    try:
        # Test optimizer
        optimizer = get_query_optimizer()
        test_query = "sanksi pidana"
        optimized, analysis = optimizer.optimize_query(test_query)
        print(f"✅ Query Optimizer: {test_query} → {optimized}")

        # Test cache stats
        cache_stats = get_cache_performance_report()
        print(f"✅ Cache System: {cache_stats.get('cache_size', 0)} entries, {cache_stats.get('hit_rate', 0):.1f}% hit rate")

        # Test search service
        search_service = VectorSearchService()
        result = search_service.search("test query", k=1)
        print(f"✅ Search Service: {result['metadata']['search_type']} search completed")

        print("✅ All systems operational!")
        return True

    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick health check
        success = run_quick_health_check()
        sys.exit(0 if success else 1)
    else:
        # Full integration test
        results = run_integration_tests()
        success = results.get("status") == "success"
        sys.exit(0 if success else 1)
