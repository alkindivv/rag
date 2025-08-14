#!/usr/bin/env python3
"""
Setup script for Week 2 Hybrid Search implementation.

This script sets up and validates the hybrid search system combining:
- PostgreSQL FTS for BM25 search
- RRF (Reciprocal Rank Fusion) engine
- Enhanced comparative query handling

Usage:
    python scripts/setup_hybrid_search.py [--validate-only] [--benchmark]
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.session import get_db_session
from src.services.search.vector_search import VectorSearchService, create_vector_search_service
from src.services.search.bm25_search import BM25SearchService, create_bm25_search_service
from src.services.search.rrf_fusion import RRFFusionEngine, RRFConfig, create_rrf_fusion_engine
from src.services.search.hybrid_search import HybridSearchService, create_hybrid_search_service
from src.utils.logging import get_logger
from sqlalchemy import text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)


class HybridSearchSetup:
    """Setup and validation for hybrid search system."""

    def __init__(self):
        """Initialize setup manager."""
        self.vector_service = None
        self.bm25_service = None
        self.rrf_engine = None
        self.hybrid_service = None

    def run_full_setup(self) -> bool:
        """
        Run complete setup process.

        Returns:
            True if setup successful, False otherwise
        """
        logger.info("Starting Week 2 Hybrid Search Setup")
        logger.info("=" * 50)

        try:
            # Step 1: Run database migration
            if not self._run_migration():
                return False

            # Step 2: Initialize services
            if not self._initialize_services():
                return False

            # Step 3: Validate FTS setup
            if not self._validate_fts_setup():
                return False

            # Step 4: Test individual components
            if not self._test_components():
                return False

            # Step 5: Test hybrid search
            if not self._test_hybrid_search():
                return False

            logger.info("✅ Week 2 Hybrid Search setup completed successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False

    def _run_migration(self) -> bool:
        """Run the FTS migration."""
        logger.info("Step 1: Running FTS migration...")

        try:
            # Import alembic here to avoid dependency issues
            from alembic.config import Config
            from alembic import command

            # Run the migration
            alembic_cfg = Config("alembic.ini")
            command.upgrade(alembic_cfg, "head")

            logger.info("✅ Migration completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            logger.info("You may need to run the migration manually:")
            logger.info("  alembic upgrade head")
            return False

    def _initialize_services(self) -> bool:
        """Initialize all search services."""
        logger.info("Step 2: Initializing search services...")

        try:
            # Initialize services
            self.vector_service = create_vector_search_service()
            self.bm25_service = create_bm25_search_service()
            self.rrf_engine = create_rrf_fusion_engine()
            self.hybrid_service = create_hybrid_search_service(
                self.vector_service,
                self.bm25_service
            )

            logger.info("✅ All services initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            return False

    def _validate_fts_setup(self) -> bool:
        """Validate FTS database setup."""
        logger.info("Step 3: Validating FTS database setup...")

        try:
            with get_db_session() as db:
                # Check if tsvector column exists
                check_column = """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'legal_units'
                AND column_name = 'bm25_tsvector';
                """
                result = db.execute(text(check_column))
                if not result.fetchone():
                    logger.error("❌ bm25_tsvector column not found")
                    return False

                # Check if GIN index exists
                check_index = """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'legal_units'
                AND indexname = 'idx_legal_units_bm25_tsvector_gin';
                """
                result = db.execute(text(check_index))
                if not result.fetchone():
                    logger.error("❌ FTS GIN index not found")
                    return False

                # Check if trigger exists
                check_trigger = """
                SELECT trigger_name
                FROM information_schema.triggers
                WHERE event_object_table = 'legal_units'
                AND trigger_name = 'trigger_update_bm25_tsvector';
                """
                result = db.execute(text(check_trigger))
                if not result.fetchone():
                    logger.error("❌ FTS trigger not found")
                    return False

                # Check corpus statistics
                stats_query = """
                SELECT
                    COUNT(*) as total_units,
                    COUNT(CASE WHEN bm25_tsvector IS NOT NULL THEN 1 END) as indexed_units
                FROM legal_units lu
                JOIN legal_documents ld ON ld.id = lu.document_id
                WHERE ld.doc_status = 'BERLAKU';
                """
                result = db.execute(text(stats_query))
                row = result.fetchone()

                if row.total_units == 0:
                    logger.warning("⚠️ No legal units found in database")
                    return False

                coverage = row.indexed_units / row.total_units if row.total_units > 0 else 0
                logger.info(f"📊 FTS Coverage: {row.indexed_units}/{row.total_units} ({coverage:.1%})")

                if coverage < 0.8:
                    logger.warning(f"⚠️ Low FTS coverage: {coverage:.1%}")

            logger.info("✅ FTS setup validation passed")
            return True

        except Exception as e:
            logger.error(f"❌ FTS validation failed: {e}")
            return False

    def _test_components(self) -> bool:
        """Test individual search components."""
        logger.info("Step 4: Testing individual components...")

        # Test queries
        test_queries = [
            "ekonomi kreatif",
            "pelaku ekonomi kreatif",
            "kreativitas manusia",
            "warisan budaya"
        ]

        try:
            # Test Vector Search
            logger.info("Testing Vector Search...")
            for query in test_queries[:2]:  # Test 2 queries
                try:
                    response = self.vector_service.search(query, k=3)
                    results = response["results"] if isinstance(response, dict) and "results" in response else response
                    logger.info(f"  '{query}': {len(results)} results")
                    if results and len(results) > 0:
                        logger.info(f"    Top result: {results[0].citation_string}")
                    else:
                        logger.warning(f"    No results returned for query: {query}")
                except Exception as e:
                    logger.error(f"    Vector search failed for query '{query}': {e}")

            # Test BM25 Search
            logger.info("Testing BM25 Search...")
            for query in test_queries[:2]:  # Test 2 queries
                try:
                    response = self.bm25_service.search(query, k=3)
                    results = response["results"] if isinstance(response, dict) and "results" in response else response
                    logger.info(f"  '{query}': {len(results)} results")
                    if results and len(results) > 0:
                        logger.info(f"    Top result: {results[0].citation_string}")
                    else:
                        logger.warning(f"    No results returned for BM25 query: {query}")
                except Exception as e:
                    logger.error(f"    BM25 search failed for query '{query}': {e}")

            # Test RRF Fusion
            logger.info("Testing RRF Fusion...")
            try:
                vector_response = self.vector_service.search(test_queries[0], k=5)
                bm25_response = self.bm25_service.search(test_queries[0], k=5)

                vector_results = vector_response["results"] if isinstance(vector_response, dict) and "results" in vector_response else vector_response
                bm25_results = bm25_response["results"] if isinstance(bm25_response, dict) and "results" in bm25_response else bm25_response

                logger.info(f"  Got {len(vector_results)} vector results, {len(bm25_results)} BM25 results")

                if vector_results and bm25_results:
                    fused_results = self.rrf_engine.fuse_results(vector_results, bm25_results, max_results=3)
                    logger.info(f"  RRF fusion: {len(vector_results)} + {len(bm25_results)} → {len(fused_results)} results")

                    if fused_results and len(fused_results) > 0:
                        logger.info(f"    Top fused result: {fused_results[0].citation_string}")

                        # Analyze fusion quality
                        quality = self.rrf_engine.analyze_fusion_quality(vector_results, bm25_results, fused_results)
                        logger.info(f"    Fusion quality - Overlap: {quality.get('overlap_ratio', 0):.2f}")
                    else:
                        logger.warning("    RRF fusion returned no results")
                else:
                    if not vector_results:
                        logger.warning("    No vector results for RRF testing")
                    if not bm25_results:
                        logger.warning("    No BM25 results for RRF testing")
            except Exception as e:
                logger.error(f"    RRF fusion testing failed: {e}")

            logger.info("✅ Component testing completed")
            return True

        except Exception as e:
            logger.error(f"❌ Component testing failed: {e}")
            return False

    def _test_hybrid_search(self) -> bool:
        """Test hybrid search functionality."""
        logger.info("Step 5: Testing hybrid search...")

        test_cases = [
            {
                "query": "definisi ekonomi kreatif",
                "strategy": "hybrid",
                "expected_min_results": 1
            },
            {
                "query": "apa bedanya ekonomi kreatif dengan pelaku ekonomi kreatif?",
                "strategy": "auto",
                "expected_min_results": 1,
                "is_comparative": True
            },
            {
                "query": "UU 24 tahun 2019",
                "strategy": "auto",
                "expected_min_results": 1
            }
        ]

        try:
            for i, test_case in enumerate(test_cases, 1):
                query = test_case["query"]
                strategy = test_case["strategy"]
                expected_min = test_case["expected_min_results"]

                logger.info(f"Test {i}: '{query}' (strategy: {strategy})")

                start_time = time.time()
                results = self.hybrid_service.search(query, k=5, strategy=strategy)
                duration = (time.time() - start_time) * 1000

                logger.info(f"  Results: {len(results)} in {duration:.1f}ms")

                if len(results) < expected_min:
                    logger.warning(f"  ⚠️ Expected at least {expected_min} results, got {len(results)}")

                # Show top result
                if results:
                    top_result = results[0]
                    search_type = top_result.metadata.get('search_type', 'unknown') if top_result.metadata else 'unknown'
                    logger.info(f"  Top result [{search_type}]: {top_result.citation_string}")
                    logger.info(f"  Score: {top_result.score:.4f}")

                    # Check comparative query handling
                    if test_case.get("is_comparative"):
                        if top_result.metadata and top_result.metadata.get("comparative_query"):
                            logger.info("  ✅ Comparative query detected and handled")
                        else:
                            logger.warning("  ⚠️ Comparative query not properly detected")

            logger.info("✅ Hybrid search testing completed")
            return True

        except Exception as e:
            logger.error(f"❌ Hybrid search testing failed: {e}")
            return False

    def run_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmark."""
        logger.info("Running hybrid search benchmark...")

        benchmark_queries = [
            "ekonomi kreatif",
            "pelaku ekonomi kreatif indonesia",
            "definisi kreativitas manusia",
            "warisan budaya teknologi",
            "apa bedanya ekonomi kreatif dengan industri kreatif?",
            "perbedaan UU dan PP dalam sistem hukum",
            "badan usaha berbadan hukum",
            "kegiatan ekonomi kreatif di indonesia"
        ]

        try:
            benchmark_result = self.hybrid_service.benchmark_search_methods(benchmark_queries, k=5)

            logger.info("📊 BENCHMARK RESULTS")
            logger.info("=" * 40)

            for method, stats in benchmark_result["methods"].items():
                logger.info(f"\n{method.upper()}:")
                logger.info(f"  Avg time per query: {stats['avg_time_per_query']:.1f}ms")
                logger.info(f"  Avg results per query: {stats['avg_results_per_query']:.1f}")
                logger.info(f"  Error rate: {stats['errors']}/{len(benchmark_queries)}")

                if stats['avg_time_per_query'] > 0:
                    qps = 1000 / stats['avg_time_per_query']
                    logger.info(f"  Throughput: {qps:.1f} queries/second")

            logger.info(f"\nTotal benchmark time: {benchmark_result['total_benchmark_time']:.1f}ms")

            return benchmark_result

        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            return {}

    def validate_only(self) -> bool:
        """Run validation without setup."""
        logger.info("Running validation checks only...")

        try:
            if not self._initialize_services():
                return False

            if not self._validate_fts_setup():
                return False

            if not self._test_components():
                return False

            if not self._test_hybrid_search():
                return False

            logger.info("✅ All validation checks passed")
            return True

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            if not self.hybrid_service:
                self._initialize_services()

            stats = self.hybrid_service.get_service_stats()

            with get_db_session() as db:
                # Get database stats
                db_stats_query = """
                SELECT
                    COUNT(*) as total_documents,
                    COUNT(DISTINCT lu.unit_id) as total_units,
                    COUNT(CASE WHEN lu.bm25_tsvector IS NOT NULL THEN 1 END) as indexed_units,
                    COUNT(dv.id) as vector_count
                FROM legal_documents ld
                LEFT JOIN legal_units lu ON lu.document_id = ld.id
                LEFT JOIN document_vectors dv ON dv.document_id = ld.id
                WHERE ld.doc_status = 'BERLAKU';
                """
                result = db.execute(text(db_stats_query))
                row = result.fetchone()

                stats["database"] = {
                    "total_documents": row.total_documents,
                    "total_units": row.total_units,
                    "indexed_units": row.indexed_units,
                    "vector_count": row.vector_count,
                    "fts_coverage": row.indexed_units / row.total_units if row.total_units > 0 else 0
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup Week 2 Hybrid Search implementation"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation only (skip setup)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status"
    )

    args = parser.parse_args()

    setup = HybridSearchSetup()

    try:
        if args.status:
            logger.info("Getting system status...")
            status = setup.get_system_status()

            logger.info("📊 SYSTEM STATUS")
            logger.info("=" * 30)
            for category, info in status.items():
                if isinstance(info, dict):
                    logger.info(f"\n{category.upper()}:")
                    for key, value in info.items():
                        if isinstance(value, float):
                            logger.info(f"  {key}: {value:.2f}")
                        else:
                            logger.info(f"  {key}: {value}")
                else:
                    logger.info(f"{category}: {info}")

            return True

        elif args.validate_only:
            success = setup.validate_only()
        else:
            success = setup.run_full_setup()

        if success and args.benchmark:
            logger.info("\n" + "="*50)
            setup.run_benchmark()

        if success:
            logger.info("\n🎉 Week 2 Hybrid Search is ready to use!")
            logger.info("\nYou can now use the hybrid search in your application:")
            logger.info("  from src.services.search.hybrid_search import create_hybrid_search_service")
            logger.info("  service = create_hybrid_search_service()")
            logger.info("  results = service.search('your query here')")

            return True
        else:
            logger.error("\n❌ Setup incomplete. Please check the errors above.")
            return False

    except KeyboardInterrupt:
        logger.info("\n⏹️ Setup interrupted by user")
        return False
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
