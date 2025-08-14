"""
Hybrid Search Service combining Vector Search, BM25, and RRF Fusion.

Provides unified search interface that leverages both semantic similarity (vector search)
and keyword matching (BM25) with Reciprocal Rank Fusion for optimal results.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any

from ...config.settings import settings
from ...utils.logging import get_logger, log_timing, log_error
from .vector_search import VectorSearchService, SearchResult, SearchFilters
from .bm25_search import BM25SearchService
from .rrf_fusion import RRFFusionEngine, RRFConfig

logger = get_logger(__name__)


class HybridSearchService:
    """
    Hybrid search service combining vector search and BM25 with RRF fusion.

    Provides best of both worlds: semantic understanding from vectors
    and precise keyword matching from BM25, unified via RRF algorithm.
    """

    def __init__(
        self,
        vector_service: Optional[VectorSearchService] = None,
        bm25_service: Optional[BM25SearchService] = None,
        rrf_config: Optional[RRFConfig] = None
    ):
        """
        Initialize hybrid search service.

        Args:
            vector_service: Vector search service instance
            bm25_service: BM25 search service instance
            rrf_config: RRF fusion configuration
        """
        self.vector_service = vector_service or VectorSearchService()
        self.bm25_service = bm25_service or BM25SearchService()
        self.rrf_engine = RRFFusionEngine(rrf_config)

        # Hybrid search configuration
        self.enable_hybrid = True
        self.vector_k_multiplier = 1.5  # Get more results for fusion
        self.bm25_k_multiplier = 1.5
        self.min_query_length_for_bm25 = 3
        self.comparative_query_patterns = [
            r'(?:apa\s+)?(?:bedanya|perbedaan)\s+(?:antara\s+)?(.+?)\s+(?:dengan|dan)\s+(.+?)(?:\?|$)',
            r'(?:beda|berbeda)\s+(.+?)\s+(?:dengan|dan)\s+(.+?)(?:\?|$)',
            r'bandingkan\s+(.+?)\s+(?:dengan|dan)\s+(.+?)(?:\?|$)',
        ]

        logger.info("HybridSearchService initialized with vector + BM25 + RRF")

    async def search_async(
        self,
        query: str,
        k: int = 10,
        filters: Optional[SearchFilters] = None,
        strategy: str = "auto"
    ) -> List[SearchResult]:
        """
        Async hybrid search with multiple strategies.

        Args:
            query: Search query
            k: Number of results to return
            filters: Optional search filters
            strategy: Search strategy ("auto", "hybrid", "vector_only", "bm25_only")

        Returns:
            List of SearchResult objects
        """
        start_time = time.time()

        try:
            # Determine strategy
            effective_strategy = self._determine_strategy(query, strategy)

            # Handle different query types
            if self._is_comparative_query(query):
                results = await self._handle_comparative_query_async(query, k, filters)
            elif effective_strategy == "hybrid":
                results = await self._hybrid_search_async(query, k, filters)
            elif effective_strategy == "vector_only":
                response = await self.vector_service.search_async(query, k, filters)
                results = response["results"] if isinstance(response, dict) and "results" in response else response
            elif effective_strategy == "bm25_only":
                response = await self.bm25_service.search_async(query, k, filters)
                results = response["results"] if isinstance(response, dict) and "results" in response else response
            else:
                # Fallback to hybrid
                results = await self._hybrid_search_async(query, k, filters)

            duration_ms = (time.time() - start_time) * 1000

            self._log_search(query, results, effective_strategy, duration_ms)

            return results

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Hybrid search failed for query '{query}': {e}",
                extra=log_error(e, context={
                    "query": query,
                    "k": k,
                    "strategy": strategy,
                    "duration_ms": duration_ms
                })
            )
            # Fallback to vector search only
            try:
                return await self.vector_service.search_async(query, k, filters)
            except Exception as fallback_e:
                logger.error(f"Fallback vector search also failed: {fallback_e}")
                return []

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[SearchFilters] = None,
        strategy: str = "auto"
    ) -> List[SearchResult]:
        """
        Sync wrapper for hybrid search.

        Args:
            query: Search query
            k: Number of results to return
            filters: Optional search filters
            strategy: Search strategy

        Returns:
            List of SearchResult objects
        """
        try:
            # Run async version in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.search_async(query, k, filters, strategy)
                )
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Sync hybrid search failed: {e}")
            # Fallback to sync vector search
            return self.vector_service.search(query, k, filters)

    async def _hybrid_search_async(
        self,
        query: str,
        k: int,
        filters: Optional[SearchFilters]
    ) -> List[SearchResult]:
        """
        Execute hybrid search combining vector and BM25 results.

        Args:
            query: Search query
            k: Number of results to return
            filters: Optional search filters

        Returns:
            Fused search results
        """
        # EXACT multi-level strategy limits per specification
        vector_k = 10  # Max 10 PASAL units for semantic context (PASAL.content)
        bm25_k = 15    # Max 15 units from all levels (PASAL.content + granular.bm25_body)

        # Execute searches concurrently
        search_tasks = []

        # Always do vector search
        vector_task = asyncio.create_task(
            self.vector_service.search_async(query, vector_k, filters)
        )
        search_tasks.append(("vector", vector_task))

        # Do BM25 search if query is long enough
        if len(query.strip()) >= self.min_query_length_for_bm25:
            bm25_task = asyncio.create_task(
                self.bm25_service.search_async(query, bm25_k, filters)
            )
            search_tasks.append(("bm25", bm25_task))

        # Wait for all searches to complete
        results = {}
        for search_type, task in search_tasks:
            try:
                response = await task
                # Extract results list from dict format (both services return {"results": [...], "metadata": {...}})
                if isinstance(response, dict) and "results" in response:
                    results[search_type] = response["results"]
                    logger.debug(f"{search_type} search returned {len(response['results'])} results")
                else:
                    # Fallback for unexpected format
                    results[search_type] = response if isinstance(response, list) else []
                    logger.debug(f"{search_type} search returned {len(results[search_type])} results (fallback)")
            except Exception as e:
                logger.warning(f"{search_type} search failed: {e}")
                results[search_type] = []

        # Fuse results using RRF
        vector_results = results.get("vector", [])
        bm25_results = results.get("bm25", [])

        if not vector_results and not bm25_results:
            logger.warning("Both vector and BM25 searches returned no results")
            return []

        # If only one type of results, return them directly
        if not bm25_results:
            logger.debug("Using vector results only (no BM25 results)")
            return vector_results[:k]
        elif not vector_results:
            logger.debug("Using BM25 results only (no vector results)")
            return bm25_results[:k]

        # Fuse results using RRF
        fused_results = self.rrf_engine.fuse_results(
            vector_results, bm25_results, max_results=k
        )

        logger.debug(f"RRF fusion produced {len(fused_results)} final results")
        return fused_results

    async def _handle_comparative_query_async(
        self,
        query: str,
        k: int,
        filters: Optional[SearchFilters]
    ) -> List[SearchResult]:
        """
        Handle comparative queries by decomposing and searching separately.

        Args:
            query: Comparative query
            k: Number of results to return
            filters: Optional search filters

        Returns:
            Combined results from decomposed queries
        """
        logger.debug(f"Processing comparative query: {query}")

        # Extract comparison terms
        concepts = self._extract_comparative_concepts(query)
        if not concepts:
            logger.warning("Could not extract concepts from comparative query")
            return await self._hybrid_search_async(query, k, filters)

        concept_a, concept_b = concepts
        logger.debug(f"Comparative concepts: '{concept_a}' vs '{concept_b}'")

        # Generate sub-queries for each concept
        sub_queries = [
            f"definisi {concept_a}",
            f"definisi {concept_b}",
            f"hukuman {concept_a}",
            f"hukuman {concept_b}",
            query  # Original query
        ]

        # Search for each sub-query
        all_results = []
        k_per_query = max(2, k // len(sub_queries))

        search_tasks = [
            self._hybrid_search_async(sub_query, k_per_query, filters)
            for sub_query in sub_queries
        ]

        sub_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        for i, result in enumerate(sub_results):
            if isinstance(result, Exception):
                logger.warning(f"Sub-query {i} failed: {result}")
                continue

            # Add metadata about which sub-query this came from
            for res in result:
                if hasattr(res, 'metadata') and res.metadata:
                    res.metadata['comparative_query'] = True
                    res.metadata['sub_query'] = sub_queries[i]
                    res.metadata['concept_a'] = concept_a
                    res.metadata['concept_b'] = concept_b

            all_results.extend(result)

        # Deduplicate by unit_id while preserving order
        seen_ids = set()
        deduplicated = []

        for result in all_results:
            unit_id = result.unit_id or result.id
            if unit_id not in seen_ids:
                seen_ids.add(unit_id)
                deduplicated.append(result)

        # Return top k results
        final_results = deduplicated[:k]
        logger.debug(f"Comparative query returned {len(final_results)} results")

        return final_results

    def _determine_strategy(self, query: str, requested_strategy: str) -> str:
        """
        Determine the most appropriate search strategy.

        Args:
            query: Search query
            requested_strategy: User-requested strategy

        Returns:
            Effective strategy to use
        """
        if requested_strategy != "auto":
            return requested_strategy

        # Auto-determine based on query characteristics
        query_len = len(query.strip())

        # Very short queries - prefer vector search
        if query_len < 10:
            return "vector_only"

        # Long, detailed queries - prefer hybrid
        if query_len > 50:
            return "hybrid"

        # Medium queries - use hybrid if BM25 is viable
        if query_len >= self.min_query_length_for_bm25:
            return "hybrid"
        else:
            return "vector_only"

    def _is_comparative_query(self, query: str) -> bool:
        """Check if query is asking for a comparison."""
        import re

        for pattern in self.comparative_query_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def _extract_comparative_concepts(self, query: str) -> Optional[tuple[str, str]]:
        """Extract the two concepts being compared."""
        import re

        for pattern in self.comparative_query_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                concept_a = match.group(1).strip()
                concept_b = match.group(2).strip()
                return (concept_a, concept_b)

        return None

    def _log_search(
        self,
        query: str,
        results: List[SearchResult],
        strategy: str,
        duration_ms: float
    ):
        """Log search performance and results."""

        # Analyze result sources
        source_counts = {}
        for result in results:
            search_type = result.metadata.get('search_type', 'unknown') if result.metadata else 'unknown'
            source_counts[search_type] = source_counts.get(search_type, 0) + 1

        logger.info(
            f"Hybrid search completed",
            extra=log_timing(
                "hybrid_search",
                duration_ms,
                query_length=len(query),
                results_count=len(results),
                strategy=strategy,
                source_breakdown=source_counts,
                avg_score=sum(r.score for r in results) / len(results) if results else 0
            )
        )

    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get statistics about hybrid search services.

        Returns:
            Dictionary with service statistics
        """
        try:
            from ...db.session import get_db_session

            stats = {
                "hybrid_search": {
                    "enabled": self.enable_hybrid,
                    "vector_k_multiplier": self.vector_k_multiplier,
                    "bm25_k_multiplier": self.bm25_k_multiplier,
                    "min_query_length_for_bm25": self.min_query_length_for_bm25
                },
                "rrf_config": {
                    "k": self.rrf_engine.config.k,
                    "vector_weight": self.rrf_engine.config.vector_weight,
                    "bm25_weight": self.rrf_engine.config.bm25_weight
                }
            }

            # Get corpus statistics
            with get_db_session() as db:
                if hasattr(self.bm25_service, 'get_search_stats'):
                    stats["bm25_corpus"] = self.bm25_service.get_search_stats(db)

            return stats

        except Exception as e:
            logger.error(f"Failed to get hybrid search stats: {e}")
            return {"error": str(e)}

    def benchmark_search_methods(
        self,
        test_queries: List[str],
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark different search methods on test queries.

        Args:
            test_queries: List of queries to test
            k: Number of results per query

        Returns:
            Benchmark results
        """
        if not test_queries:
            return {"error": "no_test_queries"}

        benchmark_results = {
            "query_count": len(test_queries),
            "k": k,
            "methods": {}
        }

        methods = ["vector_only", "bm25_only", "hybrid"]

        for method in methods:
            method_stats = {
                "total_time_ms": 0,
                "total_results": 0,
                "avg_time_per_query": 0,
                "avg_results_per_query": 0,
                "errors": 0
            }

            start_time = time.time()

            for query in test_queries:
                try:
                    query_start = time.time()
                    results = self.search(query, k, strategy=method)
                    query_time = (time.time() - query_start) * 1000

                    method_stats["total_time_ms"] += query_time
                    method_stats["total_results"] += len(results)

                except Exception as e:
                    logger.warning(f"Benchmark query failed for {method}: {e}")
                    method_stats["errors"] += 1

            # Calculate averages
            if len(test_queries) > 0:
                method_stats["avg_time_per_query"] = method_stats["total_time_ms"] / len(test_queries)
                method_stats["avg_results_per_query"] = method_stats["total_results"] / len(test_queries)

            benchmark_results["methods"][method] = method_stats

        benchmark_results["total_benchmark_time"] = (time.time() - start_time) * 1000

        return benchmark_results


def create_hybrid_search_service(
    vector_service: Optional[VectorSearchService] = None,
    bm25_service: Optional[BM25SearchService] = None,
    rrf_config: Optional[RRFConfig] = None
) -> HybridSearchService:
    """Factory function to create hybrid search service."""
    return HybridSearchService(vector_service, bm25_service, rrf_config)


# Example usage
if __name__ == "__main__":
    # Example usage
    service = create_hybrid_search_service()

    # Test different query types
    test_queries = [
        "ekonomi kreatif",  # Simple keyword
        "apa bedanya pembunuhan berencana dan tidak berencana?",  # Comparative
        "UU 24 tahun 2019 pasal 1",  # Citation
        "definisi pelaku ekonomi kreatif menurut undang-undang"  # Contextual
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = service.search(query, k=3)

        for i, result in enumerate(results, 1):
            search_type = result.metadata.get('search_type', 'unknown') if result.metadata else 'unknown'
            print(f"{i}. [{search_type}] {result.citation_string}")
            print(f"   Score: {result.score:.4f}")
            print(f"   Content: {result.content[:100]}...")

    # Benchmark
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)

    benchmark = service.benchmark_search_methods(test_queries, k=5)
    for method, stats in benchmark["methods"].items():
        print(f"\n{method.upper()}:")
        print(f"  Avg time per query: {stats['avg_time_per_query']:.1f}ms")
        print(f"  Avg results per query: {stats['avg_results_per_query']:.1f}")
        print(f"  Errors: {stats['errors']}")
