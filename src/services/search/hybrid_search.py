"""
Hybrid Search Service combining Vector Search, BM25, and RRF Fusion.

Provides unified search interface that leverages both semantic similarity (vector search)
and keyword matching (BM25) with Reciprocal Rank Fusion for optimal results.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from ...config.settings import settings
from ...utils.logging import get_logger, log_timing, log_error
from ...db.session import get_db_session
from ...db import queries as db_queries
from .vector_search import VectorSearchService, SearchResult, SearchFilters
# BM25 is optional; allow running without it
try:
    from .bm25_search import BM25SearchService  # type: ignore
except Exception:  # module or class may be removed
    BM25SearchService = None  # type: ignore
from .explicit_pg import ExplicitPGService
from .explicit.regex import parse as parse_explicit_citation
if TYPE_CHECKING:
    # Type-only imports to avoid importing RRF module at bootstrap
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
        rrf_config: Optional['RRFConfig'] = None
    ):
        """
        Initialize hybrid search service.

        Args:
            vector_service: Vector search service instance
            bm25_service: BM25 search service instance
            rrf_config: RRF fusion configuration
        """
        self.vector_service = vector_service or VectorSearchService()
        # Instantiate BM25 only if available or explicitly provided
        if bm25_service is not None:
            self.bm25_service = bm25_service
        else:
            if BM25SearchService is not None:
                self.bm25_service = BM25SearchService()  # type: ignore
            else:
                self.bm25_service = None
                logger.warning("BM25SearchService unavailable; running without BM25 branch")
        self.explicit_service = ExplicitPGService()
        # Initialize RRF engine only when SQL fusion is disabled
        if not settings.USE_SQL_FUSION:
            # Lazy import to avoid bootstrap import cost when not needed
            try:
                from .rrf_fusion import RRFFusionEngine as _RRFFusionEngine  # type: ignore
                self.rrf_engine = _RRFFusionEngine(rrf_config)
            except Exception:
                self.rrf_engine = None
                logger.warning("RRF fusion module unavailable; falling back to simple merge")
        else:
            self.rrf_engine = None

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
        strategy: str = "auto",
        use_reranking: bool = False,
    ) -> Dict[str, Any]:
        """
        Async hybrid search with multiple strategies.

        Args:
            query: Search query
            k: Number of results to return
            filters: Optional search filters
            strategy: Search strategy
            use_reranking: Whether to use reranking

        Returns:
            List of SearchResult objects
        """
        start = time.time()
        logger.info("HybridSearchService.search_async.start", extra={
            "event": "hybrid_search_start",
            "query": query,
            "k": k,
            "strategy": strategy,
            "flags": {
                "NEW_PG_RETRIEVAL": settings.NEW_PG_RETRIEVAL,
                "USE_SQL_FUSION": settings.USE_SQL_FUSION,
                "USE_RERANKER": settings.USE_RERANKER,
            },
        })

        try:
            # Determine strategy and route explicit PG early if matched
            effective_strategy = strategy

            if settings.NEW_PG_RETRIEVAL and strategy in ("auto", "hybrid") and self._is_explicit_reference(query):
                effective_strategy = "explicit_pg"
                exp = await self.explicit_service.resolve_by_citation(query, k, filters)
                duration_ms = (time.time() - start) * 1000
                logger.info("HybridSearchService.search_async.explicit_pg", extra={
                    "event": "explicit_pg_route",
                    "duration_ms": duration_ms,
                    "total_results": len(exp.get("results", [])),
                })
                return exp

            # Handle different query types and strategies
            if self._is_comparative_query(query):
                results = await self._handle_comparative_query_async(query, k, filters)
            elif effective_strategy == "hybrid":
                results = await self._hybrid_search_async(query, k, filters)
            elif effective_strategy == "vector_only":
                results = await self.vector_service.search_async(query, k, filters)
            elif effective_strategy == "bm25_only":
                results = await self.bm25_service.search_async(query, k, filters)
            else:  # auto (default to hybrid)
                results = await self._hybrid_search_async(query, k, filters)

            duration = (time.time() - start) * 1000.0
            logger.info("HybridSearchService.search_async.finish", extra={
                "event": "hybrid_search_finish",
                "query": query,
                "strategy": strategy,
                "duration_ms": duration,
                "total_results": len(results if isinstance(results, list) else results.get("results", [])),
            })

            # Normalize results to list of dicts for unified schema
            unified_results = [
                (res if isinstance(res, dict) else (res.to_dict() if hasattr(res, "to_dict") else res))
                for res in results
            ]

            return {
                "results": unified_results[:k],
                "metadata": {
                    "query": query,
                    "strategy": strategy,
                    "total_results": len(unified_results),
                    "limit": k,
                    "duration_ms": duration,
                    "feature_flags": {
                        "NEW_PG_RETRIEVAL": settings.NEW_PG_RETRIEVAL,
                        "USE_SQL_FUSION": settings.USE_SQL_FUSION,
                        "USE_RERANKER": settings.USE_RERANKER,
                    },
                },
            }

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            logger.error(
                f"Hybrid search failed for query '{query}': {e}",
                extra=log_error(e, context={
                    "query": query,
                    "k": k,
                    "strategy": strategy,
                    "duration_ms": duration_ms
                })
            )
            # Fallback to vector search only (already unified dict)
            try:
                fallback = await self.vector_service.search_async(query, k, filters)
                # Ensure feature_flags present in metadata
                if isinstance(fallback, dict):
                    meta = fallback.setdefault("metadata", {})
                    meta.setdefault("search_type", "vector")
                    meta.setdefault("strategy", "vector_only_fallback")
                    meta["duration_ms"] = meta.get("duration_ms", round(duration_ms, 2))
                    meta.setdefault("feature_flags", {
                        "NEW_PG_RETRIEVAL": settings.NEW_PG_RETRIEVAL,
                        "USE_SQL_FUSION": settings.USE_SQL_FUSION,
                        "USE_RERANKER": settings.USE_RERANKER,
                    })
                return fallback
            except Exception as fallback_e:
                logger.error(f"Fallback vector search also failed: {fallback_e}")
                return {
                    "results": [],
                    "metadata": {
                        "query": query,
                        "strategy": strategy,
                        "search_type": "error",
                        "total_results": 0,
                        "duration_ms": round(duration_ms, 2),
                        "error": str(fallback_e),
                        "feature_flags": {
                            "NEW_PG_RETRIEVAL": settings.NEW_PG_RETRIEVAL,
                            "USE_SQL_FUSION": settings.USE_SQL_FUSION,
                            "USE_RERANKER": settings.USE_RERANKER,
                        },
                    },
                }

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

        # If SQL fusion is enabled, build inputs and delegate to DB fusion
        force_merge = False
        if settings.USE_SQL_FUSION:
            try:
                sql_fusion_start = time.time()
                # For plainto_tsquery, pass the raw (trimmed) user query to let PostgreSQL parse terms
                def _as_plain_query(text_q: str) -> str:
                    return (text_q or "").strip()

                tsq = _as_plain_query(query)

                # Try to embed query using vector service (private or public method)
                embed_fn = getattr(self.vector_service, "_embed_query", None) or getattr(self.vector_service, "embed_query", None)
                qvec = embed_fn(query) if callable(embed_fn) else None

                # Diagnostics: summarize tsquery and vector
                try:
                    tsq_token_count = len([t for t in (tsq or '').split() if t])
                except Exception:
                    tsq_token_count = None
                try:
                    vec_dim = len(qvec) if qvec is not None else None
                    vec_norm = None
                    if qvec is not None and hasattr(qvec, '__iter__'):
                        # Compute L2 norm defensively
                        s = 0.0
                        for _v in qvec:
                            try:
                                s += float(_v) * float(_v)
                            except Exception:
                                continue
                        vec_norm = s ** 0.5
                except Exception:
                    vec_dim, vec_norm = None, None

                with get_db_session() as db:
                    fused = db_queries.search_fusion(
                        db,
                        lquery=None,
                        tsquery=tsq if tsq else None,
                        query_vector=qvec if qvec is not None else None,
                        limit=k,
                        fts_weight=settings.fts_weight,
                        vector_weight=settings.vector_weight,
                        min_ts_rank=settings.min_ts_rank,
                    )
                # Summarize per-branch counts if available
                match_counts = {"explicit": 0, "fts": 0, "vector": 0, "unknown": 0}
                try:
                    for r in fused:
                        mt = None
                        if isinstance(r, dict):
                            mt = r.get('match_type')
                        else:
                            mt = getattr(r, 'match_type', None)
                        if mt in match_counts:
                            match_counts[mt] += 1
                        else:
                            match_counts['unknown'] += 1
                except Exception:
                    pass

                logger.info("HybridSearchService.hybrid.sql_fusion", extra={
                    "event": "sql_fusion",
                    "duration_ms": (time.time() - sql_fusion_start) * 1000.0,
                    "total_results": len(fused),
                    "fusion_weights": {
                        "fts_weight": settings.fts_weight,
                        "vector_weight": settings.vector_weight,
                        "min_ts_rank": settings.min_ts_rank,
                    },
                    "tsquery_stats": {
                        "tsquery": tsq,
                        "token_count": tsq_token_count,
                    },
                    "vector_stats": {
                        "dim": vec_dim,
                        "l2_norm": vec_norm,
                    },
                    "branch_counts": match_counts,
                })
                # Fallback: if fusion returns no rows, use in-app merge path
                if fused and len(fused) > 0:
                    return fused
                else:
                    force_merge = True
            except Exception as e:
                logger.warning(f"SQL fusion unavailable, falling back to in-app merge: {e}")
                # Ensure we bypass RRF and perform simple merge
                force_merge = True

        # Execute searches concurrently (non-SQL fusion path)
        search_tasks = []

        # Always do vector search
        vector_task = asyncio.create_task(
            self.vector_service.search_async(query, vector_k, filters)
        )
        search_tasks.append(("vector", vector_task))

        # Do BM25 search if query is long enough
        if self.bm25_service is not None and len(query.strip()) >= self.min_query_length_for_bm25:
            bm25_task = asyncio.create_task(
                self.bm25_service.search_async(query, bm25_k, filters)  # type: ignore
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

        logger.info("HybridSearchService.hybrid.merge_prepare", extra={
            "event": "hybrid_merge_prepare",
            "vector_count": len(results.get("vector", [])),
            "bm25_count": len(results.get("bm25", [])),
        })

        # Fuse results using in-app RRF/merge
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

        # Fuse results depending on feature flags
        if force_merge or not self.rrf_engine:
            # Simple merge and dedupe preserving vector priority
            merged: list[Any] = []
            seen = set()
            for res in vector_results + bm25_results:
                uid = None
                if isinstance(res, dict):
                    uid = res.get('unit_id') or res.get('id')
                else:
                    uid = getattr(res, 'unit_id', None) or getattr(res, 'id', None)
                if uid in seen:
                    continue
                seen.add(uid)
                merged.append(res)

            logger.debug(f"Merged {len(vector_results)}+{len(bm25_results)} -> {len(merged)}")
            return merged[:k]
        else:
            # RRF engine expects attribute-style access. Wrap dicts if needed.
            class _Wrap:
                __slots__ = ("unit_id", "id", "score", "citation_string", "content", "metadata", "unit_type", "title")
                def __init__(self, d: Dict[str, Any]):
                    self.unit_id = d.get("unit_id")
                    self.id = d.get("id")
                    self.score = d.get("score")
                    self.citation_string = d.get("citation_string")
                    self.content = d.get("content")
                    self.metadata = d.get("metadata") or {}
                    self.unit_type = d.get("unit_type")
                    self.title = d.get("title")
                def __getattr__(self, name: str):
                    # Safely return None for any unexpected attribute access
                    return None

            def _ensure_obj_list(items: List[Any]) -> List[Any]:
                out = []
                for it in items:
                    if isinstance(it, dict):
                        out.append(_Wrap(it))
                    else:
                        out.append(it)
                return out

            vector_objs = _ensure_obj_list(vector_results)
            bm25_objs = _ensure_obj_list(bm25_results)

            fused_results = self.rrf_engine.fuse_results(
                vector_objs, bm25_objs, max_results=k
            )

            logger.debug(f"RRF fusion produced {len(fused_results)} final results")
            # Return original dicts if they were dicts originally; to be safe, map back to dicts
            mapped: List[Any] = []
            for r in fused_results:
                if isinstance(r, dict):
                    mapped.append(r)
                else:
                    # Convert back to dict if wrapper or object has to_dict
                    if hasattr(r, "to_dict"):
                        mapped.append(r.to_dict())
                    else:
                        mapped.append({
                            "unit_id": getattr(r, "unit_id", None) or getattr(r, "id", None),
                            "id": getattr(r, "id", None),
                            "score": getattr(r, "score", None),
                            "citation_string": getattr(r, "citation_string", None),
                            "content": getattr(r, "content", None),
                            "metadata": getattr(r, "metadata", None) or {},
                            "unit_type": getattr(r, "unit_type", None),
                            "title": getattr(r, "title", None),
                        })
            return mapped

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

    def _is_explicit_reference(self, query: str) -> bool:
        """Detect explicit legal citation patterns using the regex parser.

        Returns True when parser can extract at least a pasal number.
        """
        try:
            p = parse_explicit_citation(query)
            return bool(p and p.pasal)
        except Exception:
            return False

    def _is_comparative_query(self, query: str) -> bool:
        """Check if query is asking for a comparison."""
        import re

        for pattern in self.comparative_query_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def _log_search(
        self,
        query: str,
        results: List[Any],
        strategy: str,
        duration_ms: float,
    ) -> None:
        """Safely log search execution without assuming result object shape.

        Accepts either a list of dicts or a list of SearchResult-like objects.
        """
        try:
            total = len(results) if results is not None else 0

            def _uid(r: Any):
                if isinstance(r, dict):
                    return r.get("unit_id") or r.get("id")
                return getattr(r, "unit_id", None) or getattr(r, "id", None)

            sample_ids = []
            for r in (results or [])[:5]:
                sample_ids.append(_uid(r))

            logger.info(
                "Hybrid search executed",
                extra={
                    "event": "hybrid_search",
                    "query": query,
                    "strategy": strategy,
                    "duration_ms": round(duration_ms, 2),
                    "total_results": total,
                    "sample_unit_ids": sample_ids,
                    "feature_flags": {
                        "NEW_PG_RETRIEVAL": settings.NEW_PG_RETRIEVAL,
                        "USE_SQL_FUSION": settings.USE_SQL_FUSION,
                        "USE_RERANKER": settings.USE_RERANKER,
                    },
                },
            )
        except Exception as e:
            # Never let logging break the search flow
            logger.debug(f"_log_search failed: {e}")

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
