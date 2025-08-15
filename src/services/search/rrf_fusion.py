# """
# Reciprocal Rank Fusion (RRF) Engine for Hybrid Search.

# Combines vector search and BM25 search results using the RRF algorithm
# to create a unified ranking that leverages both semantic and keyword-based relevance.
# """

# from __future__ import annotations

# import logging
# import time
# from typing import List, Dict, Optional, Tuple, Set
# from dataclasses import dataclass
# from collections import defaultdict

# from ...utils.logging import get_logger, log_timing
# from .vector_search import SearchResult, SearchFilters

# logger = get_logger(__name__)


# @dataclass
# class RRFConfig:
#     """Configuration for RRF fusion."""
#     k: int = 60  # RRF constant (typically 60)
#     vector_weight: float = 1.0  # Weight for vector search results
#     bm25_weight: float = 1.0   # Weight for BM25 search results
#     min_score_threshold: float = 0.001  # Minimum score to include result
#     max_results: int = 100  # Maximum results to process per source


# class RRFFusionEngine:
#     """
#     Reciprocal Rank Fusion engine for combining multiple search result sets.

#     Implements the RRF algorithm: score = Σ(weight / (k + rank))
#     where k is typically 60, and rank starts from 1.
#     """

#     def __init__(self, config: Optional[RRFConfig] = None):
#         """
#         Initialize RRF fusion engine.

#         Args:
#             config: RRF configuration, uses defaults if None
#         """
#         self.config = config or RRFConfig()
#         logger.info(f"RRFFusionEngine initialized with k={self.config.k}")

#     def fuse_results(
#         self,
#         vector_results: List[SearchResult],
#         bm25_results: List[SearchResult],
#         max_results: int = 10
#     ) -> List[SearchResult]:
#         """
#         Fuse vector and BM25 search results using RRF.

#         Args:
#             vector_results: Results from vector search
#             bm25_results: Results from BM25 search
#             max_results: Maximum number of results to return

#         Returns:
#             List of fused SearchResult objects sorted by RRF score
#         """
#         start_time = time.time()

#         # Validate inputs
#         if not vector_results and not bm25_results:
#             logger.warning("No results provided for RRF fusion")
#             return []

#         # Apply RRF algorithm
#         rrf_scores = self._calculate_rrf_scores(vector_results, bm25_results)

#         # Build unified result set
#         unified_results = self._build_unified_results(
#             vector_results, bm25_results, rrf_scores
#         )

#         # Sort by RRF score and limit results
#         final_results = sorted(
#             unified_results,
#             key=lambda x: x.score,
#             reverse=True
#         )[:max_results]

#         duration_ms = (time.time() - start_time) * 1000

#         self._log_fusion(vector_results, bm25_results, final_results, duration_ms)

#         return final_results

#     def _calculate_rrf_scores(
#         self,
#         vector_results: List[SearchResult],
#         bm25_results: List[SearchResult]
#     ) -> Dict[str, float]:
#         """
#         Calculate RRF scores for all unique documents.

#         Args:
#             vector_results: Vector search results
#             bm25_results: BM25 search results

#         Returns:
#             Dictionary mapping unit_id to RRF score
#         """
#         rrf_scores = defaultdict(float)

#         # Process vector results
#         for rank, result in enumerate(vector_results[:self.config.max_results], 1):
#             unit_id = result.unit_id or result.id
#             if unit_id:
#                 score = self.config.vector_weight / (self.config.k + rank)
#                 rrf_scores[unit_id] += score

#                 logger.debug(f"Vector rank {rank}: {unit_id} -> +{score:.6f}")

#         # Process BM25 results
#         for rank, result in enumerate(bm25_results[:self.config.max_results], 1):
#             unit_id = result.unit_id or result.id
#             if unit_id:
#                 score = self.config.bm25_weight / (self.config.k + rank)
#                 rrf_scores[unit_id] += score

#                 logger.debug(f"BM25 rank {rank}: {unit_id} -> +{score:.6f}")

#         # Filter by minimum score threshold
#         filtered_scores = {
#             unit_id: score for unit_id, score in rrf_scores.items()
#             if score >= self.config.min_score_threshold
#         }

#         logger.debug(f"RRF calculated scores for {len(filtered_scores)} unique documents")

#         return filtered_scores

#     def _build_unified_results(
#         self,
#         vector_results: List[SearchResult],
#         bm25_results: List[SearchResult],
#         rrf_scores: Dict[str, float]
#     ) -> List[SearchResult]:
#         """
#         Build unified result set with RRF scores.

#         Args:
#             vector_results: Vector search results
#             bm25_results: BM25 search results
#             rrf_scores: Calculated RRF scores

#         Returns:
#             List of SearchResult objects with updated scores
#         """
#         # Create lookup maps for fast access
#         vector_map = {(r.unit_id or r.id): r for r in vector_results}
#         bm25_map = {(r.unit_id or r.id): r for r in bm25_results}

#         unified_results = []
#         processed_ids = set()

#         # Process all documents that have RRF scores
#         for unit_id, rrf_score in rrf_scores.items():
#             if unit_id in processed_ids:
#                 continue

#             # Choose the best source for this document
#             result = self._select_best_source(unit_id, vector_map, bm25_map)

#             if result:
#                 # Create new result with RRF score
#                 unified_result = self._create_unified_result(result, rrf_score, vector_map, bm25_map)
#                 unified_results.append(unified_result)
#                 processed_ids.add(unit_id)

#         return unified_results

#     def _select_best_source(
#         self,
#         unit_id: str,
#         vector_map: Dict[str, SearchResult],
#         bm25_map: Dict[str, SearchResult]
#     ) -> Optional[SearchResult]:
#         """
#         Select the best source result for a document.

#         Preference: vector result if available, then BM25 result.
#         Vector results typically have better metadata and content quality.

#         Args:
#             unit_id: Document unit ID
#             vector_map: Vector results lookup
#             bm25_map: BM25 results lookup

#         Returns:
#             Best SearchResult or None
#         """
#         # Prefer vector result (better content and metadata)
#         if unit_id in vector_map:
#             return vector_map[unit_id]
#         elif unit_id in bm25_map:
#             return bm25_map[unit_id]
#         else:
#             return None

#     def _create_unified_result(
#         self,
#         base_result: SearchResult,
#         rrf_score: float,
#         vector_map: Dict[str, SearchResult],
#         bm25_map: Dict[str, SearchResult]
#     ) -> SearchResult:
#         """
#         Create unified result with RRF score and fusion metadata.

#         Args:
#             base_result: Base SearchResult to copy from
#             rrf_score: Calculated RRF score
#             vector_map: Vector results for reference
#             bm25_map: BM25 results for reference

#         Returns:
#             New SearchResult with RRF score and metadata
#         """
#         unit_id = base_result.unit_id or base_result.id

#         # Determine source information
#         sources = []
#         original_scores = {}

#         if unit_id in vector_map:
#             sources.append("vector")
#             original_scores["vector"] = vector_map[unit_id].score

#         if unit_id in bm25_map:
#             sources.append("bm25")
#             original_scores["bm25"] = bm25_map[unit_id].score

#         # Create enhanced metadata
#         fusion_metadata = {
#             **(base_result.metadata or {}),
#             "search_type": "rrf_hybrid",
#             "fusion_sources": sources,
#             "original_scores": original_scores,
#             "rrf_score": rrf_score,
#             "rrf_config": {
#                 "k": self.config.k,
#                 "vector_weight": self.config.vector_weight,
#                 "bm25_weight": self.config.bm25_weight
#             }
#         }

#         # Create new result with RRF score
#         return SearchResult(
#             id=base_result.id,
#             content=base_result.content,
#             citation_string=base_result.citation_string,
#             score=rrf_score,  # Use RRF score
#             unit_type=base_result.unit_type,
#             unit_id=base_result.unit_id,
#             doc_form=base_result.doc_form,
#             doc_year=base_result.doc_year,
#             doc_number=base_result.doc_number,
#             hierarchy_path=base_result.hierarchy_path,
#             metadata=fusion_metadata
#         )

#     def _log_fusion(
#         self,
#         vector_results: List[SearchResult],
#         bm25_results: List[SearchResult],
#         final_results: List[SearchResult],
#         duration_ms: float
#     ):
#         """Log fusion performance and statistics."""

#         # Calculate overlap statistics
#         vector_ids = {r.unit_id or r.id for r in vector_results}
#         bm25_ids = {r.unit_id or r.id for r in bm25_results}
#         final_ids = {r.unit_id or r.id for r in final_results}

#         overlap = len(vector_ids & bm25_ids)
#         vector_only = len(vector_ids - bm25_ids)
#         bm25_only = len(bm25_ids - vector_ids)

#         logger.info(
#             f"RRF fusion completed",
#             extra=log_timing(
#                 "rrf_fusion",
#                 duration_ms,
#                 vector_results=len(vector_results),
#                 bm25_results=len(bm25_results),
#                 final_results=len(final_results),
#                 overlap_count=overlap,
#                 vector_only=vector_only,
#                 bm25_only=bm25_only,
#                 fusion_ratio=len(final_results) / max(len(vector_results) + len(bm25_results), 1)
#             )
#         )

#     def analyze_fusion_quality(
#         self,
#         vector_results: List[SearchResult],
#         bm25_results: List[SearchResult],
#         fused_results: List[SearchResult]
#     ) -> Dict[str, float]:
#         """
#         Analyze the quality of fusion results.

#         Args:
#             vector_results: Original vector results
#             bm25_results: Original BM25 results
#             fused_results: Fused results

#         Returns:
#             Dictionary with quality metrics
#         """
#         if not fused_results:
#             return {"error": "no_fused_results"}

#         try:
#             # Create ID sets for analysis
#             vector_ids = {r.unit_id or r.id for r in vector_results}
#             bm25_ids = {r.unit_id or r.id for r in bm25_results}
#             fused_ids = {r.unit_id or r.id for r in fused_results}

#             # Calculate metrics
#             total_unique = len(vector_ids | bm25_ids)
#             overlap = len(vector_ids & bm25_ids)
#             preserved_in_fusion = len(fused_ids)

#             # Quality metrics
#             metrics = {
#                 "total_unique_documents": total_unique,
#                 "overlap_count": overlap,
#                 "overlap_ratio": overlap / max(total_unique, 1),
#                 "preservation_ratio": preserved_in_fusion / max(total_unique, 1),
#                 "vector_coverage": len(fused_ids & vector_ids) / max(len(vector_ids), 1),
#                 "bm25_coverage": len(fused_ids & bm25_ids) / max(len(bm25_ids), 1),
#                 "score_diversity": self._calculate_score_diversity(fused_results),
#                 "rank_correlation": self._calculate_rank_correlation(vector_results, fused_results)
#             }

#             return metrics

#         except Exception as e:
#             logger.error(f"Failed to analyze fusion quality: {e}")
#             return {"error": str(e)}

#     def _calculate_score_diversity(self, results: List[SearchResult]) -> float:
#         """Calculate score diversity (standard deviation of scores)."""
#         if len(results) < 2:
#             return 0.0

#         scores = [r.score for r in results]
#         mean_score = sum(scores) / len(scores)
#         variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

#         return variance ** 0.5

#     def _calculate_rank_correlation(
#         self,
#         original_results: List[SearchResult],
#         fused_results: List[SearchResult]
#     ) -> float:
#         """Calculate rank correlation between original and fused results."""
#         if not original_results or not fused_results:
#             return 0.0

#         # Create rank mappings
#         original_ranks = {
#             (r.unit_id or r.id): rank for rank, r in enumerate(original_results)
#         }
#         fused_ranks = {
#             (r.unit_id or r.id): rank for rank, r in enumerate(fused_results)
#         }

#         # Find common documents
#         common_ids = set(original_ranks.keys()) & set(fused_ranks.keys())

#         if len(common_ids) < 2:
#             return 0.0

#         # Calculate Spearman correlation (simplified)
#         rank_diffs = []
#         for doc_id in common_ids:
#             diff = original_ranks[doc_id] - fused_ranks[doc_id]
#             rank_diffs.append(diff ** 2)

#         n = len(common_ids)
#         spearman = 1 - (6 * sum(rank_diffs)) / (n * (n**2 - 1))

#         return max(-1.0, min(1.0, spearman))  # Clamp to [-1, 1]


# def create_rrf_fusion_engine(config: Optional[RRFConfig] = None) -> RRFFusionEngine:
#     """Factory function to create RRF fusion engine."""
#     return RRFFusionEngine(config)


# # Example usage and testing
# if __name__ == "__main__":
#     # Example usage
#     from .vector_search import SearchResult

#     # Mock results for testing
#     vector_results = [
#         SearchResult(id="doc1", content="Vector content 1", score=0.9, citation_string="Citation 1"),
#         SearchResult(id="doc2", content="Vector content 2", score=0.8, citation_string="Citation 2"),
#         SearchResult(id="doc3", content="Vector content 3", score=0.7, citation_string="Citation 3")
#     ]

#     bm25_results = [
#         SearchResult(id="doc2", content="BM25 content 2", score=0.95, citation_string="Citation 2"),  # Overlap
#         SearchResult(id="doc4", content="BM25 content 4", score=0.85, citation_string="Citation 4"),
#         SearchResult(id="doc5", content="BM25 content 5", score=0.75, citation_string="Citation 5")
#     ]

#     # Test fusion
#     fusion_engine = create_rrf_fusion_engine()
#     fused_results = fusion_engine.fuse_results(vector_results, bm25_results, max_results=5)

#     print(f"Fused {len(fused_results)} results:")
#     for i, result in enumerate(fused_results, 1):
#         print(f"{i}. {result.citation_string}")
#         print(f"   RRF Score: {result.score:.6f}")
#         print(f"   Sources: {result.metadata.get('fusion_sources', [])}")
#         print()

#     # Analyze quality
#     quality = fusion_engine.analyze_fusion_quality(vector_results, bm25_results, fused_results)
#     print("Fusion Quality Metrics:")
#     for metric, value in quality.items():
#         print(f"  {metric}: {value}")
