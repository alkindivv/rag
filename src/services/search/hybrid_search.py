"""
Hybrid search service orchestrating retrieval and reranking for Legal RAG system.

This service combines FTS, vector search, and reranking to provide comprehensive
search capabilities for legal documents.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ...config.settings import settings
from ...db.session import get_db_session
from ..retriever.hybrid_retriever import HybridRetriever, SearchFilters, SearchResult
from ...services.embedding.embedder import JinaEmbedder
from ...utils.logging import get_logger, log_timing
from .reranker import RerankerFactory


logger = get_logger(__name__)


class HybridSearchService:
    """
    Main search service orchestrating retrieval and reranking.

    Provides unified interface for all search operations in the Legal RAG system,
    combining multiple search strategies and optional reranking.
    """

    def __init__(
        self,
        embedder: Optional[JinaEmbedder] = None,
        reranker_provider: Optional[str] = None
    ):
        """
        Initialize hybrid search service.

        Args:
            embedder: Optional embedder instance
            reranker_provider: Optional reranker provider override
        """
        self.embedder = embedder or JinaEmbedder()
        self.retriever = HybridRetriever(embedder=self.embedder)
        self.reranker = RerankerFactory.create_reranker(provider=reranker_provider)

        logger.info("Initialized HybridSearchService")

    def search(
        self,
        query: str,
        limit: int = 20,
        filters: Optional[SearchFilters] = None,
        use_reranking: bool = True,
        fts_weight: float = 0.4,
        vector_weight: float = 0.6,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive search with optional reranking.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional search filters
            use_reranking: Whether to apply reranking
            fts_weight: Weight for FTS results in hybrid scoring
            vector_weight: Weight for vector results in hybrid scoring
            session_id: Optional session ID for logging

        Returns:
            Search response with results and metadata
        """
        start_time = time.time()

        try:
            logger.info(
                f"Starting search",
                extra={
                    "query": query,
                    "limit": limit,
                    "use_reranking": use_reranking,
                    "session_id": session_id
                }
            )

            # Validate input
            if not query.strip():
                return {
                    "results": [],
                    "total": 0,
                    "query": query,
                    "strategy": "empty_query",
                    "reranked": False,
                    "duration_ms": 0
                }

            # Perform retrieval
            retrieval_start = time.time()
            results = self.retriever.search(
                query=query,
                filters=filters,
                limit=limit * 2 if use_reranking else limit,  # Get more for reranking
                fts_weight=fts_weight,
                vector_weight=vector_weight
            )
            retrieval_duration = (time.time() - retrieval_start) * 1000

            logger.debug(f"Retrieved {len(results)} initial results")

            # Apply reranking if enabled and provider is available
            reranked = False
            if use_reranking and settings.rerank_provider != "none":
                rerank_start = time.time()
                results = self.reranker.rerank(query, results, top_k=limit)
                rerank_duration = (time.time() - rerank_start) * 1000
                reranked = True

                logger.debug(f"Reranked to {len(results)} final results")
                logger.info(
                    "Reranking completed",
                    extra=log_timing("rerank", rerank_duration)
                )
            else:
                # Just apply limit without reranking
                results = results[:limit]

            # Log search to database
            self._log_search(query, len(results), session_id)

            # Prepare response
            total_duration = (time.time() - start_time) * 1000

            response = {
                "results": [self._format_result(result) for result in results],
                "total": len(results),
                "query": query,
                "strategy": self._determine_strategy(query),
                "reranked": reranked,
                "duration_ms": total_duration,
                "metadata": {
                    "retrieval_duration_ms": retrieval_duration,
                    "rerank_duration_ms": rerank_duration if reranked else 0,
                    "filters_applied": asdict(filters) if filters else None,
                    "weights": {
                        "fts": fts_weight,
                        "vector": vector_weight
                    }
                }
            }

            logger.info(
                f"Search completed: {len(results)} results",
                extra=log_timing("search_total", total_duration, session_id=session_id)
            )

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Search failed: {e}",
                extra={
                    "query": query,
                    "duration_ms": duration_ms,
                    "session_id": session_id
                }
            )

            return {
                "results": [],
                "total": 0,
                "query": query,
                "strategy": "error",
                "reranked": False,
                "duration_ms": duration_ms,
                "error": str(e)
            }

    def get_related_content(
        self,
        pasal_id: str,
        include_children: bool = True
    ) -> Dict[str, Any]:
        """
        Get related content for a specific pasal.

        Args:
            pasal_id: Pasal unit ID
            include_children: Whether to include child units

        Returns:
            Related content response
        """
        start_time = time.time()

        try:
            logger.info(f"Getting related content for pasal: {pasal_id}")

            results = self.retriever.get_related_units(
                pasal_id=pasal_id,
                include_children=include_children
            )

            duration_ms = (time.time() - start_time) * 1000

            response = {
                "pasal_id": pasal_id,
                "results": [self._format_result(result) for result in results],
                "total": len(results),
                "include_children": include_children,
                "duration_ms": duration_ms
            }

            logger.info(
                f"Related content retrieved: {len(results)} units",
                extra=log_timing("get_related_content", duration_ms)
            )

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to get related content for {pasal_id}: {e}")

            return {
                "pasal_id": pasal_id,
                "results": [],
                "total": 0,
                "include_children": include_children,
                "duration_ms": duration_ms,
                "error": str(e)
            }

    def _determine_strategy(self, query: str) -> str:
        """Determine which search strategy was used."""
        if self.retriever.router.is_explicit_query(query):
            return "explicit"
        else:
            return "thematic"

    def _format_result(self, result: SearchResult) -> Dict[str, Any]:
        """Format search result for API response."""
        return {
            "id": result.id,
            "text": result.text,
            "citation": result.citation_string,
            "score": result.score,
            "source_type": result.source_type,
            "unit_type": result.unit_type,
            "unit_id": result.unit_id,
            "document": {
                "form": result.doc_form,
                "year": result.doc_year,
                "number": result.doc_number
            },
            "metadata": result.metadata or {}
        }

    def _log_search(
        self,
        query: str,
        results_count: int,
        session_id: Optional[str] = None
    ) -> None:
        """Log search operation to database."""
        try:
            from ...db.models import VectorSearchLog

            with get_db_session() as db:
                log_entry = VectorSearchLog(
                    query_text=query,
                    limit_requested=20,  # Default limit
                    results_found=results_count,
                    user_session=session_id or "anonymous"
                )
                db.add(log_entry)

        except Exception as e:
            logger.warning(f"Failed to log search: {e}")


class AdvancedSearchService(HybridSearchService):
    """
    Advanced search service with additional features.

    Extends HybridSearchService with more sophisticated search capabilities.
    """

    def multi_query_search(
        self,
        queries: List[str],
        limit_per_query: int = 10,
        final_limit: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform search with multiple queries and combine results.

        Args:
            queries: List of search queries
            limit_per_query: Limit per individual query
            final_limit: Final result limit after combination
            **kwargs: Additional search parameters

        Returns:
            Combined search response
        """
        start_time = time.time()

        try:
            logger.info(f"Multi-query search with {len(queries)} queries")

            all_results = []

            for i, query in enumerate(queries):
                logger.debug(f"Processing query {i+1}/{len(queries)}: {query}")

                search_response = self.search(
                    query=query,
                    limit=limit_per_query,
                    use_reranking=False,  # Rerank once at the end
                    **kwargs
                )

                results = [
                    SearchResult(**{
                        k: v for k, v in result.items()
                        if k in SearchResult.__annotations__
                    })
                    for result in search_response["results"]
                ]

                all_results.extend(results)

            # Deduplicate and rerank combined results
            deduplicated = self.retriever._deduplicate_and_rank(all_results)

            # Apply final reranking with the first query as primary
            if kwargs.get("use_reranking", True) and queries:
                final_results = self.reranker.rerank(
                    queries[0], deduplicated, top_k=final_limit
                )
            else:
                final_results = deduplicated[:final_limit]

            duration_ms = (time.time() - start_time) * 1000

            response = {
                "results": [self._format_result(result) for result in final_results],
                "total": len(final_results),
                "queries": queries,
                "strategy": "multi_query",
                "reranked": kwargs.get("use_reranking", True),
                "duration_ms": duration_ms,
                "metadata": {
                    "queries_count": len(queries),
                    "total_retrieved": len(all_results),
                    "deduplicated": len(deduplicated)
                }
            }

            logger.info(
                f"Multi-query search completed: {len(final_results)} results",
                extra=log_timing("multi_query_search", duration_ms)
            )

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Multi-query search failed: {e}")

            return {
                "results": [],
                "total": 0,
                "queries": queries,
                "strategy": "multi_query_error",
                "reranked": False,
                "duration_ms": duration_ms,
                "error": str(e)
            }

    def contextual_search(
        self,
        query: str,
        context_pasal_ids: List[str],
        limit: int = 20,
        context_boost: float = 1.2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform search with contextual boosting based on related pasal.

        Args:
            query: Search query
            context_pasal_ids: List of pasal IDs to boost in results
            limit: Maximum number of results
            context_boost: Boost factor for contextual results
            **kwargs: Additional search parameters

        Returns:
            Search response with context-boosted results
        """
        start_time = time.time()

        try:
            logger.info(f"Contextual search with {len(context_pasal_ids)} context pasal")

            # Perform regular search
            search_response = self.search(query=query, limit=limit * 2, **kwargs)

            results = []
            for result_data in search_response["results"]:
                result = SearchResult(
                    id=result_data["id"],
                    text=result_data["text"],
                    citation_string=result_data["citation"],
                    score=result_data["score"],
                    source_type=result_data["source_type"],
                    unit_type=result_data["unit_type"],
                    unit_id=result_data["unit_id"],
                    doc_form=result_data["document"]["form"],
                    doc_year=result_data["document"]["year"],
                    doc_number=result_data["document"]["number"],
                    metadata=result_data["metadata"]
                )

                # Apply context boosting
                for context_id in context_pasal_ids:
                    if context_id in result.unit_id or result.unit_id.startswith(context_id):
                        result.score *= context_boost
                        if result.metadata is None:
                            result.metadata = {}
                        result.metadata["context_boosted"] = True
                        break

                results.append(result)

            # Re-sort by updated scores and apply limit
            results.sort(key=lambda x: x.score, reverse=True)
            final_results = results[:limit]

            duration_ms = (time.time() - start_time) * 1000

            response = {
                "results": [self._format_result(result) for result in final_results],
                "total": len(final_results),
                "query": query,
                "strategy": "contextual",
                "reranked": search_response.get("reranked", False),
                "duration_ms": duration_ms,
                "metadata": {
                    **search_response.get("metadata", {}),
                    "context_pasal_ids": context_pasal_ids,
                    "context_boost_factor": context_boost
                }
            }

            logger.info(
                f"Contextual search completed: {len(final_results)} results",
                extra=log_timing("contextual_search", duration_ms)
            )

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Contextual search failed: {e}")

            return {
                "results": [],
                "total": 0,
                "query": query,
                "strategy": "contextual_error",
                "reranked": False,
                "duration_ms": duration_ms,
                "error": str(e)
            }

    def get_document_outline(
        self,
        doc_id: str,
        include_content: bool = False
    ) -> Dict[str, Any]:
        """
        Get structured outline of a legal document.

        Args:
            doc_id: Document ID (e.g., "UU-2025-2")
            include_content: Whether to include unit content

        Returns:
            Document outline response
        """
        start_time = time.time()

        try:
            logger.info(f"Getting document outline for: {doc_id}")

            with get_db_session() as db:
                # Get document info
                from ...db.models import LegalDocument

                doc = db.query(LegalDocument).filter_by(doc_id=doc_id).first()
                if not doc:
                    return {
                        "doc_id": doc_id,
                        "outline": [],
                        "error": "Document not found"
                    }

                # Get units ordered by hierarchy
                units_query = text("""
                    SELECT
                        unit_id,
                        unit_type,
                        number_label,
                        ordinal_int,
                        ordinal_suffix,
                        label_display,
                        title,
                        citation_string,
                        parent_pasal_id,
                        hierarchy_path,
                        CASE WHEN :include_content THEN local_content ELSE NULL END as content
                    FROM legal_units
                    WHERE document_id = :doc_id
                    ORDER BY seq_sort_key, ordinal_int, ordinal_suffix
                """)

                results = db.execute(units_query, {
                    "doc_id": doc.id,
                    "include_content": include_content
                }).fetchall()

                # Build hierarchical outline
                outline = self._build_outline_tree(results)

                duration_ms = (time.time() - start_time) * 1000

                response = {
                    "doc_id": doc_id,
                    "document": {
                        "title": doc.doc_title,
                        "form": doc.doc_form.value,
                        "number": doc.doc_number,
                        "year": doc.doc_year,
                        "status": doc.doc_status.value
                    },
                    "outline": outline,
                    "total_units": len(results),
                    "duration_ms": duration_ms
                }

                logger.info(
                    f"Document outline retrieved: {len(results)} units",
                    extra=log_timing("get_document_outline", duration_ms)
                )

                return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to get document outline for {doc_id}: {e}")

            return {
                "doc_id": doc_id,
                "outline": [],
                "duration_ms": duration_ms,
                "error": str(e)
            }

    def _build_outline_tree(self, units: List[Any]) -> List[Dict[str, Any]]:
        """Build hierarchical outline tree from flat unit list."""
        # This is a simplified implementation
        # In practice, you might want to build a proper tree structure
        outline = []

        for unit in units:
            outline_item = {
                "unit_id": unit.unit_id,
                "type": unit.unit_type,
                "label": unit.label_display,
                "title": unit.title,
                "citation": unit.citation_string,
                "hierarchy_path": unit.hierarchy_path
            }

            if hasattr(unit, 'content') and unit.content:
                outline_item["content"] = unit.content

            outline.append(outline_item)

        return outline

    def _format_result(self, result: SearchResult) -> Dict[str, Any]:
        """Format search result for API response."""
        return {
            "id": result.id,
            "text": result.text,
            "citation": result.citation_string,
            "score": result.score,
            "source_type": result.source_type,
            "unit_type": result.unit_type,
            "unit_id": result.unit_id,
            "document": {
                "form": result.doc_form,
                "year": result.doc_year,
                "number": result.doc_number
            },
            "metadata": result.metadata or {}
        }


# Factory function for easy instantiation
def create_search_service(
    embedder: Optional[JinaEmbedder] = None,
    reranker_provider: Optional[str] = None
) -> HybridSearchService:
    """
    Create search service with default configuration.

    Args:
        embedder: Optional embedder instance
        reranker_provider: Optional reranker provider

    Returns:
        Configured search service
    """
    return HybridSearchService(
        embedder=embedder,
        reranker_provider=reranker_provider
    )


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    async def test_search():
        """Test search functionality."""
        # Initialize service
        service = create_search_service()

        # Test queries
        test_queries = [
            "pasal 1 ayat 2",  # Explicit query
            "pertambangan mineral dan batubara",  # Thematic query
            "UU 4/2009",  # Document reference
            "hilirisasi industri pertambangan"  # Semantic query
        ]

        for query in test_queries:
            print(f"\nTesting query: {query}")
            response = service.search(query, limit=5)

            print(f"Strategy: {response['strategy']}")
            print(f"Results: {response['total']}")
            print(f"Duration: {response['duration_ms']:.2f}ms")

            for i, result in enumerate(response["results"][:3]):
                print(f"  {i+1}. {result['citation']} (score: {result['score']:.3f})")

        # Test document outline
        print(f"\nTesting document outline:")
        outline_response = service.get_document_outline("UU-2025-2")
        print(f"Document: {outline_response.get('document', {}).get('title', 'N/A')}")
        print(f"Units: {outline_response['total_units']}")

    # Run test if executed directly
    if __name__ == "__main__":
        asyncio.run(test_search())
