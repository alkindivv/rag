"""
Jina reranker service for re-ordering search results.

Provides semantic reranking of search results using Jina's reranking API
to improve relevance ordering for Legal RAG system.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

# Removed JinaEmbedder import - not needed for reranker
from ...config.settings import settings
from ..retriever.hybrid_retriever import SearchResult
from ...utils.http import HttpClient
from ...utils.logging import get_logger, log_timing, log_error

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Abstract base class for reranking services."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank search results based on query relevance.

        Args:
            query: Original search query
            results: List of search results to rerank
            top_k: Maximum number of results to return

        Returns:
            Reranked list of search results
        """
        pass


class NoOpReranker(BaseReranker):
    """No-operation reranker that returns results unchanged."""

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Return results unchanged, optionally limited by top_k."""
        if top_k is not None:
            return results[:top_k]
        return results


class JinaReranker(BaseReranker):
    """Jina reranking service for semantic result ordering."""

    def __init__(self, client: Optional[HttpClient] = None):
        """
        Initialize Jina reranker.

        Args:
            client: Optional HTTP client for dependency injection
        """
        self.client = client or HttpClient()
        self.base_url = settings.jina_rerank_base
        self.model_name = settings.jina_rerank_model
        self.max_batch_size = 100  # Jina API limit

        if not settings.jina_api_key:
            raise ValueError("JINA_API_KEY is required for Jina reranker")

        self.headers = {
            "Authorization": f"Bearer {settings.jina_api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"Initialized JinaReranker with model: {self.model_name}")

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank search results using Jina reranking API.

        Args:
            query: Original search query
            results: List of search results to rerank
            top_k: Maximum number of results to return

        Returns:
            Reranked list of search results with updated scores
        """
        if not results:
            return []

        if not query.strip():
            logger.warning("Empty query provided to reranker")
            return results[:top_k] if top_k else results

        start_time = time.time()

        try:
            # Limit input size
            input_results = results[:self.max_batch_size]

            # Prepare documents for reranking
            documents = []
            for result in input_results:
                # Combine text content for better ranking
                doc_text = result.text
                if result.citation_string:
                    doc_text = f"{result.citation_string}: {doc_text}"
                documents.append(doc_text)

            if not documents:
                logger.warning("No documents to rerank")
                return []

            # Call Jina rerank API
            reranked_scores = self._call_rerank_api(query, documents)

            if not reranked_scores:
                logger.warning("Reranking API returned no scores, returning original order")
                return input_results[:top_k] if top_k else input_results

            # Update scores and sort
            for i, (result, new_score) in enumerate(zip(input_results, reranked_scores)):
                result.score = new_score
                # Add reranking metadata
                if result.metadata is None:
                    result.metadata = {}
                result.metadata["rerank_score"] = new_score
                result.metadata["original_rank"] = i
                result.source_type = f"{result.source_type}_reranked"

            # Sort by new scores
            reranked_results = sorted(input_results, key=lambda x: x.score, reverse=True)

            # Apply top_k limit
            final_results = reranked_results[:top_k] if top_k else reranked_results

            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Reranking completed: {len(results)} -> {len(final_results)} results",
                extra=log_timing(
                    "rerank",
                    duration_ms,
                    input_count=len(results),
                    output_count=len(final_results)
                )
            )

            return final_results

        except Exception as e:
            logger.error(f"Reranking failed: {e}", extra=log_error(e))
            # Return original results on error
            return results[:top_k] if top_k else results

    def _call_rerank_api(self, query: str, documents: List[str]) -> List[float]:
        """
        Call Jina reranking API.

        Args:
            query: Search query
            documents: List of document texts to rank

        Returns:
            List of relevance scores
        """
        try:
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "top_n": len(documents),  # Return scores for all documents
                "return_documents": True  # Jina v2 requires this to be true
            }

            logger.debug(f"Calling Jina rerank API with {len(documents)} documents")

            response = self.client.post_json(
                url=self.base_url,
                data=payload,
                headers=self.headers
            )

            if "results" not in response:
                logger.error("Invalid response from Jina rerank API")
                return []

            # Extract relevance scores from response
            scores = []
            for result in response["results"]:
                if "relevance_score" not in result:
                    logger.warning(f"Missing relevance_score in result: {result}")
                    scores.append(0.0)
                else:
                    scores.append(float(result["relevance_score"]))

            logger.debug(f"Received {len(scores)} rerank scores from v2-base-multilingual")
            return scores

        except Exception as e:
            logger.error(f"Jina rerank API call failed: {e}")
            return []


class RerankerFactory:
    """Factory for creating reranker instances based on configuration."""

    @staticmethod
    def create_reranker(
        provider: Optional[str] = None,
        client: Optional[HttpClient] = None
    ) -> BaseReranker:
        """
        Create reranker instance based on provider configuration.

        Args:
            provider: Reranker provider ("jina" or "none")
            client: Optional HTTP client for dependency injection

        Returns:
            Reranker instance
        """
        provider = provider or settings.rerank_provider

        if provider == "jina":
            try:
                return JinaReranker(client=client)
            except Exception as e:
                logger.warning(f"Failed to create Jina reranker: {e}, falling back to NoOp")
                return NoOpReranker()
        elif provider == "none" or not provider:
            return NoOpReranker()
        else:
            logger.warning(f"Unknown reranker provider: {provider}, using NoOp")
            return NoOpReranker()


# Convenience function for quick reranking
def rerank_results(
    query: str,
    results: List[SearchResult],
    top_k: Optional[int] = None,
    provider: Optional[str] = None
) -> List[SearchResult]:
    """
    Convenience function to rerank search results.

    Args:
        query: Original search query
        results: List of search results to rerank
        top_k: Maximum number of results to return
        provider: Optional reranker provider override

    Returns:
        Reranked search results
    """
    reranker = RerankerFactory.create_reranker(provider=provider)
    return reranker.rerank(query, results, top_k)


# Example usage and testing
if __name__ == "__main__":
    # Example test
    from ...retriever.hybrid_retriever import SearchResult

    # Create sample results
    sample_results = [
        SearchResult(
            id="1",
            text="Pasal tentang pertambangan mineral",
            citation_string="UU No. 4 Tahun 2009, Pasal 1",
            score=0.8,
            source_type="vector",
            unit_type="pasal",
            unit_id="UU-2009-4/pasal-1"
        ),
        SearchResult(
            id="2",
            text="Ayat tentang batubara dan energi",
            citation_string="UU No. 4 Tahun 2009, Pasal 2, Ayat 1",
            score=0.6,
            source_type="fts",
            unit_type="ayat",
            unit_id="UU-2009-4/pasal-2/ayat-1"
        )
    ]

    # Test reranking
    reranked = rerank_results(
        query="pertambangan mineral dan batubara",
        results=sample_results,
        top_k=5
    )

    print(f"Reranked {len(reranked)} results:")
    for i, result in enumerate(reranked):
        print(f"{i+1}. {result.citation_string} (score: {result.score:.3f})")
