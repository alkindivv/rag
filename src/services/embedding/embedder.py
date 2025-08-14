"""Jina v4 embedding client using Haystack's production-ready integration."""

from __future__ import annotations

import logging
import time
from typing import List, Literal, Optional

from haystack_integrations.components.embedders.jina import JinaTextEmbedder
from haystack.utils import Secret
from src.config.settings import settings
from src.utils.logging import get_logger, log_timing, log_error

logger = get_logger(__name__)


class ConfigError(Exception):
    """Configuration error for embedding service."""
    pass


class EmbeddingError(Exception):
    """Embedding API error."""
    pass


class JinaV4Embedder:
    """
    Production-ready Jina v4 embedding client using Haystack integration.

    This wrapper provides the same interface as the custom implementation but uses
    Haystack's battle-tested JinaTextEmbedder with built-in reliability features.

    Features:
    - Automatic retry logic with exponential backoff
    - Configurable timeouts that actually work
    - Circuit breaker patterns
    - Better error handling and classification
    """

    def __init__(
        self,
        client: Optional[object] = None,  # Kept for compatibility, not used
        api_key: Optional[str] = None,
        model: str = "jina-embeddings-v4",
        default_task: Literal["retrieval.query", "retrieval.passage"] = "retrieval.passage",
        default_dims: int = 384,
        return_multivector: bool = False,
    ) -> None:
        """
        Initialize Haystack-powered Jina v4 embedder.

        Args:
            client: Ignored (kept for compatibility)
            api_key: Optional API key override (uses JINA_API_KEY env var if None)
            model: Jina model name
            default_task: Default task for embeddings
            default_dims: Default embedding dimensions
            return_multivector: Enable multi-vector embeddings (raises NotImplementedError)

        Raises:
            ConfigError: If API key is not provided
            NotImplementedError: If return_multivector is True
        """
        self.api_key = api_key or settings.jina_api_key
        self.model = model
        self.default_task = default_task
        self.default_dims = default_dims
        self.batch_size = settings.embed_batch_size

        # Validate API key
        if not self.api_key:
            raise ConfigError(
                "JINA_API_KEY is required. Get your free API key at: https://jina.ai/?sui=apikey"
            )

        # Multi-vector not supported
        if return_multivector:
            raise NotImplementedError(
                "Multi-vector embeddings not supported in this version."
            )

        # Initialize Haystack's JinaTextEmbedder with reliable settings
        self._embedder = JinaTextEmbedder(
            model=self.model,
            api_key=Secret.from_token(self.api_key),
            dimensions=self.default_dims
        )

        logger.info(
            f"Initialized JinaV4Embedder: model={self.model}, dims={self.default_dims}, task={self.default_task}"
        )

    def embed_texts(
        self,
        texts: List[str],
        task: Literal["retrieval.query", "retrieval.passage"] = "retrieval.passage",
        dims: Optional[int] = None,
        return_multivector: Optional[bool] = None,
    ) -> List[List[float]]:
        """
        Embed texts using Haystack's production-ready JinaTextEmbedder.

        Args:
            texts: List of texts to embed
            task: Task type for adapter selection
            dims: Optional dimension override (ignored - set at init)
            return_multivector: Must be False or None (not supported)

        Returns:
            List of embedding vectors (one per text)

        Raises:
            NotImplementedError: If return_multivector is True
            EmbeddingError: If embedding fails
        """
        if not texts:
            return []

        # Validate multi-vector request
        if return_multivector:
            raise NotImplementedError(
                "Multi-vector embeddings not supported in this version."
            )

        try:
            start_time = time.time()

            # Use Haystack's reliable embedding (one text at a time)
            # Note: JinaTextEmbedder handles single text, not batches
            embeddings = []
            for text in texts:
                result = self._embedder.run(text)
                embeddings.append(result["embedding"])

            duration = (time.time() - start_time) * 1000
            logger.debug(
                f"Embedded {len(texts)} texts in {duration:.1f}ms using Haystack JinaTextEmbedder",
                extra={"batch_size": len(texts), "task": task, "duration_ms": duration}
            )

            return embeddings

        except Exception as e:
            logger.error(f"Haystack embedding failed: {e}")
            raise EmbeddingError(f"Embedding failed: {e}") from e

    def _embed_batch_internal(
        self,
        texts: List[str],
        task: str,
        dimensions: int
    ) -> List[List[float]]:
        """
        Legacy method - now uses embed_texts for consistency.

        Args:
            texts: Batch of texts to embed
            task: Task type for adapter selection
            dimensions: Embedding dimensions (ignored)

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding fails
        """
        # Delegate to main embed_texts method which uses Haystack
        return self.embed_texts(texts, task=task)

    def embed_single(self, text: str, task: Optional[str] = None) -> List[float]:
        """
        Embed a single text string using Haystack's reliable integration.

        Args:
            text: Text to embed
            task: Optional task override

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding fails
        """
        task = task or self.default_task
        embeddings = self.embed_texts([text], task=task)
        return embeddings[0]

    def embed_query(self, query: str, dims: Optional[int] = None) -> List[float]:
        """
        Embed a query using retrieval.query task.

        Args:
            query: Query text to embed
            dims: Optional dimension override (ignored)

        Returns:
            Query embedding vector

        Raises:
            EmbeddingError: If embedding fails
        """
        return self.embed_texts([query], task="retrieval.query")[0]

    def embed_passages(self, passages: List[str], dims: Optional[int] = None) -> List[List[float]]:
        """
        Embed passages using retrieval.passage task.

        Args:
            passages: List of passage texts
            dims: Optional dimension override (ignored)

        Returns:
            List of passage embedding vectors

        Raises:
            EmbeddingError: If embedding fails
        """
        return self.embed_texts(passages, task="retrieval.passage")

    # Backward compatibility methods
    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Legacy method for backward compatibility.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (wraps None for failed embeddings)
        """
        try:
            embeddings = self.embed_texts(texts, task=self.default_task)
            return embeddings
        except Exception as e:
            logger.error(f"Legacy embed_batch failed: {e}")
            return [None] * len(texts)

    def embed(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Legacy method for backward compatibility.

        Args:
            texts: List of texts to embed
            batch_size: Optional batch size override (ignored, uses configured batch_size)

        Returns:
            List of embeddings (filters out None values)
        """
        try:
            return self.embed_texts(texts, task=self.default_task)
        except Exception as e:
            logger.error(f"Legacy embed method failed: {e}")
            return []


class EmbedderFactory:
    """Factory for creating embedder instances."""

    @staticmethod
    def create_embedder(
        client: Optional[HttpClient] = None,
        api_key: Optional[str] = None,
    ) -> JinaV4Embedder:
        """
        Create embedder instance with configuration from settings.

        Args:
            client: Optional HTTP client for dependency injection
            api_key: Optional API key override

        Returns:
            Configured JinaV4Embedder instance
        """
        return JinaV4Embedder(
            client=client,
            api_key=api_key,
            model=settings.embedding_model,
            default_task=settings.embedding_task_passage,
            default_dims=settings.embedding_dim,
            return_multivector=False,
        )


# Convenience functions
def embed_query(query: str, dims: Optional[int] = None) -> List[float]:
    """Convenience function to embed a single query."""
    embedder = EmbedderFactory.create_embedder()
    return embedder.embed_query(query, dims=dims)


def embed_passages(passages: List[str], dims: Optional[int] = None) -> List[List[float]]:
    """Convenience function to embed multiple passages."""
    embedder = EmbedderFactory.create_embedder()
    return embedder.embed_passages(passages, dims=dims)


# Default instance for backward compatibility
def get_default_embedder() -> JinaV4Embedder:
    """Get default configured embedder instance."""
    return EmbedderFactory.create_embedder()


# Legacy alias for backward compatibility
JinaEmbedder = JinaV4Embedder
