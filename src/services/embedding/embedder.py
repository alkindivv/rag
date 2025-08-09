"""Jina v4 embedding client with strict API compliance and configurable dimensions."""

from __future__ import annotations

import logging
import re
import time
from typing import List, Literal, Optional

from src.config.settings import settings
from src.utils.http import HttpClient, AuthError, NetworkError, ServerError, HttpError
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
    Jina v4 embedding client with exact API compliance.

    Supports configurable dimensions, task-specific adapters, and proper error handling.
    Get your Jina AI API key for free: https://jina.ai/?sui=apikey
    """

    def __init__(
        self,
        client: Optional[HttpClient] = None,
        api_key: Optional[str] = None,
        model: str = "jina-embeddings-v4",
        default_task: Literal["retrieval.query", "retrieval.passage"] = "retrieval.passage",
        default_dims: int = 1024,
        return_multivector: bool = False,
    ) -> None:
        """
        Initialize Jina v4 embedder.

        Args:
            client: Optional HTTP client for dependency injection
            api_key: Optional API key override (uses JINA_API_KEY env var if None)
            model: Jina model name
            default_task: Default task for embeddings
            default_dims: Default embedding dimensions
            return_multivector: Enable multi-vector embeddings (raises NotImplementedError)

        Raises:
            ConfigError: If API key is not provided
            NotImplementedError: If return_multivector is True
        """
        self.client = client or HttpClient()
        self.api_key = api_key or settings.jina_api_key
        self.model = model
        self.default_task = default_task
        self.default_dims = default_dims
        self.batch_size = settings.embed_batch_size
        self.base_url = settings.jina_embed_base

        # Validate API key
        if not self.api_key:
            raise ConfigError(
                "JINA_API_KEY is required. Get your free API key at: https://jina.ai/?sui=apikey"
            )

        # Multi-vector not supported in v1
        if return_multivector:
            raise NotImplementedError(
                "Multi-vector embeddings not supported in v1. Will be added in future release."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

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
        Embed texts using Jina v4 API with task-specific adapters.

        Args:
            texts: List of texts to embed
            task: Task type for adapter selection
            dims: Optional dimension truncation (uses default_dims if None)
            return_multivector: Must be False or None (not supported in v1)

        Returns:
            List of embedding vectors (one per text)

        Raises:
            NotImplementedError: If return_multivector is True
            EmbeddingError: If API call fails
        """
        if not texts:
            return []

        # Validate multi-vector request
        if return_multivector:
            raise NotImplementedError(
                "Multi-vector embeddings not supported in v1. Will be added in future release."
            )

        # Use configured dimensions if not specified
        dimensions = dims or self.default_dims

        # Validate dimensions (Jina v4 supports 128-2048)
        if dimensions < 128 or dimensions > 2048:
            raise ValueError(f"Dimensions must be between 128-2048, got {dimensions}")

        all_embeddings: List[List[float]] = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch_internal(batch, task, dimensions)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_batch_internal(
        self,
        texts: List[str],
        task: str,
        dimensions: int
    ) -> List[List[float]]:
        """
        Internal method to embed a single batch with exact Jina v4 API format.

        Args:
            texts: Batch of texts to embed
            task: Task type for adapter selection
            dimensions: Embedding dimensions

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If API call fails after retries
        """
        start_time = time.time()

        try:
            # Exact Jina v4 API format per documentation
            payload = {
                "model": self.model,
                "input": texts,
                "task": task,
                "dimensions": dimensions,
                "return_multivector": False,
                "late_chunking": False,
                "truncate": True
            }

            logger.debug(
                f"Embedding batch: {len(texts)} texts, task={task}, dims={dimensions}",
                extra={"batch_size": len(texts), "task": task, "dimensions": dimensions}
            )

            response = self.client.post_json(self.base_url, payload, headers=self.headers)

            # Validate response structure
            if not response or "data" not in response:
                raise EmbeddingError("Invalid response format from Jina API")

            embeddings = []
            for i, item in enumerate(response["data"]):
                if "embedding" not in item:
                    raise EmbeddingError(f"No embedding found for text {i}")

                embedding = item["embedding"]

                # Validate embedding dimensions
                if len(embedding) != dimensions:
                    raise EmbeddingError(
                        f"Unexpected embedding dimension: {len(embedding)}, expected {dimensions}"
                    )

                embeddings.append(embedding)

            # Validate we got embeddings for all inputs
            if len(embeddings) != len(texts):
                raise EmbeddingError(
                    f"Mismatch: requested {len(texts)} embeddings, got {len(embeddings)}"
                )

            duration_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"Successfully embedded {len(embeddings)} texts",
                extra=log_timing("embed_batch", duration_ms, batch_size=len(texts))
            )

            return embeddings

        except AuthError as e:
            # Auth errors are not retryable - log at debug level to reduce noise
            duration_ms = (time.time() - start_time) * 1000
            logger.debug(f"Embedding failed due to auth error: {e}")
            raise EmbeddingError(f"Authentication failed: {e}") from e

        except (NetworkError, ServerError) as e:
            # Network/server errors - log at warning level
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(f"Embedding failed due to network/server error: {e}")
            raise EmbeddingError(f"Network error: {e}") from e

        except Exception as e:
            # Other errors - full logging
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Embedding batch failed: {e}",
                extra=log_error(e, context={
                    "batch_size": len(texts),
                    "task": task,
                    "dimensions": dimensions,
                    "duration_ms": duration_ms
                })
            )
            raise EmbeddingError(f"Embedding failed: {e}") from e

    def embed_single(self, text: str, task: Optional[str] = None) -> List[float]:
        """
        Embed a single text string.

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
            dims: Optional dimension override

        Returns:
            Query embedding vector
        """
        embeddings = self.embed_texts([query], task="retrieval.query", dims=dims)
        return embeddings[0]

    def embed_passages(
        self,
        passages: List[str],
        dims: Optional[int] = None
    ) -> List[List[float]]:
        """
        Embed passages using retrieval.passage task.

        Args:
            passages: List of passage texts to embed
            dims: Optional dimension override

        Returns:
            List of passage embedding vectors
        """
        return self.embed_texts(passages, task="retrieval.passage", dims=dims)

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
