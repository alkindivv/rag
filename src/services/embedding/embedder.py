from __future__ import annotations
"""Jina embedding client with batching and retry for v4 1024-dimensional embeddings."""

import logging
from typing import List, Optional

from src.config.settings import settings
from src.utils.http import HttpClient

logger = logging.getLogger(__name__)


class JinaEmbedder:
    """Embed text using Jina v4 API with 1024-dimensional vectors."""

    def __init__(self, client: Optional[HttpClient] = None) -> None:
        """Initialize Jina embedder with v4 model configuration."""
        self.client = client or HttpClient()
        self.base_url = settings.jina_embed_base
        self.model_name = settings.jina_embed_model
        self.batch_size = settings.embed_batch_size

        if not settings.jina_api_key:
            raise ValueError("JINA_API_KEY is required")

        self.headers = {"Authorization": f"Bearer {settings.jina_api_key}"}
        logger.info(f"Initialized JinaEmbedder with model: {self.model_name}")

    def embed_single(self, text: str) -> Optional[List[float]]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            List of 1024 floats or None if error
        """
        try:
            embeddings = self.embed_batch([text])
            return embeddings[0] if embeddings else None
        except Exception as e:
            logger.error(f"Error embedding single text: {e}")
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Embed a batch of texts with automatic chunking.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (1024-dim vectors) or None for failed embeddings
        """
        if not texts:
            return []

        all_embeddings: List[Optional[List[float]]] = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch_internal(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_batch_internal(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Internal method to embed a single batch.

        Args:
            texts: Batch of texts (within batch_size limit)

        Returns:
            List of embeddings or None for failures
        """
        try:
            payload = {
                "model": self.model_name,
                "input": texts,
                "encoding_format": "float"
            }

            logger.debug(f"Embedding batch of {len(texts)} texts")
            response = self.client.post_json(self.base_url, payload, headers=self.headers)

            if not response or "data" not in response:
                logger.error("Invalid response from Jina API")
                return [None] * len(texts)

            embeddings = []
            for i, item in enumerate(response["data"]):
                if "embedding" in item:
                    embedding = item["embedding"]
                    # Validate embedding dimensions
                    if len(embedding) == 1024:
                        embeddings.append(embedding)
                    else:
                        logger.warning(f"Unexpected embedding dimension: {len(embedding)}")
                        embeddings.append(None)
                else:
                    logger.warning(f"No embedding found for text {i}")
                    embeddings.append(None)

            # Pad with None if response is shorter than input
            while len(embeddings) < len(texts):
                embeddings.append(None)

            logger.debug(f"Successfully embedded {sum(1 for e in embeddings if e is not None)}/{len(texts)} texts")
            return embeddings

        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            return [None] * len(texts)

    def embed(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Legacy method for backward compatibility.

        Args:
            texts: List of texts to embed
            batch_size: Optional batch size override

        Returns:
            List of embeddings (filters out None values)
        """
        old_batch_size = self.batch_size
        if batch_size:
            self.batch_size = batch_size

        try:
            embeddings = self.embed_batch(texts)
            # Filter out None values for backward compatibility
            return [emb for emb in embeddings if emb is not None]
        finally:
            self.batch_size = old_batch_size
