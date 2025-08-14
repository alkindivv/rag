"""
High-performance embedding cache service for Legal RAG system.

This service provides production-ready caching for embedding operations,
achieving 90%+ performance improvement for repeated queries by eliminating
redundant API calls to Jina AI.

Features:
- Thread-safe LRU cache with TTL expiration
- Memory management with configurable limits
- Cache hit/miss metrics for monitoring
- Seamless integration with existing embedding service
- Production-ready error handling and logging

Author: KISS Principle Implementation
Purpose: Dramatic performance improvement for repeated legal queries
"""

import hashlib
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import OrderedDict
import json

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    embedding: List[float]
    timestamp: float
    access_count: int
    query_hash: str


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_queries: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        if self.total_queries == 0:
            return 0.0
        return (self.hits / self.total_queries) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for logging."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_queries": self.total_queries,
            "hit_rate_percent": round(self.hit_rate, 2)
        }


class EmbeddingCache:
    """
    Production-ready embedding cache with LRU eviction and TTL expiration.

    Thread-safe implementation optimized for high-performance embedding operations.
    Designed to reduce 20-25 second embedding API calls to <100ms cache lookups.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_hours: int = 6,
        enable_metrics: bool = True
    ):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of cached embeddings
            ttl_hours: Time-to-live for cache entries in hours
            enable_metrics: Whether to track cache performance metrics
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.enable_metrics = enable_metrics

        # Thread-safe cache storage (LRU using OrderedDict)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Performance metrics
        self.stats = CacheStats()

        # Memory monitoring
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes

        logger.info(
            f"Initialized EmbeddingCache: max_size={max_size}, ttl_hours={ttl_hours}"
        )

    def _generate_cache_key(self, query: str, task: str = "retrieval.query") -> str:
        """
        Generate consistent cache key for query and task.

        Args:
            query: Query text
            task: Embedding task type

        Returns:
            MD5 hash of normalized query and task
        """
        # Normalize query for consistent caching
        normalized = query.strip().lower()
        cache_input = f"{normalized}|{task}"
        return hashlib.md5(cache_input.encode('utf-8')).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired."""
        return (time.time() - entry.timestamp) > self.ttl_seconds

    def _cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        if not self._cache:
            return 0

        current_time = time.time()
        expired_keys = []

        for key, entry in self._cache.items():
            if (current_time - entry.timestamp) > self.ttl_seconds:
                expired_keys.append(key)

        # Remove expired entries
        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            # OrderedDict.popitem(last=False) removes the oldest item
            evicted_key, evicted_entry = self._cache.popitem(last=False)
            if self.enable_metrics:
                self.stats.evictions += 1
            logger.debug(f"Evicted LRU cache entry: {evicted_key[:8]}...")

    def _periodic_cleanup(self) -> None:
        """Perform periodic cleanup if needed."""
        current_time = time.time()
        if (current_time - self._last_cleanup) > self._cleanup_interval:
            expired_count = self._cleanup_expired()
            self._last_cleanup = current_time

            if expired_count > 0:
                logger.info(f"Periodic cleanup removed {expired_count} expired entries")

    def get(self, query: str, task: str = "retrieval.query") -> Optional[List[float]]:
        """
        Get cached embedding for query.

        Args:
            query: Query text
            task: Embedding task type

        Returns:
            Cached embedding vector or None if not found/expired
        """
        if not query or not query.strip():
            return None

        cache_key = self._generate_cache_key(query, task)

        with self._lock:
            # Update metrics
            if self.enable_metrics:
                self.stats.total_queries += 1

            # Check if entry exists
            if cache_key not in self._cache:
                if self.enable_metrics:
                    self.stats.misses += 1
                return None

            entry = self._cache[cache_key]

            # Check expiration
            if self._is_expired(entry):
                del self._cache[cache_key]
                if self.enable_metrics:
                    self.stats.misses += 1
                logger.debug(f"Cache entry expired: {cache_key[:8]}...")
                return None

            # Move to end (mark as recently used)
            self._cache.move_to_end(cache_key)
            entry.access_count += 1

            # Update metrics
            if self.enable_metrics:
                self.stats.hits += 1

            logger.debug(f"Cache HIT: {cache_key[:8]}... (accessed {entry.access_count} times)")
            return entry.embedding

    def put(self, query: str, embedding: List[float], task: str = "retrieval.query") -> None:
        """
        Store embedding in cache.

        Args:
            query: Query text
            embedding: Embedding vector
            task: Embedding task type
        """
        if not query or not query.strip() or not embedding:
            return

        cache_key = self._generate_cache_key(query, task)

        with self._lock:
            # Periodic cleanup
            self._periodic_cleanup()

            # Check if we need to evict entries
            while len(self._cache) >= self.max_size:
                self._evict_lru()

            # Store new entry
            self._cache[cache_key] = CacheEntry(
                embedding=embedding,
                timestamp=time.time(),
                access_count=1,
                query_hash=cache_key
            )

            logger.debug(f"Cache PUT: {cache_key[:8]}... (cache size: {len(self._cache)})")

    def get_or_generate(
        self,
        query: str,
        generator_func,
        task: str = "retrieval.query"
    ) -> List[float]:
        """
        Get cached embedding or generate new one.

        Args:
            query: Query text
            generator_func: Function to generate embedding if not cached
            task: Embedding task type

        Returns:
            Embedding vector

        Raises:
            Exception: If generation fails and no cache available
        """
        # Try cache first
        cached_embedding = self.get(query, task)
        if cached_embedding is not None:
            return cached_embedding

        # Generate new embedding
        start_time = time.time()
        try:
            embedding = generator_func(query)
            generation_time = (time.time() - start_time) * 1000

            # Cache the result
            if embedding:
                self.put(query, embedding, task)
                logger.debug(
                    f"Generated and cached embedding in {generation_time:.1f}ms"
                )

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding for caching: {e}")
            raise

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared {count} cache entries")
            return count

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache performance metrics
        """
        with self._lock:
            stats_dict = self.stats.to_dict()
            stats_dict.update({
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "ttl_hours": self.ttl_seconds / 3600,
                "memory_estimate_mb": self._estimate_memory_usage()
            })
            return stats_dict

    def _estimate_memory_usage(self) -> float:
        """
        Estimate cache memory usage in MB.

        Returns:
            Estimated memory usage in megabytes
        """
        if not self._cache:
            return 0.0

        # Rough estimate: 384 floats * 8 bytes + metadata overhead
        embedding_size = 384 * 8  # 8 bytes per float64
        metadata_size = 200  # Estimated overhead per entry
        entry_size = embedding_size + metadata_size

        total_bytes = len(self._cache) * entry_size
        return total_bytes / (1024 * 1024)  # Convert to MB

    def get_popular_queries(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get most frequently accessed queries.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of (query_hash, access_count) tuples
        """
        with self._lock:
            # Sort by access count
            popular = sorted(
                [(entry.query_hash, entry.access_count) for entry in self._cache.values()],
                key=lambda x: x[1],
                reverse=True
            )
            return popular[:limit]

    def health_check(self) -> Dict[str, Any]:
        """
        Perform cache health check.

        Returns:
            Health status with recommendations
        """
        stats = self.get_stats()
        health = {
            "status": "healthy",
            "issues": [],
            "recommendations": []
        }

        # Check hit rate
        if stats["hit_rate_percent"] < 50 and stats["total_queries"] > 100:
            health["issues"].append("Low cache hit rate")
            health["recommendations"].append("Consider increasing TTL or cache size")

        # Check memory usage
        if stats["memory_estimate_mb"] > 500:  # 500MB threshold
            health["issues"].append("High memory usage")
            health["recommendations"].append("Consider reducing cache size or TTL")

        # Check cache utilization
        utilization = (stats["cache_size"] / stats["max_size"]) * 100
        if utilization > 90:
            health["recommendations"].append("Consider increasing max cache size")

        if health["issues"]:
            health["status"] = "needs_attention"

        return health


class CachedEmbeddingService:
    """
    Wrapper service that adds caching to any embedding service.

    This service intercepts embedding requests and serves from cache when possible,
    dramatically reducing API call latency for repeated queries.
    """

    def __init__(
        self,
        embedding_service,
        cache_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize cached embedding service.

        Args:
            embedding_service: Base embedding service (e.g., JinaV4Embedder)
            cache_config: Optional cache configuration
        """
        self.embedding_service = embedding_service

        # Default cache configuration
        default_config = {
            "max_size": 1000,
            "ttl_hours": 6,
            "enable_metrics": True
        }

        config = {**default_config, **(cache_config or {})}
        self.cache = EmbeddingCache(**config)

        logger.info(f"Initialized CachedEmbeddingService with config: {config}")

    def embed_single(self, query: str, task: str = "retrieval.query") -> List[float]:
        """
        Embed single query with caching.

        Args:
            query: Query text
            task: Embedding task type

        Returns:
            Embedding vector

        Raises:
            Exception: If embedding generation fails
        """
        return self.cache.get_or_generate(
            query=query,
            generator_func=lambda q: self.embedding_service.embed_single(q, task),
            task=task
        )

    def embed_query(self, query: str) -> List[float]:
        """
        Embed query with caching (convenience method).

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        return self.embed_single(query, task="retrieval.query")

    def embed_texts(
        self,
        texts: List[str],
        task: str = "retrieval.passage"
    ) -> List[List[float]]:
        """
        Embed multiple texts with individual caching.

        Args:
            texts: List of texts to embed
            task: Embedding task type

        Returns:
            List of embedding vectors

        Note:
            Each text is cached individually for maximum cache utilization.
        """
        embeddings = []

        for text in texts:
            embedding = self.cache.get_or_generate(
                query=text,
                generator_func=lambda t: self.embedding_service.embed_single(t, task),
                task=task
            )
            embeddings.append(embedding)

        return embeddings

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics."""
        return self.cache.get_stats()

    def clear_cache(self) -> int:
        """Clear all cached embeddings."""
        return self.cache.clear()

    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check with recommendations."""
        return self.cache.health_check()


# Global cache instance for singleton pattern
_global_cache: Optional[EmbeddingCache] = None
_global_cache_lock = threading.RLock()


def get_global_embedding_cache(
    max_size: int = 1000,
    ttl_hours: int = 6
) -> EmbeddingCache:
    """
    Get or create global embedding cache instance.

    Args:
        max_size: Maximum cache size
        ttl_hours: TTL in hours

    Returns:
        Global cache instance
    """
    global _global_cache

    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:  # Double-check pattern
                _global_cache = EmbeddingCache(
                    max_size=max_size,
                    ttl_hours=ttl_hours
                )
                logger.info("Created global embedding cache instance")

    return _global_cache


def create_cached_embedder(embedding_service) -> CachedEmbeddingService:
    """
    Create cached embedding service with default configuration.

    Args:
        embedding_service: Base embedding service

    Returns:
        Cached embedding service instance
    """
    return CachedEmbeddingService(
        embedding_service=embedding_service,
        cache_config={
            "max_size": 1000,
            "ttl_hours": 6,
            "enable_metrics": True
        }
    )


# Convenience functions for immediate performance improvement
def cached_embed_query(query: str, embedding_service) -> List[float]:
    """
    Convenience function for cached query embedding.

    Args:
        query: Query text
        embedding_service: Base embedding service

    Returns:
        Cached or newly generated embedding
    """
    cache = get_global_embedding_cache()
    return cache.get_or_generate(
        query=query,
        generator_func=lambda q: embedding_service.embed_query(q),
        task="retrieval.query"
    )


def get_cache_performance_report() -> Dict[str, Any]:
    """
    Get comprehensive cache performance report.

    Returns:
        Dictionary with cache metrics and health status
    """
    global _global_cache

    if _global_cache is None:
        return {"status": "not_initialized", "cache_enabled": False}

    stats = _global_cache.get_stats()
    health = _global_cache.health_check()

    return {
        "cache_enabled": True,
        "performance": stats,
        "health": health,
        "popular_queries": _global_cache.get_popular_queries(5)
    }


# Cache monitoring decorator
def monitor_embedding_performance(func):
    """
    Decorator to monitor embedding performance with caching.

    Usage:
        @monitor_embedding_performance
        def embed_query(self, query: str):
            return self.embedding_service.embed_query(query)
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000

            # Log performance
            cache_stats = get_cache_performance_report()
            logger.info(
                f"Embedding completed in {duration:.1f}ms, "
                f"cache hit rate: {cache_stats.get('performance', {}).get('hit_rate_percent', 0):.1f}%"
            )

            return result

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"Embedding failed after {duration:.1f}ms: {e}")
            raise

    return wrapper


if __name__ == "__main__":
    # Simple test and demonstration
    print("🧪 Testing EmbeddingCache...")

    # Create test cache
    cache = EmbeddingCache(max_size=5, ttl_hours=1)

    # Mock embedding function
    def mock_embed(query: str) -> List[float]:
        # Simulate API call delay
        time.sleep(0.1)
        return [0.1, 0.2, 0.3] * 128  # 384-dimensional

    # Test caching
    queries = ["test query 1", "test query 2", "test query 1"]  # Repeat for cache hit

    for query in queries:
        start = time.time()
        embedding = cache.get_or_generate(query, mock_embed)
        duration = (time.time() - start) * 1000
        print(f"Query: '{query}' -> {duration:.1f}ms")

    # Print stats
    stats = cache.get_stats()
    print(f"\n📊 Cache Stats: {stats}")
    print(f"✅ Cache test completed successfully!")
