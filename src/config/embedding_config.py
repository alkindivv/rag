"""
Embedding Services Configuration
Single responsibility: Centralized configuration for all embedding services.
"""

import os
from typing import Dict, List, Any
from enum import Enum


class EmbeddingProvider(str, Enum):
    """Embedding service providers"""
    GEMINI = "gemini"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class EmbeddingModel(str, Enum):
    """Available embedding models"""
    GEMINI_EMBEDDING_001 = "models/embedding-001"
    GEMINI_TEXT_EMBEDDING_004 = "models/text-embedding-004"
    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"


class VectorStore(str, Enum):
    """Vector database options"""
    POSTGRES_PGVECTOR = "postgres_pgvector"
    CHROMA = "chroma"
    FAISS = "faiss"
    PINECONE = "pinecone"


class EmbeddingConfig:
    """
    Centralized configuration for embedding services
    Single source of truth for all embedding-related settings
    """

    # === PROVIDER CONFIGURATION ===

    # Primary embedding provider
    PRIMARY_PROVIDER = EmbeddingProvider(os.getenv("EMBEDDING_PRIMARY_PROVIDER", "gemini"))
    FALLBACK_PROVIDER = EmbeddingProvider(os.getenv("EMBEDDING_FALLBACK_PROVIDER", "openai"))

    # === GEMINI CONFIGURATION ===

    # Gemini API settings
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = EmbeddingModel(os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"))
    GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
    GEMINI_API_VERSION = os.getenv("GEMINI_API_VERSION", "v1")

    # Gemini performance settings
    GEMINI_BATCH_SIZE = int(os.getenv("GEMINI_BATCH_SIZE", "100"))
    GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
    GEMINI_RETRY_DELAY = float(os.getenv("GEMINI_RETRY_DELAY", "1.0"))
    GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "30"))
    GEMINI_RATE_LIMIT_RPM = int(os.getenv("GEMINI_RATE_LIMIT_RPM", "60"))

    # === TEXT PROCESSING CONFIGURATION ===

    # Text limits
    MAX_TEXT_LENGTH = int(os.getenv("EMBEDDING_MAX_TEXT_LENGTH", "8000"))
    MIN_TEXT_LENGTH = int(os.getenv("EMBEDDING_MIN_TEXT_LENGTH", "10"))
    TRUNCATE_STRATEGY = os.getenv("EMBEDDING_TRUNCATE_STRATEGY", "end")  # start, end, middle

    # Text preprocessing
    NORMALIZE_TEXT = os.getenv("EMBEDDING_NORMALIZE_TEXT", "true").lower() == "true"
    REMOVE_STOPWORDS = os.getenv("EMBEDDING_REMOVE_STOPWORDS", "false").lower() == "true"
    LOWERCASE_TEXT = os.getenv("EMBEDDING_LOWERCASE_TEXT", "false").lower() == "true"

    # === ADAPTIVE EMBEDDING CONFIGURATION ===

    # Legal document enhancement
    ENABLE_LEGAL_ENHANCEMENT = os.getenv("EMBEDDING_LEGAL_ENHANCEMENT", "true").lower() == "true"
    LEGAL_WEIGHT_FACTOR = float(os.getenv("EMBEDDING_LEGAL_WEIGHT", "1.2"))
    STRUCTURE_WEIGHT_FACTOR = float(os.getenv("EMBEDDING_STRUCTURE_WEIGHT", "1.1"))

    # Indonesian language optimization
    ENABLE_INDONESIAN_OPTIMIZATION = os.getenv("EMBEDDING_INDONESIAN_OPT", "true").lower() == "true"
    INDONESIAN_LEGAL_TERMS_WEIGHT = float(os.getenv("EMBEDDING_ID_LEGAL_WEIGHT", "1.3"))

    # === VECTOR STORAGE CONFIGURATION ===

    # Vector database settings
    VECTOR_STORE = VectorStore(os.getenv("VECTOR_STORE", "postgres_pgvector"))
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
    SIMILARITY_THRESHOLD = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", "0.7"))

    # Index configuration
    INDEX_TYPE = os.getenv("VECTOR_INDEX_TYPE", "ivfflat")  # ivfflat, hnsw
    INDEX_LISTS = int(os.getenv("VECTOR_INDEX_LISTS", "100"))
    INDEX_PROBES = int(os.getenv("VECTOR_INDEX_PROBES", "10"))

    # === CACHING CONFIGURATION ===

    # Embedding caching
    ENABLE_EMBEDDING_CACHE = os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL_HOURS = int(os.getenv("EMBEDDING_CACHE_TTL_HOURS", "24"))
    CACHE_MAX_SIZE = int(os.getenv("EMBEDDING_CACHE_MAX_SIZE", "10000"))

    # Cache storage
    CACHE_BACKEND = os.getenv("EMBEDDING_CACHE_BACKEND", "redis")  # redis, memory, file
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # === LEGAL DOCUMENT PATTERNS ===

    # Indonesian legal terminology
    LEGAL_KEYWORDS = [
        'pasal', 'ayat', 'huruf', 'angka', 'butir',
        'bab', 'bagian', 'paragraf',
        'undang-undang', 'peraturan', 'keputusan', 'instruksi',
        'ketentuan', 'kewenangan', 'wewenang', 'tugas', 'fungsi',
        'hak', 'kewajiban', 'larangan', 'sanksi',
        'pidana', 'denda', 'kurungan', 'penjara',
        'berlaku', 'mencabut', 'mengubah', 'menambah',
        'menetapkan', 'mengesahkan', 'memberlakukan'
    ]

    # Legal document structure markers
    STRUCTURE_MARKERS = {
        'chapter': ['bab', 'chapter'],
        'section': ['bagian', 'section'],
        'article': ['pasal', 'article'],
        'verse': ['ayat', 'verse'],
        'letter': ['huruf', 'letter'],
        'number': ['angka', 'number'],
        'point': ['butir', 'point']
    }

    # === PERFORMANCE CONFIGURATION ===

    # Batch processing
    DEFAULT_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
    MAX_CONCURRENT_REQUESTS = int(os.getenv("EMBEDDING_MAX_CONCURRENT", "5"))
    REQUEST_TIMEOUT = int(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "30"))

    # Memory management
    MAX_MEMORY_USAGE_MB = int(os.getenv("EMBEDDING_MAX_MEMORY_MB", "1024"))
    CLEANUP_INTERVAL_MINUTES = int(os.getenv("EMBEDDING_CLEANUP_INTERVAL", "30"))

    # === ERROR HANDLING CONFIGURATION ===

    # Retry configuration
    MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
    RETRY_BACKOFF_FACTOR = float(os.getenv("EMBEDDING_RETRY_BACKOFF", "2.0"))
    RETRY_JITTER = os.getenv("EMBEDDING_RETRY_JITTER", "true").lower() == "true"

    # Fallback behavior
    ENABLE_FALLBACK = os.getenv("EMBEDDING_ENABLE_FALLBACK", "true").lower() == "true"
    FALLBACK_TIMEOUT = int(os.getenv("EMBEDDING_FALLBACK_TIMEOUT", "10"))

    # === METHOD IMPLEMENTATIONS ===

    @classmethod
    def get_gemini_config(cls) -> Dict[str, Any]:
        """Get Gemini embedding service configuration"""
        return {
            "api_key": cls.GEMINI_API_KEY,
            "model": cls.GEMINI_MODEL,
            "base_url": cls.GEMINI_BASE_URL,
            "api_version": cls.GEMINI_API_VERSION,
            "batch_size": cls.GEMINI_BATCH_SIZE,
            "max_retries": cls.GEMINI_MAX_RETRIES,
            "retry_delay": cls.GEMINI_RETRY_DELAY,
            "timeout": cls.GEMINI_TIMEOUT,
            "rate_limit_rpm": cls.GEMINI_RATE_LIMIT_RPM
        }

    @classmethod
    def get_text_processing_config(cls) -> Dict[str, Any]:
        """Get text preprocessing configuration"""
        return {
            "max_length": cls.MAX_TEXT_LENGTH,
            "min_length": cls.MIN_TEXT_LENGTH,
            "truncate_strategy": cls.TRUNCATE_STRATEGY,
            "normalize": cls.NORMALIZE_TEXT,
            "remove_stopwords": cls.REMOVE_STOPWORDS,
            "lowercase": cls.LOWERCASE_TEXT
        }

    @classmethod
    def get_adaptive_config(cls) -> Dict[str, Any]:
        """Get adaptive embedding configuration"""
        return {
            "legal_enhancement": cls.ENABLE_LEGAL_ENHANCEMENT,
            "legal_weight": cls.LEGAL_WEIGHT_FACTOR,
            "structure_weight": cls.STRUCTURE_WEIGHT_FACTOR,
            "indonesian_optimization": cls.ENABLE_INDONESIAN_OPTIMIZATION,
            "indonesian_legal_weight": cls.INDONESIAN_LEGAL_TERMS_WEIGHT,
            "legal_keywords": cls.LEGAL_KEYWORDS.copy(),
            "structure_markers": cls.STRUCTURE_MARKERS.copy()
        }

    @classmethod
    def get_vector_config(cls) -> Dict[str, Any]:
        """Get vector storage configuration"""
        return {
            "store": cls.VECTOR_STORE,
            "dimension": cls.VECTOR_DIMENSION,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "index_type": cls.INDEX_TYPE,
            "index_lists": cls.INDEX_LISTS,
            "index_probes": cls.INDEX_PROBES
        }

    @classmethod
    def get_cache_config(cls) -> Dict[str, Any]:
        """Get caching configuration"""
        return {
            "enabled": cls.ENABLE_EMBEDDING_CACHE,
            "ttl_hours": cls.CACHE_TTL_HOURS,
            "max_size": cls.CACHE_MAX_SIZE,
            "backend": cls.CACHE_BACKEND,
            "redis_url": cls.REDIS_URL
        }

    @classmethod
    def get_performance_config(cls) -> Dict[str, Any]:
        """Get performance configuration"""
        return {
            "batch_size": cls.DEFAULT_BATCH_SIZE,
            "max_concurrent": cls.MAX_CONCURRENT_REQUESTS,
            "timeout": cls.REQUEST_TIMEOUT,
            "max_memory_mb": cls.MAX_MEMORY_USAGE_MB,
            "cleanup_interval": cls.CLEANUP_INTERVAL_MINUTES
        }

    @classmethod
    def get_error_handling_config(cls) -> Dict[str, Any]:
        """Get error handling configuration"""
        return {
            "max_retries": cls.MAX_RETRIES,
            "backoff_factor": cls.RETRY_BACKOFF_FACTOR,
            "jitter": cls.RETRY_JITTER,
            "enable_fallback": cls.ENABLE_FALLBACK,
            "fallback_timeout": cls.FALLBACK_TIMEOUT
        }

    @classmethod
    def validate_config(cls) -> Dict[str, List[str]]:
        """Validate configuration and return any issues"""
        issues = {"errors": [], "warnings": []}

        # Validate API keys
        if not cls.GEMINI_API_KEY:
            issues["errors"].append("GEMINI_API_KEY is required")

        # Validate numeric ranges
        if cls.VECTOR_DIMENSION <= 0:
            issues["errors"].append("VECTOR_DIMENSION must be positive")

        if not 0 <= cls.SIMILARITY_THRESHOLD <= 1:
            issues["errors"].append("SIMILARITY_THRESHOLD must be between 0 and 1")

        if cls.GEMINI_BATCH_SIZE <= 0:
            issues["warnings"].append("GEMINI_BATCH_SIZE should be positive")

        if cls.MAX_TEXT_LENGTH <= cls.MIN_TEXT_LENGTH:
            issues["errors"].append("MAX_TEXT_LENGTH must be greater than MIN_TEXT_LENGTH")

        # Validate weights
        if cls.LEGAL_WEIGHT_FACTOR < 0:
            issues["warnings"].append("LEGAL_WEIGHT_FACTOR should be non-negative")

        return issues

    @classmethod
    def get_provider_config(cls, provider: EmbeddingProvider) -> Dict[str, Any]:
        """Get configuration for specific provider"""
        if provider == EmbeddingProvider.GEMINI:
            return cls.get_gemini_config()
        # Add other providers as needed
        return {}


__all__ = [
    'EmbeddingProvider',
    'EmbeddingModel',
    'VectorStore',
    'EmbeddingConfig'
]
