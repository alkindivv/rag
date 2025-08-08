"""
PGVector Configuration
Simple configuration for PostgreSQL with pgvector extension.

Author: KISS Principle Implementation
Purpose: Simple pgvector database configuration
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class PGVectorConfig:
    """
    Simple but complete pgvector configuration.
    No overengineering, just what we need.
    """

    # Database Connection
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DB", "postgres")
    username: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "Batam123")

    # Connection Pool Settings
    pool_size: int = int(os.getenv("POSTGRES_POOL_SIZE", "5"))
    max_overflow: int = int(os.getenv("POSTGRES_MAX_OVERFLOW", "10"))
    pool_timeout: int = int(os.getenv("POSTGRES_POOL_TIMEOUT", "30"))
    pool_recycle: int = int(os.getenv("POSTGRES_POOL_RECYCLE", "1800"))

    # Vector Settings
    vector_dimension: int = int(os.getenv("VECTOR_DIMENSION", "768"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

    # Index Settings for HNSW
    hnsw_m: int = int(os.getenv("HNSW_M", "16"))
    hnsw_ef_construction: int = int(os.getenv("HNSW_EF_CONSTRUCTION", "64"))

    # Performance Settings
    search_timeout: int = int(os.getenv("SEARCH_TIMEOUT", "30"))
    max_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "100"))

    # SSL Settings
    ssl_mode: str = os.getenv("POSTGRES_SSL_MODE", "prefer")

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )

    @property
    def async_connection_string(self) -> str:
        """Generate async PostgreSQL connection string."""
        return (
            f"postgresql+asyncpg://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )

    def validate(self) -> bool:
        """Simple validation."""
        required_fields = [self.host, self.database, self.username, self.password]
        return all(field for field in required_fields)


# Global config instance
pgvector_config = PGVectorConfig()


# Simple helper functions
def get_connection_string() -> str:
    """Get PostgreSQL connection string."""
    return pgvector_config.connection_string


def get_async_connection_string() -> str:
    """Get async PostgreSQL connection string."""
    return pgvector_config.async_connection_string


def is_pgvector_configured() -> bool:
    """Check if pgvector is properly configured."""
    return pgvector_config.validate()
