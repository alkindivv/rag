from __future__ import annotations
"""Application settings using environment variables."""

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration loaded from environment."""

    database_url: str = Field(..., env="DATABASE_URL")
    jina_api_key: str | None = Field(default=None, env="JINA_API_KEY")
    jina_embed_base: str = Field(default="https://api.jina.ai/v1/embeddings", env="JINA_EMBED_BASE")
    jina_embed_model: str = Field(default="jina-embeddings-v4", env="JINA_EMBED_MODEL")
    jina_rerank_base: str = Field(default="https://api.jina.ai/v1/rerank", env="JINA_RERANK_BASE")
    jina_rerank_model: str = Field(default="jina-reranker-v2-base-multilingual", env="JINA_RERANK_MODEL")

    # Jina v4 Embedding Configuration
    embedding_model: str = Field(default="jina-embeddings-v4", env="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1024, env="EMBEDDING_DIM")  # 768, 1024, or 2048
    embedding_task_query: str = Field(default="retrieval.query", env="EMBEDDING_TASK_QUERY")
    embedding_task_passage: str = Field(default="retrieval.passage", env="EMBEDDING_TASK_PASSAGE")

    # Reranker Configuration
    reranker_model: str = Field(default="jina-reranker-v2-base-multilingual", env="RERANKER_MODEL")
    enable_reranker: bool = Field(default=True, env="ENABLE_RERANKER")

    llm_provider: str = Field(default="gemini", env="LLM_PROVIDER")
    llm_model: str = Field(default="gemini-2.0-flash-exp", env="LLM_MODEL")
    gemini_api_key: str | None = Field(default=None, env="GEMINI_API_KEY")
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, env="ANTHROPIC_API_KEY")
    embed_batch_size: int = Field(default=16, env="EMBED_BATCH_SIZE")
    rerank_provider: str = Field(default="none", env="RERANK_PROVIDER")
    enable_stream: bool = Field(default=False, env="ENABLE_STREAM")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra fields from .env file
    )


settings = Settings()
