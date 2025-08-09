from __future__ import annotations
"""Application settings using environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration loaded from environment."""

    database_url: str = Field(..., env="DATABASE_URL")
    jina_api_key: str | None = Field(default=None, env="JINA_API_KEY")
    llm_provider: str = Field(default="gemini", env="LLM_PROVIDER")
    gemini_api_key: str | None = Field(default=None, env="GEMINI_API_KEY")
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, env="ANTHROPIC_API_KEY")
    embed_batch_size: int = Field(default=16, env="EMBED_BATCH_SIZE")
    rerank_provider: str | None = Field(default=None, env="RERANK_PROVIDER")
    enable_stream: bool = Field(default=False, env="ENABLE_STREAM")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
