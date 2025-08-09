## Utilities and Configuration (`src/config/*` and helpers)
- Adopt `pydantic.BaseSettings` for environment configuration to reduce manual `.env` parsing.
- Normalise naming across config classes and remove unused options to simplify onboarding.
- Provide a single `AppConfig` aggregator that composes service-specific settings and centralises logging configuration.


src/config/

Gunalan pydantic settings v2

## backend/src/config/settings.py
Variabel env: 

# src/config/settings.py
database_url: str = Field(..., env="DATABASE_URL")

# LLM
llm_provider: str = Field("gemini", env="LLM_PROVIDER")  # gemini|openai|anthropic
gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")

# Embedding (Jina v4)
jina_api_key: str = Field(..., env="JINA_API_KEY")
jina_embed_base: str = Field("https://api.jina.ai/v1/embeddings", env="JINA_EMBED_BASE")
jina_embed_model: str = Field("jina-embeddings-v4", env="JINA_EMBED_MODEL")
embed_batch_size: int = Field(16, env="EMBED_BATCH_SIZE")

# Reranker (opsional)
rerank_provider: str = Field("none", env="RERANK_PROVIDER")  # jina|none
jina_rerank_base: str = Field("https://api.jina.ai/v1/rerank", env="JINA_RERANK_BASE")
jina_rerank_model: str = Field("jina-reranker-v1", env="JINA_RERANK_MODEL")

# Server
enable_stream: bool = Field(True, env="ENABLE_STREAM")
log_level: str = Field("INFO", env="LOG_LEVEL")

class Config:
    env_file = ".env"

settings = Settings()

backend/src/config/config.py
Ini digunakan untuk mengatur variabel global agar tifak ada hardcoded pattern di setiap file > pindahkan seluruh src/*.config ke dalam config.py global ini, pastikam yang dipindahlan adalah seluruh variable yang hanya digunakan saja, bukan menimpa seluruh src/*.config kedalam config.py, namun melakukan seleksi dan anisis variable apa saja yang memang digunakan