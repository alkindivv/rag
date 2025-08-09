# Code Review and Structural Suggestions

## Ingestion Pipeline (`src/ingestion.py`)
- Separate command-line parsing from business logic for easier testing.
- Consider converting sequential PDF processing to asynchronous tasks to utilise the existing event loop.
- Use structured logging instead of print statements for consistent output and easier log aggregation.
- Wrap file I/O with context managers and handle potential JSON decoding errors explicitly.

## Crawler Services (`src/services/crawler/*`)
- Consolidate duplicated crawler modules under `src/crawler` and `src/services/crawler` to avoid maintenance divergence.
- Implement retry/backoff utilities as shared helpers rather than inline loops for clearer flow.
- Use dependency injection for the HTTP client session to improve testability.
- Break large methods such as `_extract_uji_materi_decisions` into smaller units and add type hints for intermediate structures.

## PDF Processing (`src/pdf/*` and `src/services/pdf/*`)
- Merge duplicate orchestrator modules and keep a single entry point for PDF handling.
- Move hierarchy pattern definitions to a configuration or data file to reduce clutter in `PDFOrchestrator`.
- Streaming downloads could write directly to disk instead of holding all chunks in memory.
- Introduce unit tests for tree-building logic to ensure Indonesian legal hierarchy is parsed correctly.

## Text Cleaning (`src/utils/text_cleaner.py`)
- The monolithic `clean_legal_document_comprehensive` performs many responsibilities; break it into a pipeline of composable steps.
- Maintain a registry of cleaning operations with clear names and allow enabling/disabling steps via configuration.
- Add profiling or benchmarking hooks to understand the cost of each cleaning stage.

## Embedding Service (`src/services/embedding/embedder.py`)
- Provide optional batching to reduce API calls when embedding many chunks.
- Cache API key validation and model configuration to avoid repeated setup in every instantiation.
- Expose an interface that accepts an HTTP client to ease mocking during tests.
- Replace `time.sleep` with asyncio-aware rate limiting to avoid blocking the event loop.

## AI Question Answering (`src/services/ai/*`)
- Load prompts and templates from external files to decouple content from code.
- Introduce guardrails for user input (length limits, profanity filtering) before sending to the model.
- Record detailed telemetry (prompt tokens, latency) for monitoring model usage and costs.
- Provide graceful fallbacks when the model is unavailable instead of returning generic errors.

## Vector and Document Models (`src/models/*`)
- Add SQLAlchemy relationships between vector and document tables for easier joins.
- Use `Enum` types for fields like `doc_type` and `doc_status` to enforce valid values at the schema level.
- Store `subject_areas` as a separate table with many-to-many relationships to enable efficient querying and analytics.
- Include migration scripts or alembic configurations to manage schema evolution.

## Utilities and Configuration (`src/config/*` and helpers)
- Adopt `pydantic.BaseSettings` for environment configuration to reduce manual `.env` parsing.
- Normalise naming across config classes and remove unused options to simplify onboarding.
- Provide a single `AppConfig` aggregator that composes service-specific settings and centralises logging configuration.

## General Recommendations
- Add type-checked unit tests and integration tests; the repository currently lacks automated tests.
- Apply automatic formatting (e.g., `black` or `ruff`) and enforce linting in CI to maintain code quality.
- Create contributor documentation describing the high-level architecture and data flow.
- Document external dependencies (Gemini, pgvector, PostgreSQL) with setup instructions for new developers.
