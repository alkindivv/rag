# AGENTS.md - Legal RAG Indonesia

## Overview

This repository implements a Retrieval-Augmented Generation (RAG) system specifically designed for Indonesian legal documents. The system handles various document types including UUD 1945, UU, PP, PERPRES, PERPU, PERMEN, POJK, SEOJK, SEMA, SEMK, and PERDA with constitutional-grade accuracy.

## Modular Architecture

The system is organized into the following modular agents:

1. [LLM Router & Intent-Aware Prompts](src/services/llm/AGENTS.md) - Handles LLM provider routing and intent-aware prompt templates
2. [Retrieval Router + Parallel FTS + Vector](src/services/retriever/AGENTS.md) - Implements query routing and parallel retrieval
3. [Hybrid Search Orchestrator](src/services/search/AGENTS.md) - Coordinates hybrid search pipeline
4. [Embedding Service (Jina v4)](src/services/embedding/AGENTS.md) - Generates embeddings with caching and dimension guard
5. [Answer Builder (JSON-first)](src/services/answers/AGENTS.md) - Builds structured answers with citations
6. [SQL Contracts & Queries](src/db/AGENTS.md) - Database queries for explicit, FTS, and vector search
7. [JSON Validator & Data Sanity Checks](src/validators/AGENTS.md) - Validates JSON schema compliance
8. [JSON Canonical Schema & Evolution](src/schemas/AGENTS.md) - Defines canonical JSON schema V2
9. [Indexer & Post-Index Housekeeping](pipeline/AGENTS.md) - Indexing pipeline and maintenance tasks
10. [Test Coverage & Validation](tests/AGENTS.md) - Comprehensive test coverage
11. [HTTP Client, Logging, Circuit Breaker, Metrics](src/utils/AGENTS.md) - Utility services

## Implementation Order

Follow this orchestrated plan:

1. [schemas](src/schemas/AGENTS.md) → [validators](src/validators/AGENTS.md) → [db](src/db/AGENTS.md) → [embedding](src/services/embedding/AGENTS.md)
2. [pipeline](pipeline/AGENTS.md) → [retriever](src/services/retriever/AGENTS.md) → [search](src/services/search/AGENTS.md)
3. [llm/prompts](src/services/llm/AGENTS.md) → [answers](src/services/answers/AGENTS.md) → [tests](tests/AGENTS.md)
4. [utils](src/utils/AGENTS.md) → [docs](docs/AGENTS.md)

## Quick Commands

```bash
# Validate all agents structure
python scripts/validate_agents.py

# Check status of a specific anchor
python scripts/checklist.py --status anchor:llm.router

# Run quick tests
python tests/run_tests.py --quick
```

## Global Roadmap

See [CHECKLIST.md](CHECKLIST.md) for the global implementation roadmap with stable anchor IDs that can be marked as done by sub-agents.

## Notes

- Keep each diff small and focused
- Document every public method with concise docstrings
- If ambiguous, prefer: more deterministic, simpler, less redundancy