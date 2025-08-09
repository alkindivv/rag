# Legal RAG System - Agent Context Documentation

## 🎯 Project Overview

**Legal RAG System** adalah sistem Retrieval-Augmented Generation (RAG) untuk dokumen hukum Indonesia yang mengkombinasikan:
- **Full-Text Search (FTS)** untuk pencarian eksplisit (referensi pasal spesifik)
- **Vector Search** untuk pencarian semantik/tematik
- **Hybrid Retrieval** dengan optional reranking untuk hasil optimal
- **LLM Integration** untuk menjawab pertanyaan hukum dengan sitasi akurat

### Primary Goals
- **Akurat & Cepat**: Hybrid retrieval (FTS leaf + vector pasal + rerank)
- **Sederhana & Maintainable**: Modul <300 LOC/file, tipe jelas, logging terstruktur
- **Production-ready**: Alembic migrations, retry/backoff, DI untuk HTTP client

## 🏗️ Current Architecture

```
Rag/
├── src/
│   ├── main.py                    # ✅ CLI entry point
│   ├── config/
│   │   ├── settings.py            # ✅ Pydantic v2 settings (consolidated)
│   │   ├── config.md              # ✅ Configuration documentation
│   │   └── [legacy configs]       # ⚠️ Multiple legacy config files exist
│   ├── db/
│   │   ├── models.py              # ✅ SQLAlchemy models (complete)
│   │   ├── session.py             # ✅ DB session management
│   │   └── migrations/            # ✅ Alembic setup
│   ├── pipeline/
│   │   ├── indexer.py             # ✅ JSON→DB indexer (with deduplication)
│   │   └── post_index.py          # ❌ TODO: Post-processing
│   ├── services/
│   │   ├── embedding/
│   │   │   └── embedder.py        # ✅ Jina v4 1024-dim embeddings
│   │   ├── search/
│   │   │   ├── reranker.py        # ✅ Jina reranker service
│   │   │   └── hybrid_search.py   # ✅ Main search orchestrator
│   │   ├── retriever/
│   │   │   └── hybrid_retriever.py # ✅ FTS + Vector + Explicit search
│   │   ├── llm/                   # ⚠️ Exists but needs integration
│   │   ├── api/                   # ⚠️ Exists but needs FastAPI app
│   │   ├── crawler/               # ✅ Existing crawler (needs light refactor)
│   │   └── pdf/                   # ✅ Existing PDF processing
│   └── utils/
│       ├── logging.py             # ✅ Structured JSON logging
│       ├── http.py                # ✅ Retry/backoff HTTP client
│       ├── text_cleaner.py        # ✅ Existing text cleaning
│       └── pattern_manager.py     # ✅ Existing legal hierarchy patterns
├── data/
│   ├── json/                      # ✅ Processed JSON documents
│   ├── pdfs/                      # ✅ Raw PDF files
│   └── text/                      # ✅ Text extracts
├── requirements.txt               # ✅ Updated with all dependencies
├── alembic.ini                    # ✅ Alembic configuration
├── .env.example                   # ✅ Environment template
└── [docs and plans]               # ✅ Various documentation files
```

## 📊 Database Schema

### Core Tables

#### `legal_documents` - Main document metadata
```sql
- id: UUID (PK)
- doc_id: VARCHAR(100) UNIQUE  -- e.g., "UU-2025-2"
- doc_form: ENUM (UU, PP, PERPU, PERPRES, POJK, PERMEN, PERDA, LAINNYA, SE)
- doc_number, doc_year, doc_title, doc_status
- doc_relationships: JSONB  -- Raw relationships from crawler
- doc_uji_materi: JSONB     -- Raw court decisions
- [metadata fields...]
```

#### `legal_units` - Hierarchical document structure
```sql
- id: UUID (PK)
- document_id: UUID (FK to legal_documents)
- unit_type: ENUM (dokumen, buku, bab, bagian, paragraf, pasal, ayat, huruf, angka, angka_amandement)
- unit_id: VARCHAR(500) UNIQUE  -- e.g., "UU-2025-2/pasal-1/ayat-2"
- parent_pasal_id: VARCHAR(500) -- For leaf nodes
- content, local_content, bm25_body: TEXT
- content_vector: TSVECTOR  -- For FTS
- path: JSONB, citation_string: TEXT
```

#### `document_vectors` - Vector embeddings (pasal-level)
```sql
- id: UUID (PK)
- document_id: UUID (FK to legal_documents)
- unit_id: VARCHAR(500)  -- Pasal unit_id
- embedding: VECTOR(1024)  -- Jina v4 embeddings
- doc_form, doc_year, doc_number: Metadata for fast filtering
- pasal_number, bab_number, ayat_number: Hierarchy metadata
```

#### `subjects` + `document_subject` - Document topics
```sql
- Many-to-many relationship for document categorization
```

### Key Constraints & Indexes
- **Unique**: `(doc_form, doc_number, doc_year)` on `legal_documents`
- **Unique**: `(document_id, unit_id)` on `legal_units`
- **GIN**: `content_vector` for FTS on `legal_units`
- **HNSW**: `embedding` for vector similarity on `document_vectors`

## 🔄 Data Flow

### 1. Ingestion Pipeline
```
PDF/JSON Files → Crawler/Parser → JSON Output → Indexer → Database
```

**Current JSON Structure** (from `data/json/*.json`):
```json
{
  "doc_id": "UU-2025-2",
  "doc_form": "UU", 
  "doc_number": "2",
  "doc_year": "2025",
  "doc_title": "Undang-undang...",
  "doc_subject": ["PERTAMBANGAN MIGAS", "MINERAL DAN ENERGI"],
  "relationships": {"mengubah": [...]},
  "uji_materi": [...],
  "document_tree": {
    "doc_type": "document",
    "children": [
      {
        "type": "huruf|pasal|ayat|angka",
        "unit_id": "UU-2025-2/pasal-1/ayat-2",
        "number_label": "2",
        "local_content": "...",
        "citation_string": "...",
        "path": [...],
        "parent_pasal_id": "UU-2025-2/pasal-1",
        "children": [...]
      }
    ]
  }
}
```

### 2. Search Pipeline
```
Query → Router → [Explicit|Thematic] → [FTS + Vector] → Rerank → Results
```

**Query Types**:
- **Explicit**: "pasal 1 ayat 2", "UU 4/2009" → Direct lookup via regex + SQL
- **Thematic**: "pertambangan mineral" → Semantic vector search + FTS

## 🔧 Current Implementation Status

### ✅ COMPLETED
1. **Database Models** - Complete SQLAlchemy models with proper relationships
2. **Settings Management** - Pydantic v2 with environment variable loading
3. **Indexer** - JSON→DB with deduplication logic and optional embedding skip
4. **Hybrid Retriever** - FTS + Vector + Explicit search routing
5. **Reranker Service** - Jina reranker integration with fallback
6. **HTTP Client** - Retry/backoff utility with structured logging
7. **CLI Interface** - Comprehensive CLI for testing and management

### ⚠️ PARTIALLY IMPLEMENTED
1. **Embedding Service** - Jina v4 integration (API format issues)
2. **Search Service** - Hybrid search orchestrator (SQL query issues)
3. **Legacy Services** - Existing crawler/PDF services need light refactoring

### ❌ TODO
1. **FastAPI Application** - REST API endpoints
2. **LLM Integration** - Gemini/OpenAI/Anthropic providers with prompt management
3. **Frontend** - Next.js application
4. **Production Deployment** - Docker, environment configs
5. **Testing Suite** - Comprehensive unit/integration tests

## 🚨 Current Issues & Blockers

### 1. **Jina API Integration Issues**
- **Problem**: Jina embeddings API returning 422 errors
- **Likely Cause**: Request format mismatch with Jina v4 API
- **Impact**: Vector search not functional
- **Priority**: HIGH

### 2. **SQL Query Formatting**
- **Problem**: SQL template `.format()` method failing in retriever
- **Cause**: SQLAlchemy text() object doesn't support `.format()`
- **Impact**: Hybrid search failing
- **Priority**: HIGH

### 3. **JSON Data Duplicates**
- **Problem**: Crawler output contains duplicate `unit_id` entries
- **Workaround**: Deduplication logic in indexer
- **Root Cause**: PDF parsing logic needs review
- **Priority**: MEDIUM

### 4. **Import Path Inconsistencies**
- **Problem**: Mixed relative/absolute imports causing module errors
- **Status**: Partially resolved
- **Priority**: MEDIUM

## 🛠️ Service Responsibilities

### **Indexer** (`src/pipeline/indexer.py`)
- **Input**: JSON files from `data/json/`
- **Output**: Records in `legal_documents`, `legal_units`, `document_vectors`
- **Features**: Batch processing, deduplication, optional embedding skip
- **Dependencies**: JinaEmbedder, Database session

### **Hybrid Retriever** (`src/services/retriever/hybrid_retriever.py`)
- **Input**: Query string + optional filters
- **Output**: Ranked `SearchResult` objects
- **Strategies**: 
  - Explicit: Regex-based legal reference parsing
  - FTS: PostgreSQL full-text search on leaf units
  - Vector: Cosine similarity on pasal embeddings
- **Dependencies**: Database session, JinaEmbedder

### **Search Service** (`src/services/search/hybrid_search.py`)
- **Input**: User queries
- **Output**: Formatted search responses
- **Features**: Query routing, result combination, logging
- **Dependencies**: HybridRetriever, Reranker

### **Embedding Service** (`src/services/embedding/embedder.py`)
- **Provider**: Jina v4 (1024-dimensional)
- **Features**: Batch processing, retry logic, error handling
- **Configuration**: Configurable batch size and model

### **Reranker Service** (`src/services/search/reranker.py`)
- **Provider**: Jina reranker v1
- **Fallback**: NoOp reranker when disabled/unavailable
- **Features**: Top-k limiting, score normalization

## 🔌 Configuration

### Environment Variables (via `src/config/settings.py`)
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/legal_rag

# Jina AI
JINA_API_KEY=your_key_here
JINA_EMBED_MODEL=jina-embeddings-v4
JINA_RERANK_MODEL=jina-reranker-v1

# LLM Providers
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Processing
EMBED_BATCH_SIZE=16
RERANK_PROVIDER=jina
LOG_LEVEL=INFO
```

## 🧪 Testing & Usage

### Database Operations
```bash
# Initialize fresh database
python -m src.main init-db --reset --force

# Index documents (skip embeddings for testing)
python -m src.main index data/json --skip-embeddings

# Check system status
python -m src.main status
```

### Search Testing
```bash
# Test explicit search
python -m src.main search "pasal 1 ayat 2"

# Test thematic search
python -m src.main search "pertambangan mineral"

# Test with filters
python -m src.main search "batubara" --doc-forms UU --doc-years 2009,2025

# Get document outline
python -m src.main outline UU-2025-2
```

## 📋 API Contracts (Planned)

### Search Endpoints
```
GET  /search?q={query}&limit={n}&filters={...}  # Raw search results
POST /ask {query, context, options}              # LLM-powered Q&A
GET  /health                                     # System health check
```

### Data Formats
```typescript
// Search Result
interface SearchResult {
  id: string;
  text: string;
  citation: string;
  score: number;
  source_type: "fts" | "vector" | "explicit" | "reranked";
  unit_type: string;
  document: {
    form: string;
    year: number;
    number: string;
  };
  metadata?: Record<string, any>;
}

// Search Response
interface SearchResponse {
  results: SearchResult[];
  total: number;
  query: string;
  strategy: "explicit" | "thematic" | "contextual";
  reranked: boolean;
  duration_ms: number;
}
```

## 🎨 Code Patterns & Standards

### Import Style
```python
# Absolute imports from project root
from src.config.settings import settings
from src.db.models import LegalDocument
from src.services.embedding.embedder import JinaEmbedder
```

### Error Handling
```python
# Structured logging with context
try:
    result = operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}", extra=log_error(e, context=context))
    return fallback_result
```

### Dependency Injection
```python
# Accept optional dependencies for testability
def __init__(self, client: Optional[HttpClient] = None):
    self.client = client or HttpClient()
```

### Configuration
```python
# Use Pydantic settings for all config
class ServiceConfig(BaseSettings):
    api_key: str = Field(..., env="API_KEY")
    batch_size: int = Field(16, env="BATCH_SIZE")
```

## 🗂️ Data Hierarchy

### Indonesian Legal Structure
```
Dokumen (UU-2025-2)
├── Bab I
│   ├── Pasal 1
│   │   ├── Ayat (1)
│   │   │   ├── Huruf a
│   │   │   └── Huruf b
│   │   └── Ayat (2)
│   └── Pasal 2
└── Bab II
    └── ...
```

### Unit ID Naming Convention
- **Document**: `UU-2025-2`
- **Pasal**: `UU-2025-2/pasal-1`
- **Ayat**: `UU-2025-2/pasal-1/ayat-2`
- **Huruf**: `UU-2025-2/pasal-1/ayat-2/huruf-a`
- **Angka**: `UU-2025-2/pasal-1/ayat-2/huruf-a/angka-1`

### Content Storage Strategy
- **Leaf Units** (ayat, huruf, angka): Store in `legal_units` with FTS indexing
- **Pasal Units**: Aggregate content + generate embeddings → `document_vectors`
- **Hierarchy Navigation**: Use `parent_pasal_id` and `path` JSONB

## 🔍 Search Strategies

### 1. Explicit Search (Direct References)
```python
# Query: "pasal 1 ayat 2" or "UU 4/2009"
# Action: Regex parsing → Direct SQL lookup
# Target: Exact legal units via number_label matching
```

### 2. FTS Search (Keyword-based)
```sql
-- Target: Leaf units (ayat, huruf, angka)
SELECT * FROM legal_units 
WHERE unit_type IN ('ayat','huruf','angka')
  AND content_vector @@ plainto_tsquery('indonesian', :query)
ORDER BY ts_rank(content_vector, plainto_tsquery('indonesian', :query)) DESC
```

### 3. Vector Search (Semantic)
```sql
-- Target: Pasal-level content
SELECT * FROM document_vectors 
ORDER BY embedding <=> :query_vector 
LIMIT :k
```

### 4. Hybrid Combination
- **Thematic queries**: Combine FTS + Vector → Deduplicate → Rerank
- **Explicit queries**: Direct lookup + FTS supplement if needed

## 🧩 Key Components Integration

### Retrieval Flow
```python
# 1. Query Analysis
is_explicit = QueryRouter.is_explicit_query(query)

# 2. Strategy Selection
if is_explicit:
    results = ExplicitSearcher.search(extracted_refs)
else:
    fts_results = FTSSearcher.search(query)
    vector_results = VectorSearcher.search(query) 
    results = combine_and_deduplicate(fts_results, vector_results)

# 3. Optional Reranking
if rerank_enabled:
    results = JinaReranker.rerank(query, results)
```

### Database Session Pattern
```python
# Context manager for transactions
with get_db_session() as db:
    # Database operations here
    result = db.query(Model).filter(...).all()
    # Auto-commit on success, rollback on exception
```

### Logging Pattern
```python
# Structured logging with timing
logger.info("Operation started", extra={"query": query, "limit": limit})
start_time = time.time()
# ... operation ...
duration_ms = (time.time() - start_time) * 1000
logger.info("Operation completed", extra=log_timing("operation", duration_ms))
```

## 🎛️ Configuration Management

### Settings Consolidation
- **Current**: Multiple config files (`ai_config.py`, `crawler_config.py`, etc.)
- **Target**: Single `settings.py` with Pydantic BaseSettings
- **Strategy**: Extract only used variables, ignore legacy configs

### Environment Handling
- **Development**: `.env` file loading
- **Production**: Environment variables
- **Testing**: Override settings for test isolation

## 📚 External Dependencies

### Required Services
- **PostgreSQL 16+** with pgvector extension
- **Jina AI API** for embeddings and reranking
- **LLM APIs** (Gemini 2.0, OpenAI GPT-4, Anthropic Claude)

### Optional Services
- **Neo4j** for graph relationships (future)
- **Redis** for caching (future)

## 🚀 Deployment Architecture (Planned)

### Backend Services
```
FastAPI App → Uvicorn → Nginx (Reverse Proxy)
│
├── PostgreSQL (Primary Data)
├── Redis (Caching)
└── External APIs (Jina, LLMs)
```

### Frontend
```
Next.js App Router → Vercel/Docker
│
└── Backend API (REST/WebSocket)
```

## 🎯 Success Metrics

### Performance Targets
- **Search Latency**: <500ms for hybrid search
- **Indexing Speed**: >100 documents/minute
- **Accuracy**: >90% for explicit references, >80% for thematic

### Quality Metrics
- **Code Coverage**: >95%
- **Module Size**: <300 LOC per file
- **API Compatibility**: 100% backward compatible

## 🔗 Key Integration Points

### Between Services
- **Indexer ↔ Embedder**: Batch embedding generation
- **Retriever ↔ Database**: Complex queries with filtering
- **Search ↔ Reranker**: Result score optimization
- **API ↔ Search**: JSON serialization and error handling

### With External Systems
- **Crawler Output**: JSON files in `data/json/`
- **PDF Processing**: Document tree extraction
- **LLM APIs**: Prompt management and response handling
- **Vector Database**: Embedding storage and similarity search

## 📝 Development Guidelines

### Adding New Features
1. **Read existing code** to understand patterns
2. **Follow established interfaces** (abstract base classes)
3. **Add comprehensive tests** before implementation
4. **Use dependency injection** for external services
5. **Document breaking changes** in migration notes

### Debugging Tips
1. **Use CLI commands** for isolated testing
2. **Enable structured logging** with DEBUG level
3. **Test components separately** before integration
4. **Check database state** with direct SQL queries

### Refactoring Rules
1. **Preserve external API compatibility**
2. **Extract configurations** from hardcoded values
3. **Eliminate code duplication** aggressively
4. **Split large files** into focused modules
5. **Add type hints** and error handling

## 🎪 Testing Strategy

### Unit Tests (Planned)
- **Models**: Database schema and relationships
- **Services**: Business logic with mocked dependencies
- **Utils**: Text processing and HTTP utilities

### Integration Tests (Planned)
- **Indexing**: End-to-end JSON→DB→Search pipeline
- **Search**: Multi-strategy search with real database
- **API**: HTTP endpoints with realistic payloads

### Performance Tests (Planned)
- **Search Latency**: Response time under load
- **Indexing Throughput**: Documents per minute
- **Memory Usage**: Large document processing

---

## 🎯 AI Agent Instructions

### When Working on This Project:

1. **ALWAYS** check existing code before implementing new features
2. **PRESERVE** external API compatibility during refactoring
3. **USE** the CLI (`python -m src.main`) for testing database/search operations
4. **FOLLOW** the established patterns for logging, error handling, and DI
5. **UPDATE** this AGENTS.md when making significant architectural changes

### Common Tasks:

#### Fixing Search Issues
```bash
# Test database connectivity
python -m src.main status

# Test indexing without embeddings
python -m src.main index data/json/sample.json --skip-embeddings

# Test FTS-only search
python -m src.main search "pertambangan" --no-rerank
```

#### Adding New Services
1. Create abstract base class in appropriate `services/` subdirectory
2. Implement concrete class with dependency injection
3. Add factory function for easy instantiation
4. Update settings.py with relevant configuration
5. Add CLI command for testing

#### Database Changes
1. Update models in `src/db/models.py`
2. Create Alembic migration: `alembic revision --autogenerate -m "description"`
3. Apply migration: `alembic upgrade head`
4. Update indexer logic if needed

Remember: This is a **production-ready system** - prioritize reliability, maintainability, and clear error handling over quick hacks.