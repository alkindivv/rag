 # TODO_NEXT.md - Legal RAG System Action Plan

## 🚨 CRITICAL BLOCKERS (Fix Immediately)

### 1. Fix Jina Embedding API Integration
**Priority: CRITICAL** | **Assignee: Backend Dev** | **ETA: 1-2 days**

**Problem**: Jina embeddings API returning 422 Unprocessable Entity errors
```bash
# Current error
Client error '422 Unprocessable Entity' for url 'https://api.jina.ai/v1/embeddings'
```

**Root Cause Analysis Needed**:
- [ ] Check Jina v4 API documentation for correct request format
- [ ] Validate request payload structure in `src/services/embedding/embedder.py`
- [ ] Test with minimal API call outside the application
- [ ] Verify API key permissions and rate limits

**Action Items**:
- [ ] **IMMEDIATE**: Test Jina API with curl/postman to isolate issue
- [ ] **FIX**: Update request format in `JinaEmbedder._embed_batch_internal()`
- [ ] **VALIDATE**: Test with single embedding first, then batch
- [ ] **FALLBACK**: Implement mock embedder for development if API issues persist

**Files to Modify**:
- `src/services/embedding/embedder.py` (lines 80-95)
- Add integration test in `tests/test_embedding.py`

---

### 2. Fix SQL Query Formatting in Hybrid Retriever  
**Priority: CRITICAL** | **Assignee: Backend Dev** | **ETA: 4-6 hours**

**Problem**: SQLAlchemy `text()` object doesn't support `.format()` method
```python
# Error location: src/services/retriever/hybrid_retriever.py
final_query = fts_query.format(filters_clause=filters_clause)  # FAILS
```

**Root Cause**: Using string formatting on SQLAlchemy `text()` objects

**Action Items**:
- [x] **FIX**: Replace `.format()` with proper SQLAlchemy parameter binding
- [x] **REFACTOR**: Use `text()` with bound parameters instead of string interpolation
- [x] **TEST**: Verify all query paths (FTS, Vector, Explicit) work correctly
- [x] **VALIDATE**: Test with and without filters

**Code Fix Example**:
```python
# BEFORE (broken)
fts_query = text("SELECT ... {filters_clause}")
final_query = fts_query.format(filters_clause=filters_clause)

# AFTER (working)
if filters:
    fts_query = text("SELECT ... AND " + " AND ".join(filter_conditions))
else:
    fts_query = text("SELECT ... ")
```

**Files to Modify**:
- `src/services/retriever/hybrid_retriever.py` (lines 150-180, 220-250, 350-380)

---

### 3. Resolve JSON Data Structure Inconsistencies
**Priority: HIGH** | **Assignee: Data Engineer** | **ETA: 2-3 days**

**Problem**: Crawler output contains duplicate `unit_id` entries causing indexing failures

**Current Workaround**: Deduplication in indexer (implemented)
**Root Cause**: PDF parsing creates duplicate entries in document tree

**Analysis Results**:
```bash
# Found 20 duplicate unit_ids in undang_undang_2_2025.json:
UU-2025-2/angka-2: 2 times
UU-2025-2/angka-3: 2 times
# ... and 18 more
```

**Action Items**:
- [ ] **ANALYZE**: Deep-dive into crawler/PDF parser logic to identify duplicate source
- [ ] **INVESTIGATE**: Check if duplicates have different content or just structural duplicates
- [ ] **FIX**: Update crawler to prevent duplicates at source
- [ ] **VALIDATE**: Re-crawl sample documents to verify fix
- [ ] **CLEANUP**: Remove deduplication logic from indexer once source is fixed

**Files to Investigate**:
- `src/services/crawler/*` - crawler logic
- `src/services/pdf/*` - PDF parsing logic
- `data/json/*.json` - check other files for similar issues

---

## 🔧 HIGH PRIORITY FIXES (Next Sprint)

### 4. Complete Search System Integration
**Priority: HIGH** | **Assignee: Backend Dev** | **ETA: 3-4 days**

**Current Status**: Components exist but integration fails

**Action Items**:
- [ ] **FIX**: SQL query formatting issues (see item #2)
- [ ] **IMPLEMENT**: FTS-only search mode for testing without vector search
- [ ] **TEST**: Each search strategy independently (Explicit, FTS, Vector)
- [ ] **INTEGRATE**: Combine strategies with proper error handling
- [ ] **VALIDATE**: End-to-end search pipeline works

**Test Plan**:
```bash
# Test sequence
python -m src.main search "pasal 1" --strategy explicit
python -m src.main search "pertambangan" --strategy fts
python -m src.main search "mineral" --strategy vector  # after embedding fix
python -m src.main search "batubara" --strategy hybrid # all combined
```

---

### 5. Implement FastAPI REST API
**Priority: HIGH** | **Assignee: Backend Dev** | **ETA: 2-3 days**

**Status**: Skeleton exists but needs implementation

**Requirements**:
- Health check endpoint
- Search endpoint with filters
- Document outline endpoint
- Error handling and validation
- API documentation (OpenAPI/Swagger)

**Action Items**:
- [ ] **CREATE**: `src/api/main.py` as FastAPI application entry point
- [ ] **IMPLEMENT**: Core endpoints using existing services
- [ ] **ADD**: Request/response models with Pydantic
- [ ] **CONFIGURE**: CORS, middleware, logging
- [ ] **DOCUMENT**: API specs with examples
- [ ] **TEST**: Integration tests for all endpoints

**API Endpoints to Implement**:
```python
GET  /health                           # System health
GET  /search?q={query}&limit={n}      # Search documents
POST /search/advanced                  # Advanced search with filters  
GET  /documents/{doc_id}/outline      # Document structure
GET  /documents/{doc_id}/units        # Document units
```

---

### 6. Add Comprehensive Testing Suite
**Priority: HIGH** | **Assignee: QA/Backend Dev** | **ETA: 4-5 days**

**Current Status**: No automated tests exist

**Testing Strategy**:
- Unit tests for individual components
- Integration tests for service interactions
- End-to-end tests for full pipeline
- Performance tests for search operations

**Action Items**:
- [ ] **SETUP**: pytest configuration and test structure
- [ ] **UNIT**: Test individual services (embedder, retriever, indexer)
- [ ] **INTEGRATION**: Test database operations and API endpoints
- [ ] **E2E**: Test full indexing and search pipeline
- [ ] **PERFORMANCE**: Search latency and throughput benchmarks
- [ ] **CI/CD**: GitHub Actions or similar for automated testing

**Test Structure**:
```
tests/
├── conftest.py                    # Pytest fixtures
├── unit/
│   ├── test_embedder.py          # Embedding service tests
│   ├── test_indexer.py           # Indexing logic tests
│   ├── test_retriever.py         # Search logic tests
│   └── test_models.py            # Database model tests
├── integration/
│   ├── test_api.py               # FastAPI endpoint tests
│   ├── test_search_pipeline.py  # End-to-end search tests
│   └── test_database.py         # Database integration tests
└── performance/
    └── test_search_performance.py
```

---

## 📈 MEDIUM PRIORITY FEATURES (Next Month)

### 7. LLM Integration for Q&A
**Priority: MEDIUM** | **Assignee: ML Engineer** | **ETA: 5-7 days**

**Requirements**:
- Support multiple LLM providers (Gemini, OpenAI, Anthropic)
- Prompt templates for legal Q&A
- Citation extraction and formatting
- Context window management

**Action Items**:
- [ ] **IMPLEMENT**: Base LLM interface and provider factory
- [ ] **ADD**: Provider-specific implementations
- [ ] **CREATE**: Prompt templates for legal domain
- [ ] **INTEGRATE**: With search results for RAG pipeline
- [ ] **TEST**: Q&A accuracy with legal queries

---

### 8. Database Optimization and Monitoring
**Priority: MEDIUM** | **Assignee: Database Engineer** | **ETA: 3-4 days**

**Focus Areas**:
- Query performance optimization
- Index tuning for search operations
- Database monitoring and alerting
- Connection pooling optimization

**Action Items**:
- [ ] **ANALYZE**: Slow query logs and execution plans
- [ ] **OPTIMIZE**: Critical search queries
- [ ] **MONITOR**: Database performance metrics
- [ ] **SCALE**: Connection pool and resource settings

---

### 9. Enhanced CLI and Developer Tools
**Priority: MEDIUM** | **Assignee: DevOps** | **ETA: 2-3 days**

**Current Status**: Basic CLI exists but needs enhancements

**Action Items**:
- [ ] **ADD**: Bulk indexing with progress bars
- [ ] **IMPLEMENT**: Data validation and consistency checks
- [ ] **CREATE**: Database migration and backup tools
- [ ] **ENHANCE**: Debug and troubleshooting commands

---

## 🚀 LOW PRIORITY ENHANCEMENTS (Future Sprints)

### 10. Frontend Application
**Priority: LOW** | **Assignee: Frontend Dev** | **ETA: 1-2 weeks**

**Tech Stack**: Next.js 14 with App Router

**Action Items**:
- [ ] **SETUP**: Next.js project with TypeScript
- [ ] **IMPLEMENT**: Search interface
- [ ] **ADD**: Document viewer and citation display
- [ ] **INTEGRATE**: With FastAPI backend
- [ ] **DEPLOY**: Production-ready build

---

### 11. Advanced Features
**Priority: LOW** | **Assignee: TBD** | **ETA: TBD**

**Potential Features**:
- [ ] **GRAPH**: Neo4j integration for document relationships
- [ ] **CACHE**: Redis caching layer for frequent queries
- [ ] **ANALYTICS**: Search analytics and user behavior tracking
- [ ] **EXPORT**: Document and search result export features
- [ ] **ADMIN**: Administrative interface for content management

---

### 12. Production Deployment
**Priority: LOW** | **Assignee: DevOps** | **ETA: 1 week**

**Requirements**:
- Docker containerization
- Load balancing and scaling
- Monitoring and alerting
- Backup and disaster recovery

**Action Items**:
- [ ] **CONTAINERIZE**: Create Docker images for all services
- [ ] **ORCHESTRATE**: Kubernetes or Docker Compose setup
- [ ] **MONITOR**: Prometheus/Grafana monitoring stack
- [ ] **SECURE**: Authentication, authorization, rate limiting

---

## 📋 TECHNICAL DEBT & CODE QUALITY

### 13. Code Refactoring and Cleanup
**Priority: ONGOING** | **Assignee: All Developers**

**Areas Needing Attention**:
- [ ] **CONSOLIDATE**: Multiple config files into single settings.py
- [ ] **STANDARDIZE**: Import paths and module structure
- [ ] **DOCUMENT**: Add docstrings and type hints consistently
- [ ] **OPTIMIZE**: Remove dead code and unused imports
- [ ] **SECURITY**: Add input validation and sanitization

---

### 14. Legacy Service Integration
**Priority: ONGOING** | **Assignee: Backend Dev**

**Current Legacy Services**:
- Crawler services (needs light refactoring)
- PDF processing services (needs modernization)
- Text cleaning utilities (needs modularization)

**Action Items**:
- [ ] **AUDIT**: Review legacy code for integration points
- [ ] **REFACTOR**: Update to match current architecture patterns
- [ ] **TEST**: Ensure backward compatibility
- [ ] **DOCUMENT**: Add proper documentation and examples

---

## 🔄 IMMEDIATE NEXT STEPS (This Week)

### Day 1-2: Fix Critical Blockers
1. **Morning**: Fix Jina API integration issue
2. **Afternoon**: Fix SQL query formatting in retriever
3. **Test**: Verify search pipeline works end-to-end

### Day 3-4: Complete Search System
1. **Morning**: Implement FTS-only search mode
2. **Afternoon**: Test all search strategies independently
3. **Evening**: Integrate strategies with error handling

### Day 5: Testing and Validation
1. **Morning**: Add basic unit tests for critical components
2. **Afternoon**: Integration testing of full pipeline
3. **Evening**: Performance testing and optimization

---

## 📊 SUCCESS CRITERIA

### Week 1 Goals:
- [ ] ✅ Search system works for at least FTS queries
- [ ] ✅ Indexing pipeline processes documents without errors
- [ ] ✅ Basic API endpoints are functional
- [ ] ✅ Core unit tests are passing

### Month 1 Goals:
- [ ] ✅ Full hybrid search (FTS + Vector) working
- [ ] ✅ LLM integration for Q&A complete
- [ ] ✅ Comprehensive test suite implemented
- [ ] ✅ API documentation and examples ready

### Month 3 Goals:
- [ ] ✅ Frontend application deployed
- [ ] ✅ Production deployment ready
- [ ] ✅ Performance benchmarks met
- [ ] ✅ Full feature parity with requirements

---

## 🎯 TEAM ASSIGNMENTS

| Priority | Task | Assignee | Status | Due Date |
|----------|------|----------|---------|----------|
| CRITICAL | Fix Jina API | Backend Dev | 🔄 In Progress | Tomorrow |
| CRITICAL | Fix SQL Queries | Backend Dev | ✅ Done | This Week |
| HIGH | Complete Search | Backend Dev | ⏳ Pending | Next Week |
| HIGH | FastAPI Implementation | Backend Dev | ⏳ Pending | Next Week |
| HIGH | Testing Suite | QA Engineer | ⏳ Pending | Next Sprint |
| MEDIUM | LLM Integration | ML Engineer | ⏳ Pending | Next Month |
| LOW | Frontend | Frontend Dev | ⏳ Pending | Future |

---

## 📝 NOTES AND CONSIDERATIONS

### Development Environment
- Ensure PostgreSQL with pgvector extension is available
- Set up proper environment variables (see `.env.example`)
- Use Python virtual environment for dependencies

### Testing Strategy
- Start with integration tests for critical path
- Mock external APIs (Jina, LLM providers) for reliable testing
- Use test database separate from development data

### Documentation
- Keep AGENTS.md updated with architectural changes
- Update TODO_NEXT.md as items are completed
- Add inline documentation for complex algorithms

### Risk Mitigation
- Have fallback strategies for external API failures
- Implement circuit breakers for unreliable services
- Plan for database performance under load

---

**Last Updated**: 2025-08-09  
**Next Review**: Weekly on Mondays  
**Status**: 🔄 Active Development