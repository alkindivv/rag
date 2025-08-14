# SUMMARY_TODO: Week 2 Hybrid Search Implementation

## 🚨 CURRENT PROBLEMS IDENTIFIED

### 1. **API Response Format Mismatch**
- **Problem**: Vector search returns `dict` with `{results: [], metadata: {}}` 
- **Problem**: BM25 search returns `list` directly
- **Impact**: Hybrid search fusion expects consistent `List[SearchResult]` format
- **Error**: `AttributeError: 'str' object has no attribute 'citation_string'`

### 2. **Service Integration Inconsistency**
- **Problem**: `VectorSearchService.search_async()` returns dict format
- **Problem**: `BM25SearchService.search_async()` returns list format  
- **Problem**: `HybridSearchService` expects unified list format from both
- **Impact**: RRF fusion fails due to format mismatch

### 3. **API Endpoint Compatibility**
- **Problem**: API expects `results["results"]` and `results["metadata"]`
- **Problem**: Hybrid service returns `List[SearchResult]` directly
- **Impact**: API response structure breaks

## ✅ IMMEDIATE ACTIONS REQUIRED

### Priority 1: **Fix Service Response Formats**

#### A. Standardize BM25SearchService Response
```python
# File: src/services/search/bm25_search.py
# Method: search_async() and search()
# CHANGE: Return dict format like vector search
# FROM: return List[SearchResult] 
# TO: return {"results": List[SearchResult], "metadata": dict}
```

#### B. Update HybridSearchService to Handle Both Formats
```python
# File: src/services/search/hybrid_search.py
# Method: _hybrid_search_async()
# ADD: Format normalization for both services
# ENSURE: Extract actual results list from dict format
```

#### C. Fix API Integration
```python
# File: src/api/main.py  
# Method: search_documents()
# CHANGE: Handle hybrid service response format
# ENSURE: Consistent API response structure
```

### Priority 2: **Verify Database Setup**

#### A. Run FTS Migration
```bash
# Command to execute:
alembic upgrade head
# Verify: bm25_tsvector column exists
# Verify: GIN index created
# Verify: Trigger function installed
```

#### B. Validate FTS Population
```sql
-- Check FTS coverage
SELECT 
    COUNT(*) as total,
    COUNT(CASE WHEN bm25_tsvector IS NOT NULL THEN 1 END) as indexed
FROM legal_units;
-- Target: >80% coverage
```

### Priority 3: **Integration Testing**

#### A. Test Individual Components
```python
# Test vector search (existing - should work)
vector_service = VectorSearchService()
vector_results = vector_service.search("ekonomi kreatif", k=3)

# Test BM25 search (new - needs format fix)  
bm25_service = BM25SearchService()
bm25_results = bm25_service.search("ekonomi kreatif", k=3)

# Test RRF fusion (new - needs format fix)
rrf_engine = RRFFusionEngine()
fused = rrf_engine.fuse_results(vector_list, bm25_list, max_results=5)
```

#### B. Test Hybrid Integration
```python
# Test hybrid service (main integration)
hybrid_service = HybridSearchService()
results = hybrid_service.search("ekonomi kreatif", k=5, strategy="hybrid")
# Expected: List[SearchResult] with RRF scores
```

## 🔧 COMPATIBILITY ALIGNMENT TASKS

### 1. **Response Format Standardization**
- **Vector Search**: Already returns `{"results": [], "metadata": {}}`  ✅
- **BM25 Search**: MUST return `{"results": [], "metadata": {}}` ❌
- **Hybrid Search**: MUST handle both input formats and return consistent output ❌
- **API Layer**: MUST handle hybrid service response format ❌

### 2. **SearchResult Object Consistency**
- **Vector**: Uses `SearchResult` class ✅
- **BM25**: MUST use same `SearchResult` class ❌  
- **RRF**: MUST work with `SearchResult` objects ❌
- **API**: MUST call `result.to_dict()` method ❌

### 3. **Metadata Preservation**
- **Vector**: Rich metadata with search_type, duration, etc. ✅
- **BM25**: MUST include search_type="bm25_fts" ❌
- **RRF**: MUST preserve and merge metadata ❌
- **Hybrid**: MUST aggregate metadata from all sources ❌

## 🧪 POST-FIX VERIFICATION CHECKLIST

### Database Level
- [ ] `bm25_tsvector` column exists in `legal_units` table
- [ ] GIN index `idx_legal_units_bm25_tsvector_gin` exists  
- [ ] Trigger `trigger_update_bm25_tsvector` is active
- [ ] FTS coverage ≥ 80% of legal units
- [ ] Query `SELECT to_tsvector('simple', 'test')` works

### Service Level  
- [ ] `BM25SearchService.search()` returns dict format
- [ ] `BM25SearchService.search_async()` returns dict format
- [ ] `HybridSearchService.search()` returns list format
- [ ] `RRFFusionEngine.fuse_results()` accepts list inputs
- [ ] All services handle empty results gracefully

### Integration Level
- [ ] API `/search` endpoint accepts `strategy` parameter
- [ ] API response has `results` and `metadata` keys
- [ ] Hybrid search works with strategy="hybrid"
- [ ] Comparative queries are detected and handled
- [ ] Performance: <2s response time for 95% queries

### Functional Testing
```python
# Test cases that MUST pass:
test_queries = [
    "ekonomi kreatif",  # Simple keyword
    "apa bedanya ekonomi kreatif dengan industri kreatif?",  # Comparative  
    "UU 24 tahun 2019 pasal 1",  # Citation
    "definisi pelaku ekonomi kreatif"  # Contextual
]

# Expected: All return results without errors
# Expected: Comparative queries show enhanced handling
# Expected: RRF scores in metadata
```

## 🎯 SUCCESS CRITERIA

### Technical
- [ ] Zero `AttributeError` in logs
- [ ] Consistent response formats across all services
- [ ] BM25 + Vector fusion produces better results than either alone
- [ ] API maintains backward compatibility
- [ ] Performance degradation <10% vs vector-only

### Functional  
- [ ] Comparative queries return relevant results for both concepts
- [ ] Keyword queries benefit from BM25 precision
- [ ] Semantic queries maintain vector search quality
- [ ] Hybrid strategy auto-selects appropriately

### Operational
- [ ] Setup script runs without errors
- [ ] Migration applies cleanly
- [ ] Validation passes all checks
- [ ] Monitoring captures hybrid search metrics

## 📋 EXECUTION ORDER

1. **Fix BM25SearchService response format** (30 min)
2. **Update HybridSearchService input handling** (20 min)  
3. **Fix API endpoint integration** (15 min)
4. **Run database migration** (5 min)
5. **Execute setup script with validation** (10 min)
6. **Run integration tests** (20 min)
7. **Performance benchmark** (10 min)

**Total Estimated Time**: 2 hours

## 🤝 HANDOFF NOTES FOR NEXT AI AGENT

- **Core Issue**: Service response format inconsistency between vector (dict) and BM25 (list)
- **Main Files to Edit**: `bm25_search.py`, `hybrid_search.py`, `api/main.py`
- **Test Command**: `python scripts/setup_hybrid_search.py --validate-only`
- **Success Indicator**: No AttributeError and all test queries return results
- **Context**: This is Week 2 of KISS hybrid implementation, not a major architecture change