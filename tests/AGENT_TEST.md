# Legal RAG System - Comprehensive Testing Guide

## 🎯 Overview

This testing suite provides comprehensive validation for the Legal RAG System, focusing on critical issues identified in the project roadmap and ensuring production readiness.

## 🚨 Critical Issues Addressed

### Immediate Blockers (from TODO_NEXT.md)
1. **Jina API Integration (422 Errors)** - Tests validate correct v4 API format
2. **SQL Query Formatting** - Tests ensure SQLAlchemy text() objects work properly
3. **Import Path Consistency** - Tests verify all modules import without errors
4. **JSON Data Duplicates** - Tests validate deduplication logic

## 📁 Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and test configuration
├── run_tests.py               # Comprehensive test runner script
├── README.md                  # This documentation
├── unit/                      # Fast isolated component tests
│   ├── test_embedding_service.py     # JinaEmbedder validation
│   └── test_hybrid_retriever.py      # Search logic validation
├── integration/               # Component interaction tests
│   └── test_search_pipeline.py       # Multi-component workflows
├── e2e/                      # Complete system workflows
│   └── test_complete_workflow.py     # End-to-end scenarios
└── data/                     # Test data files (auto-created)
    ├── fixtures/             # Reusable test data
    └── output/              # Test output files
```

## 🏃‍♂️ Quick Start

### Run All Tests
```bash
# From project root
python tests/run_tests.py --all
```

### Critical Issues Only
```bash
# Test fixes for critical blockers
python tests/run_tests.py --critical
```

### Quick Smoke Test
```bash
# Fast validation (< 30 seconds)
python tests/run_tests.py --quick
```

### Performance Benchmarks
```bash
# Validate performance requirements
python tests/run_tests.py --performance
```

## 📋 Test Categories

### 🔧 Unit Tests (`tests/unit/`)
**Purpose**: Fast, isolated component validation  
**Runtime**: < 60 seconds  
**Dependencies**: Minimal (mocked external services)

```bash
# Run unit tests only
python tests/run_tests.py --unit

# Or directly with pytest
python -m pytest tests/unit/ -v
```

**Coverage**:
- JinaEmbedder API integration and error handling
- HybridRetriever search strategies and SQL queries
- SearchResult data structures and validation
- Configuration management and environment handling

### 🔗 Integration Tests (`tests/integration/`)
**Purpose**: Component interaction validation  
**Runtime**: < 3 minutes  
**Dependencies**: Database, mocked APIs

```bash
# Run integration tests
python tests/run_tests.py --integration

# With database setup
python -m pytest tests/integration/ -v --tb=short
```

**Coverage**:
- Database operations and query execution
- Search pipeline component interactions
- Error propagation and recovery
- Multi-document search scenarios
- Filtering and pagination logic

### 🌍 End-to-End Tests (`tests/e2e/`)
**Purpose**: Complete workflow validation  
**Runtime**: < 10 minutes  
**Dependencies**: Full system stack

```bash
# Run end-to-end tests
python tests/run_tests.py --e2e

# With performance monitoring
python -m pytest tests/e2e/ -v --tb=line
```

**Coverage**:
- JSON ingestion → Database → Search → Results workflows
- Multi-strategy search orchestration
- Real-world legal research scenarios
- System performance under load
- Graceful degradation testing

## 🎯 Performance Requirements

Based on AGENTS.md specifications:

| Metric | Requirement | Test Validation |
|--------|-------------|-----------------|
| Search Latency | < 500ms | `test_search_latency_benchmark` |
| Indexing Rate | > 100 docs/min | `test_indexing_rate_benchmark` |
| Memory Usage | < 512MB | `test_memory_usage_large_dataset` |
| Concurrent Users | 5+ simultaneous | `test_concurrent_performance_benchmark` |

```bash
# Run performance validation
python tests/run_tests.py --performance --verbose
```

## 🧪 Test Fixtures and Utilities

### Key Fixtures (from `conftest.py`)

#### Database Fixtures
- `db_session`: Clean database session per test
- `populated_db`: Database with sample legal documents
- `test_engine`: In-memory SQLite for fast testing

#### Mock Services
- `mock_jina_embedder`: Mock embedding service (1024-dim vectors)
- `mock_jina_reranker`: Mock reranking service
- `mock_http_client`: Mock HTTP client for API calls

#### Test Data
- `sample_legal_document`: Realistic legal document structure
- `sample_legal_units`: Hierarchical document units
- `sample_json_document`: Complete JSON from crawler output

#### Utilities
- `performance_timer`: Measure test execution time
- `test_helper`: Validation and assertion utilities
- `capture_logs`: Capture logging output for verification

### Usage Examples

```python
def test_my_feature(populated_db, mock_jina_embedder, performance_timer):
    """Example test using fixtures."""
    # Database is pre-populated with test data
    # Jina embedder is mocked to return consistent results
    # Performance timer measures execution time
    
    performance_timer.start()
    # Your test logic here
    performance_timer.stop()
    
    assert performance_timer.duration_ms < 500
```

## 🐛 Testing Critical Bug Fixes

### 1. Jina API 422 Error Fix
```python
# Test validates correct Jina v4 API format
def test_jina_api_422_error_fix():
    """Ensures 422 errors are resolved with proper request format."""
    # Validates:
    # - Correct Content-Type headers
    # - Proper JSON payload structure
    # - encoding_format: "float" parameter
    # - No invalid "dimensions" parameter
```

### 2. SQL Query Format Fix
```python
# Test validates SQLAlchemy text() parameter binding
def test_sql_format_method_fix():
    """Ensures SQL queries use parameter binding, not .format()."""
    # Validates:
    # - TextClause objects used properly
    # - No string formatting on SQL objects
    # - Proper filter clause construction
    # - Column references are correct (lu.unit_type vs dv.content_type)
```

### 3. Import Path Consistency Fix
```python
# Test validates all imports work correctly
def test_import_path_consistency():
    """Ensures all components can be imported without errors."""
    # Validates:
    # - Absolute imports from project root
    # - Lazy loading patterns for circular imports
    # - Module availability across different contexts
```

## 📊 Test Execution Strategies

### Development Workflow
```bash
# 1. Quick validation during development
python tests/run_tests.py --quick

# 2. Unit tests for specific component
python -m pytest tests/unit/test_embedding_service.py -v

# 3. Integration test for workflow
python -m pytest tests/integration/test_search_pipeline.py::TestSearchPipelineEndToEnd -v

# 4. Full validation before commit
python tests/run_tests.py --all --coverage
```

### CI/CD Pipeline
```bash
# Stage 1: Fast validation
python tests/run_tests.py --quick --output quick_report.json

# Stage 2: Comprehensive testing
python tests/run_tests.py --all --coverage --output full_report.json

# Stage 3: Performance validation
python tests/run_tests.py --performance --output perf_report.json
```

### Production Validation
```bash
# Validate system ready for deployment
python tests/run_tests.py --critical --performance --output prod_validation.json
```

## 🔍 Test Data and Fixtures

### Sample Legal Document Structure
The test suite uses realistic Indonesian legal document structures:

```json
{
  "doc_id": "UU-2025-1",
  "doc_form": "UU",
  "doc_number": "1",
  "doc_year": "2025",
  "doc_title": "Test Mining Law",
  "doc_subject": ["PERTAMBANGAN", "MINERAL"],
  "document_tree": {
    "doc_type": "document",
    "children": [
      {
        "type": "pasal",
        "unit_id": "UU-2025-1/pasal-1",
        "number_label": "1",
        "local_content": "Dalam Undang-Undang ini yang dimaksud dengan:",
        "citation_string": "Pasal 1",
        "children": [
          {
            "type": "ayat",
            "unit_id": "UU-2025-1/pasal-1/ayat-1",
            "local_content": "pertambangan adalah...",
            "citation_string": "Pasal 1 ayat (1)",
            "parent_pasal_id": "UU-2025-1/pasal-1"
          }
        ]
      }
    ]
  }
}
```

### Search Query Test Cases
```python
# Explicit references
"pasal 1", "pasal 1 ayat 2", "UU 4/2009"

# Thematic searches  
"pertambangan mineral", "izin usaha", "lingkungan hidup"

# Complex queries
"bagaimana cara mengajukan izin pertambangan?"
```

## 🚀 Performance Testing

### Search Latency Validation
```python
@pytest.mark.performance
def test_search_meets_latency_requirements():
    """Validates < 500ms search latency requirement."""
    # Tests all search strategies under realistic load
    # Measures end-to-end response time
    # Validates against AGENTS.md performance targets
```

### Memory Usage Testing
```python
@pytest.mark.performance  
def test_memory_efficiency():
    """Validates memory usage under large datasets."""
    # Tests with 1000+ document units
    # Monitors memory growth during operations
    # Validates < 512MB memory limit
```

### Concurrent Load Testing
```python
@pytest.mark.performance
def test_concurrent_search_load():
    """Validates system under concurrent access."""
    # Simulates 5+ concurrent users
    # Tests thread safety and resource contention
    # Validates performance doesn't degrade
```

## 🔧 Troubleshooting

### Common Test Failures

#### 1. Database Connection Issues
```bash
# Error: Database connection failed
# Solution: Ensure PostgreSQL is running or use SQLite
export DATABASE_URL="sqlite:///:memory:"
python tests/run_tests.py --unit
```

#### 2. Missing Dependencies
```bash
# Error: ModuleNotFoundError
# Solution: Install test dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock
```

#### 3. Jina API Authentication
```bash
# Error: API key issues in integration tests
# Solution: Use test API key or skip external tests
export JINA_API_KEY="test-key"
python -m pytest -k "not external" tests/
```

#### 4. Import Path Issues
```bash
# Error: Cannot import src modules
# Solution: Run from project root
cd /path/to/Rag
python tests/run_tests.py --quick
```

### Debugging Test Failures

#### Enable Debug Logging
```bash
# See detailed test execution
LOG_LEVEL=DEBUG python tests/run_tests.py --unit --verbose
```

#### Run Specific Failing Test
```bash
# Focus on specific failure
python -m pytest tests/unit/test_embedding_service.py::TestJinaEmbedderAPIPayload::test_single_embedding_payload_format -v -s
```

#### Generate Coverage Report
```bash
# Identify untested code paths
python tests/run_tests.py --coverage
# Open tests/coverage_html/index.html in browser
```

## 📈 Test Metrics and Reporting

### Success Criteria
- **95%+ Test Coverage**: All critical paths tested
- **100% Critical Tests Pass**: No failures in critical blocker tests
- **<500ms Search Latency**: Performance requirements met
- **Zero Import Errors**: All modules load correctly

### Report Formats

#### Console Output
```
🧪 LEGAL RAG SYSTEM - TEST EXECUTION REPORT
================================================================================

📊 SUMMARY:
   Total Tests: 47
   ✅ Passed: 44
   ❌ Failed: 2
   ⏭️  Skipped: 1
   🚨 Errors: 0
   📈 Success Rate: 93.6%
   ⏱️  Total Duration: 2847ms

🟡 OVERALL STATUS: GOOD

📂 BY CATEGORY:
   ✅ Unit Tests: 15/15 passed (100.0%)
   ✅ Integration: 12/13 passed (92.3%)
   ⚠️  End-to-End: 17/19 passed (89.5%)
```

#### JSON Report (`--output report.json`)
```json
{
  "summary": {
    "total_tests": 47,
    "passed": 44,
    "failed": 2,
    "success_rate": 0.936,
    "total_duration_ms": 2847
  },
  "critical_failures": 0,
  "performance_issues": 1,
  "details": {
    "critical_failures": [],
    "performance_issues": [
      {
        "name": "Search Latency Test",
        "duration": 650,
        "message": "Exceeded 500ms requirement"
      }
    ]
  }
}
```

## 🎪 Test Scenarios

### Real-World Legal Research
```python
# Simulates lawyer researching mining permits
research_workflow = [
    ("pertambangan mineral", "fts"),        # General topic search
    ("izin usaha pertambangan", "fts"),     # Specific requirement
    ("UU 4/2009", "explicit"),             # Specific law lookup  
    ("pasal 36", "explicit"),              # Specific article
    ("pasal 1", "explicit")                # Definition lookup
]
```

### Multi-Document Analysis
```python
# Tests cross-document search and filtering
scenarios = [
    ("UU vs PP comparison", doc_forms=["UU", "PP"]),
    ("Year-based analysis", doc_years=[2009, 2025]),
    ("Unit type focus", unit_types=["ayat", "pasal"])
]
```

### Error Recovery Workflows
```python
# Tests graceful degradation
degradation_scenarios = [
    "vector_search_fails_fallback_to_fts",
    "api_timeout_retry_logic",
    "database_connection_recovery",
    "partial_component_failure"
]
```

## ⚡ Running Tests

### Standard Usage

#### Quick Development Check
```bash
# Fast feedback during development (< 30s)
python tests/run_tests.py --quick
```

#### Component-Specific Testing
```bash
# Test specific component
python -m pytest tests/unit/test_embedding_service.py -v

# Test specific functionality
python -m pytest -k "jina_api" tests/ -v

# Test with coverage
python -m pytest tests/unit/ --cov=src.services.embedding --cov-report=term-missing
```

#### Pre-Commit Validation
```bash
# Full validation before code commit
python tests/run_tests.py --all --coverage
```

### Advanced Usage

#### Performance Profiling
```bash
# Detailed performance analysis
python -m pytest tests/e2e/ -k "performance" --durations=10 -v
```

#### Parallel Execution
```bash
# Speed up test execution
python -m pytest tests/ -n auto --dist worksteal
```

#### Debug Mode
```bash
# Debug failing tests
python -m pytest tests/unit/test_embedding_service.py::test_api_error_handling_422 -v -s --pdb
```

### Environment-Specific Testing

#### Local Development
```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/test_db"
export JINA_API_KEY="your-dev-key"
python tests/run_tests.py --integration
```

#### CI/CD Pipeline
```bash
# Automated testing environment
export DATABASE_URL="sqlite:///:memory:"
export JINA_API_KEY="test-key"
python tests/run_tests.py --all --output ci_report.json
```

#### Production Validation
```bash
# Validate against production-like environment
export DATABASE_URL="your-staging-db-url"
export JINA_API_KEY="your-staging-key"
python tests/run_tests.py --critical --performance
```

## 🔍 Test Debugging

### Common Issues and Solutions

#### Test Environment Setup
```bash
# Issue: Import errors
# Solution: Ensure proper PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python tests/run_tests.py --quick

# Issue: Database connection
# Solution: Use in-memory database for testing
export DATABASE_URL="sqlite:///:memory:"
```

#### Specific Test Failures

#### Embedding Tests Failing
```bash
# Debug Jina API integration
python -m pytest tests/unit/test_embedding_service.py::TestJinaEmbedderAPIPayload -v -s

# Check API format
python -c "
from tests.conftest import mock_jina_api_success
# Examine mock response format
"
```

#### Search Tests Failing
```bash
# Debug SQL query issues
python -m pytest tests/unit/test_hybrid_retriever.py::TestHybridRetrieverRegressionTests -v -s

# Check query construction
LOG_LEVEL=DEBUG python -m pytest tests/integration/ -k "sql_format" -v
```

#### Performance Tests Failing
```bash
# Debug performance issues
python -m pytest tests/e2e/ -k "performance" --durations=0 -v

# Profile memory usage
python -m pytest tests/integration/ -k "memory_usage" --profile-svg
```

## 📊 Coverage and Quality Metrics

### Coverage Targets
- **Overall Coverage**: ≥ 95%
- **Critical Components**: 100%
  - `src/services/embedding/embedder.py`
  - `src/services/retriever/hybrid_retriever.py`
  - `src/pipeline/indexer.py`
  - `src/services/search/hybrid_search.py`

### Quality Gates
```bash
# Run with quality validation
python tests/run_tests.py --all --coverage

# Must pass:
# ✅ 95%+ test coverage
# ✅ 0 critical failures  
# ✅ <500ms search latency
# ✅ All imports working
```

### Coverage Analysis
```bash
# Generate detailed coverage report
python -m pytest --cov=src --cov-report=html:tests/coverage_html tests/

# View uncovered lines
python -m pytest --cov=src --cov-report=term-missing tests/

# Focus on specific module
python -m pytest --cov=src.services.embedding --cov-report=term tests/unit/test_embedding_service.py
```

## 🛠️ Test Development Guidelines

### Writing New Tests

#### Test Naming Convention
```python
# Pattern: test_{component}_{scenario}_{expected_outcome}
def test_embedder_api_error_422_returns_proper_exception():
def test_retriever_fts_strategy_returns_ranked_results():
def test_indexer_duplicate_units_handled_gracefully():
```

#### Test Structure Pattern
```python
def test_feature_scenario():
    """
    Test description explaining:
    - What is being tested
    - Why it's important
    - Expected behavior
    """
    # ARRANGE: Setup test data and mocks
    
    # ACT: Execute the functionality
    
    # ASSERT: Validate results and behavior
    
    # Additional validations for edge cases
```

#### Mock Usage Guidelines
```python
# Prefer dependency injection for testability
def test_with_dependency_injection(mock_jina_embedder):
    retriever = HybridRetriever(embedder=mock_jina_embedder)
    # Test with injected mock

# Use realistic mock responses
mock_response = {
    "model": "jina-embeddings-v4",
    "data": [{"embedding": [0.1] * 1024, "index": 0}]
}
```

### Test Data Management

#### Creating Test Documents
```python
# Use sample_json_document fixture for consistency
def test_with_standard_data(sample_json_document):
    # Modify as needed for specific test
    test_doc = sample_json_document.copy()
    test_doc["doc_id"] = "CUSTOM-TEST-ID"
```

#### Performance Test Data
```python
# Create large datasets for performance testing
large_units = [
    create_legal_unit(f"PERF-2025-{i}", content=f"Performance content {i}")
    for i in range(1000)
]
```

## 🚀 CI/CD Integration

### GitHub Actions Example
```yaml
name: Legal RAG Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Quick validation
        run: python tests/run_tests.py --quick
      
      - name: Full test suite
        run: python tests/run_tests.py --all --coverage
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Docker Testing
```dockerfile
# Test container
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "tests/run_tests.py", "--all"]
```

## 📋 Test Maintenance

### Regular Maintenance Tasks

#### Weekly
- [ ] Run full test suite: `python tests/run_tests.py --all`
- [ ] Check performance metrics: `python tests/run_tests.py --performance`
- [ ] Update test data if needed
- [ ] Review failed test reports

#### Monthly  
- [ ] Analyze coverage gaps: `python tests/run_tests.py --coverage`
- [ ] Update test documentation
- [ ] Review and clean up test data
- [ ] Performance benchmark review

#### Per Release
- [ ] Critical blocker validation: `python tests/run_tests.py --critical`
- [ ] Full system validation: `python tests/run_tests.py --all --performance`
- [ ] Production readiness check
- [ ] Update test requirements if needed

### Adding Tests for New Features

1. **Create unit tests first** for new components
2. **Add integration tests** for component interactions  
3. **Include end-to-end tests** for user workflows
4. **Update fixtures** if new data structures added
5. **Document test scenarios** in relevant test files

### Updating Existing Tests

1. **Preserve test intent** when refactoring
2. **Update mocks** to match real API changes
3. **Maintain performance benchmarks** as system evolves
4. **Keep test data realistic** and representative

## 🎯 Success Criteria

### Test Suite Health
- ✅ **95%+ Pass Rate**: Consistent test execution
- ✅ **<10 minute runtime**: Full suite execution time
- ✅ **Zero critical failures**: All critical paths working
- ✅ **Performance requirements met**: Latency and throughput targets

### Code Quality
- ✅ **95%+ coverage**: Comprehensive test coverage
- ✅ **Clear test intent**: Well-documented test purposes
- ✅ **Maintainable tests**: Easy to update and extend
- ✅ **Realistic scenarios**: Tests mirror production usage

## 📞 Support and Troubleshooting

### Getting Help
1. **Check test output**: Look for specific error messages
2. **Review test logs**: Enable DEBUG logging for details
3. **Run isolated tests**: Narrow down to specific failing component
4. **Check environment**: Verify all required variables are set

### Common Commands for Troubleshooting
```bash
# Debug specific test
python -m pytest tests/unit/test_embedding_service.py::test_failing_function -v -s --tb=long

# Check test environment
python -c "
import os
print('DATABASE_URL:', os.getenv('DATABASE_URL'))
print('JINA_API_KEY:', os.getenv('JINA_API_KEY'))
"

# Validate imports
python -c "
try:
    from src.services.embedding.embedder import JinaEmbedder
    print('✅ Embedder import OK')
except Exception as e:
    print('❌ Embedder import failed:', e)
"

# Test database connection
python -c "
from src.db.session import get_db_session
try:
    with get_db_session() as db:
        print('✅ Database connection OK')
except Exception as e:
    print('❌ Database connection failed:', e)
"
```

---

## 📚 Additional Resources

- **AGENTS.md**: System architecture and requirements
- **TODO_NEXT.md**: Current issues and priorities  
- **src/config/settings.py**: Configuration management
- **requirements.txt**: Dependencies and versions

For questions about specific test failures or adding new tests, refer to the test files themselves which contain detailed docstrings explaining the test scenarios and expected behavior.

---

**Last Updated**: 2025-01-15  
**Test Suite Version**: 1.0  
**Compatibility**: Python 3.11+, PostgreSQL 16+, SQLite 3.35+