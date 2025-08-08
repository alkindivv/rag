# 🐘 PGVector Setup Guide
**Simple PostgreSQL + pgvector implementation for legal document vectors**

---

## 🎯 Overview

This implementation provides a **drop-in replacement** for Qdrant using PostgreSQL with the pgvector extension. It maintains the exact same interface as `QdrantService` while leveraging your existing PostgreSQL infrastructure.

### ✅ Benefits

- **Same Interface**: Drop-in replacement for QdrantService
- **Unified Database**: Vectors + metadata in one PostgreSQL instance
- **Better Integration**: Uses existing database models and relationships
- **Cost Effective**: No separate vector database infrastructure
- **ACID Compliance**: Full transaction support
- **Mature Ecosystem**: PostgreSQL's proven reliability and tooling

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Legal Document System                    │
├─────────────────────────────────────────────────────────────┤
│  📄 Document Processing → 🧠 Embedding → 🗄️ Storage         │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   QdrantService │ ←→ │ PGVectorService │                │
│  │   (original)    │    │ (replacement)   │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Qdrant DB     │    │  PostgreSQL +   │                │
│  │   (external)    │    │   pgvector      │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Add to requirements.txt
pip install sqlalchemy==2.0.23 psycopg2-binary==2.9.9 pgvector==0.2.4 asyncpg==0.29.0
```

### 2. Setup PostgreSQL with pgvector

```bash
# Run setup script
python scripts/setup_pgvector_db.py

# Or manual setup:
sudo apt-get install postgresql-15-pgvector  # Ubuntu/Debian
brew install pgvector                        # macOS
```

### 3. Configure Environment

```bash
# .env file
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=legal_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_SSL_MODE=prefer
```

### 4. Test the Implementation

```bash
# Run test suite
python test_pgvector_service.py

# Expected output:
# ✅ Health check passed
# ✅ Vectors stored successfully
# ✅ Search working correctly
# 🎉 All tests passed successfully!
```

---

## 📝 Usage Examples

### Basic Usage (Drop-in Replacement)

```python
# OLD: Qdrant service
from services.vector.qdrant_service import QdrantService
qdrant = QdrantService(host="localhost", port=6333)

# NEW: PGVector service (same interface!)
from services.vector.pgvector_service import PGVectorService
pgvector = PGVectorService(connection_string="postgresql://...")

# Same methods, same results:
pgvector.store_vectors(records)
results = pgvector.search_vectors(query_vector, limit=10, filters={'doc_type': 'UU'})
```

### Complete Example

```python
import os
from services.vector.pgvector_service import setup_pgvector_for_legal_docs

# Setup service
service = setup_pgvector_for_legal_docs()

# Store legal document chunks
legal_chunks = [
    {
        'id': 'uu-39-1999-pasal-1',
        'vector': [0.1, 0.2, ...],  # 768-dimensional Gemini embedding
        'content': 'Pasal 1 ayat (1) Setiap orang berhak atas pengakuan...',
        'citation': 'UU No. 39 Tahun 1999 › Pasal 1 › Ayat (1)',
        'keywords': ['hak asasi manusia', 'pengakuan', 'jaminan'],
        'doc_type': 'UU',
        'doc_title': 'Hak Asasi Manusia',
        'doc_year': '1999',
        'chunk_type': 'legal_chunk',
        'token_count': 25,
        'position': {'bab': 'I', 'pasal': '1'}
    }
]

# Store vectors
success = service.store_vectors(legal_chunks)
print(f"Storage successful: {success}")

# Search with semantic similarity
query_vector = get_embedding("hak asasi manusia")  # Your embedding function
results = service.search_vectors(
    query_vector=query_vector,
    limit=10,
    score_threshold=0.7,
    filters={
        'doc_type': 'UU',
        'doc_year': '1999'
    }
)

# Process results (same format as Qdrant)
for record in results.records:
    print(f"Score: {record.score:.3f}")
    print(f"Citation: {record.metadata['citation']}")
    print(f"Content: {record.metadata['content'][:100]}...")
```

### Integration with Existing Pipeline

```python
# No changes needed in your existing pipeline!
from services.vector.pgvector_service import (
    setup_pgvector_for_legal_docs,
    store_chunked_documents,
    search_legal_documents
)

# Setup (replaces Qdrant setup)
vector_service = setup_pgvector_for_legal_docs()

# Use in your processing pipeline (same interface)
chunked_docs = your_chunking_pipeline(pdf_content)
success = store_chunked_documents(vector_service, chunked_docs)

# Search (same interface)
results = search_legal_documents(
    vector_service,
    query_vector=query_embedding,
    doc_type='UU',
    citation_filter='Pasal 1',
    limit=10
)
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Database Connection
POSTGRES_HOST=localhost                    # PostgreSQL host
POSTGRES_PORT=5432                        # PostgreSQL port
POSTGRES_DB=legal_db                      # Database name
POSTGRES_USER=postgres                    # Username
POSTGRES_PASSWORD=your_password           # Password
POSTGRES_SSL_MODE=prefer                  # SSL mode

# Connection Pool
POSTGRES_POOL_SIZE=5                      # Connection pool size
POSTGRES_MAX_OVERFLOW=10                  # Max pool overflow
POSTGRES_POOL_TIMEOUT=30                  # Pool timeout (seconds)
POSTGRES_POOL_RECYCLE=1800               # Pool recycle time (seconds)

# Vector Settings
VECTOR_DIMENSION=768                      # Gemini embedding dimension
SIMILARITY_THRESHOLD=0.7                  # Default similarity threshold

# Performance Settings
HNSW_M=16                                # HNSW index M parameter
HNSW_EF_CONSTRUCTION=64                  # HNSW index ef_construction
SEARCH_TIMEOUT=30                        # Search timeout (seconds)
MAX_SEARCH_RESULTS=100                   # Max search results
```

### Advanced Configuration

```python
from config.pgvector_config import PGVectorConfig

# Custom configuration
config = PGVectorConfig(
    host="your-db-host",
    port=5432,
    database="legal_vectors",
    username="vector_user",
    password="secure_password",
    vector_dimension=768,
    hnsw_m=32,  # Higher M for better recall
    hnsw_ef_construction=128  # Higher ef for better index quality
)

# Use custom config
service = PGVectorService(connection_string=config.connection_string)
```

---

## 🔄 Migration from Qdrant

### Automated Migration

```bash
# Run migration script
python scripts/migrate_qdrant_to_pgvector.py

# With custom settings
QDRANT_HOST=your-qdrant-host \
QDRANT_PORT=6333 \
MIGRATION_BATCH_SIZE=50 \
CLEAR_TARGET=true \
python scripts/migrate_qdrant_to_pgvector.py
```

### Manual Migration Steps

1. **Setup PGVector Database**
   ```bash
   python scripts/setup_pgvector_db.py
   ```

2. **Export from Qdrant**
   ```python
   from services.vector.qdrant_service import QdrantService
   
   qdrant = QdrantService(host="your-qdrant-host")
   # Export logic here (see migration script)
   ```

3. **Import to PGVector**
   ```python
   from services.vector.pgvector_service import PGVectorService
   
   pgvector = PGVectorService()
   pgvector.store_vectors(exported_records)
   ```

4. **Verify Migration**
   ```bash
   python test_pgvector_service.py
   ```

### Code Changes Required

**Minimal changes needed!**

```python
# OLD
from services.vector.qdrant_service import QdrantService, setup_qdrant_for_legal_docs
vector_service = setup_qdrant_for_legal_docs()

# NEW (just change imports!)
from services.vector.pgvector_service import PGVectorService, setup_pgvector_for_legal_docs
vector_service = setup_pgvector_for_legal_docs()

# Everything else stays the same!
```

---

## 📊 Performance Comparison

| Metric | Qdrant | PGVector | Notes |
|--------|--------|----------|--------|
| **Storage** | Separate service | Integrated DB | Simpler architecture |
| **Backup** | Custom solution | Standard pg_dump | Easier backup/restore |
| **Indexing** | HNSW built-in | HNSW via extension | Similar performance |
| **Filtering** | Native support | SQL WHERE clauses | More flexible filtering |
| **Transactions** | Limited | Full ACID | Better consistency |
| **Monitoring** | Custom tools | Standard PG tools | Better observability |
| **Scaling** | Horizontal | Vertical + Read replicas | Different scaling strategies |

### Performance Optimization

```sql
-- Optimize HNSW index
CREATE INDEX CONCURRENTLY idx_embedding_hnsw 
ON content_embeddings USING hnsw (embedding vector_cosine_ops) 
WITH (m = 32, ef_construction = 128);

-- Optimize metadata queries
CREATE INDEX CONCURRENTLY idx_metadata_composite 
ON content_embeddings (content_type, (content_metadata->>'doc_type'));

-- Update statistics
ANALYZE content_embeddings;
```

---

## 🛠️ Database Schema

### Main Table: `content_embeddings`

```sql
CREATE TABLE content_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID NOT NULL,
    content_type VARCHAR(20) NOT NULL,
    
    -- Content and vector
    content_text TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    
    -- Hierarchy and relationships
    hierarchy_path VARCHAR(500) NOT NULL,
    parent_id UUID,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Metadata (compatible with Qdrant format)
    structure_flags JSONB,
    content_metadata JSONB DEFAULT '{}',
    
    -- Model information
    model_used VARCHAR(50) DEFAULT 'text-embedding-004',
    embedding_version VARCHAR(20) DEFAULT 'v1',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Indexes

```sql
-- Vector similarity search (HNSW)
CREATE INDEX idx_content_embedding_vector 
ON content_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Metadata filtering
CREATE INDEX idx_content_embedding_doc_type 
ON content_embeddings USING GIN ((content_metadata->>'doc_type'));

-- Hierarchy path search
CREATE INDEX idx_content_embedding_hierarchy 
ON content_embeddings USING GIN (hierarchy_path gin_trgm_ops);

-- Full-text search
CREATE INDEX idx_content_embedding_fulltext 
ON content_embeddings USING GIN (to_tsvector('indonesian', content_text));
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. pgvector Extension Not Found
```bash
# Install pgvector
sudo apt-get install postgresql-15-pgvector  # Ubuntu
brew install pgvector                        # macOS

# Enable in database
psql -d your_db -c "CREATE EXTENSION vector;"
```

#### 2. Connection Issues
```python
# Test connection
from config.pgvector_config import PGVectorConfig
config = PGVectorConfig()
print(f"Connection string: {config.connection_string}")

# Test with psql
psql "postgresql://user:pass@host:port/db"
```

#### 3. Performance Issues
```sql
-- Check index usage
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM content_embeddings 
ORDER BY embedding <-> '[0.1,0.2,...]'::vector 
LIMIT 10;

-- Update statistics
ANALYZE content_embeddings;

-- Reindex if needed
REINDEX INDEX CONCURRENTLY idx_content_embedding_vector;
```

#### 4. Memory Issues
```bash
# Tune PostgreSQL memory settings
# postgresql.conf
shared_buffers = 256MB
work_mem = 16MB
maintenance_work_mem = 256MB
effective_cache_size = 1GB
```

### Debug Mode

```python
# Enable SQL logging
from sqlalchemy import create_engine
engine = create_engine(connection_string, echo=True)

# Check service health
service = PGVectorService()
health = service.health_check()
info = service.get_collection_info()
print(f"Health: {health}, Info: {info}")
```

---

## 📚 Additional Resources

### SQL Queries for Analysis

```sql
-- Count vectors by document type
SELECT 
    content_metadata->>'doc_type' as doc_type,
    COUNT(*) as count
FROM content_embeddings 
GROUP BY content_metadata->>'doc_type';

-- Average similarity scores
SELECT AVG(embedding <-> '[0.1,0.2,...]'::vector) as avg_distance
FROM content_embeddings;

-- Find duplicate content
SELECT content_text, COUNT(*) 
FROM content_embeddings 
GROUP BY content_text 
HAVING COUNT(*) > 1;
```

### Backup and Restore

```bash
# Backup vectors
pg_dump -t content_embeddings your_db > vectors_backup.sql

# Restore vectors
psql your_db < vectors_backup.sql
```

### Monitoring

```sql
-- Table size
SELECT pg_size_pretty(pg_total_relation_size('content_embeddings'));

-- Index size
SELECT pg_size_pretty(pg_relation_size('idx_content_embedding_vector'));

-- Query performance
SELECT query, calls, mean_exec_time 
FROM pg_stat_statements 
WHERE query LIKE '%content_embeddings%'
ORDER BY mean_exec_time DESC;
```

---

## 🎯 Next Steps

1. **Setup Database**: Run `python scripts/setup_pgvector_db.py`
2. **Test Service**: Run `python test_pgvector_service.py`
3. **Migrate Data**: Run `python scripts/migrate_qdrant_to_pgvector.py`
4. **Update Code**: Change imports from Qdrant to PGVector
5. **Monitor Performance**: Use PostgreSQL monitoring tools

---

## ✅ Success Checklist

- [ ] PostgreSQL with pgvector extension installed
- [ ] Database and tables created
- [ ] Environment variables configured
- [ ] Test suite passes
- [ ] Migration completed (if applicable)
- [ ] Application updated to use PGVector service
- [ ] Performance monitoring in place

---

**🎉 You're now using PostgreSQL + pgvector for your legal document vectors!**

This implementation provides the same functionality as Qdrant while leveraging PostgreSQL's mature ecosystem and your existing database infrastructure. The interface remains identical, making the transition seamless.