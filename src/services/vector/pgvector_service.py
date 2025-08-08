"""
Production PGVector Service
Clean, simple, powerful vector storage and search for legal documents.

Stores vectors with essential metadata for accurate semantic search.
Uses UUID5 for consistency, optimized for production performance.
Single file, no overengineering, just what works.

Author: Production System
Purpose: Semantic search for Indonesian legal documents
"""

import logging
import os
import uuid
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
from datetime import datetime

# SQLAlchemy and pgvector
from sqlalchemy import create_engine, text, and_, or_, select
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

# Import models
try:
    from ...models.vector_storage import DocumentVector, VectorSearchLog, Base
    from ...models.document_storage import LegalDocument
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from models.vector_storage import DocumentVector, VectorSearchLog, Base
    from models.document_storage import LegalDocument

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    """Simple vector record for compatibility with existing interfaces."""
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    score: float = 0.0


@dataclass
class SearchResult:
    """Search result container."""
    records: List[VectorRecord]
    total_found: int
    search_time: float


class PGVectorService:
    """
    Production-ready pgvector service for legal document semantic search.

    Features:
    - UUID5 consistency across systems
    - Optimized HNSW indexes
    - Smart filtering and search
    - Production error handling
    - Performance monitoring
    """

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize with production configuration."""
        self.connection_string = connection_string or self._get_connection_string()

        # Create engine with production settings
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Setup database
        self._setup_database()

    def _get_connection_string(self) -> str:
        """Get connection string from environment."""
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        database = os.getenv("POSTGRES_DB", "postgres")
        username = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "")

        return f"postgresql://{username}:{password}@{host}:{port}/{database}"

    def _create_engine(self):
        """Create optimized SQLAlchemy engine."""
        return create_engine(
            self.connection_string,
            pool_size=int(os.getenv("POSTGRES_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("POSTGRES_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("POSTGRES_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("POSTGRES_POOL_RECYCLE", "3600")),
            echo=os.getenv("SQL_ECHO", "false").lower() == "true"
        )

    def _setup_database(self):
        """Setup database with extensions and tables."""
        try:
            with self.engine.connect() as conn:
                # Enable pgvector extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

                # Enable trigram extension for text search
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

                conn.commit()

            # Create all tables
            Base.metadata.create_all(bind=self.engine)

            logger.info("✅ PGVector database setup completed")

        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            raise

    def store_vectors(self, records: List[Dict[str, Any]], document_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store vectors with essential metadata for semantic search.

        Args:
            records: List of chunked content with vectors
            document_metadata: Document metadata for consistency

        Returns:
            Success status
        """
        if not records:
            logger.warning("No records to store")
            return True

        try:
            with self.SessionLocal() as session:
                vectors_to_store = []

                for record in records:
                    # Generate consistent UUID5
                    doc_id = record.get('doc_id') or (document_metadata.get('doc_id') if document_metadata else None)
                    if not doc_id:
                        # Generate from document info if available
                        doc_key = f"{record.get('doc_type', '')}-{record.get('doc_number', '')}-{record.get('doc_year', '')}"
                        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
                        doc_id = str(uuid.uuid5(namespace, doc_key))

                    # Prepare document metadata - prioritize individual record metadata over document metadata
                    doc_meta = {
                        'doc_id': doc_id,
                        'doc_type': record.get('doc_type') or (document_metadata.get('doc_type') if document_metadata else ''),
                        'doc_number': record.get('doc_number') or (document_metadata.get('doc_number') if document_metadata else ''),
                        'doc_year': record.get('doc_year') or (document_metadata.get('doc_year') if document_metadata else 0),
                        'doc_status': record.get('doc_status') or (document_metadata.get('doc_status') if document_metadata else 'Berlaku'),
                        'subject_areas': record.get('subject_areas') or (document_metadata.get('subject_areas') if document_metadata else [])
                    }

                    # Create vector record
                    vector_record = DocumentVector.from_chunk_data(record, doc_meta)
                    vectors_to_store.append(vector_record)

                # Batch upsert using merge for UUID5 consistency
                for vector_record in vectors_to_store:
                    session.merge(vector_record)

                session.commit()

                logger.info(f"✅ Stored {len(vectors_to_store)} vectors successfully")
                return True

        except Exception as e:
            logger.error(f"❌ Error storing vectors: {e}")
            return False

    def search_vectors(self,
                      query_vector: List[float],
                      limit: int = 10,
                      score_threshold: float = 0.0,
                      filters: Optional[Dict[str, Any]] = None) -> SearchResult:
        """
        Semantic search with smart filtering.

        Args:
            query_vector: Query embedding vector (768-dim)
            limit: Maximum results
            score_threshold: Minimum similarity score (0-1)
            filters: Search filters (doc_type, doc_year, status, etc.)

        Returns:
            SearchResult with ranked results
        """
        start_time = time.time()

        try:
            with self.SessionLocal() as session:
                # Base query with cosine similarity
                query = session.query(
                    DocumentVector,
                    (1 - DocumentVector.embedding.cosine_distance(query_vector)).label('similarity')
                )

                # Apply filters
                if filters:
                    # Document type filter
                    if filters.get('doc_type'):
                        query = query.filter(DocumentVector.doc_type == filters['doc_type'])

                    # Year filter
                    if filters.get('doc_year'):
                        query = query.filter(DocumentVector.doc_year == filters['doc_year'])

                    # Year range filter
                    if filters.get('year_from'):
                        query = query.filter(DocumentVector.doc_year >= filters['year_from'])
                    if filters.get('year_to'):
                        query = query.filter(DocumentVector.doc_year <= filters['year_to'])

                    # Status filter
                    if filters.get('doc_status'):
                        query = query.filter(DocumentVector.doc_status == filters['doc_status'])

                    # Citation/hierarchy filter
                    if filters.get('citation'):
                        query = query.filter(DocumentVector.hierarchy_path.ilike(f"%{filters['citation']}%"))

                    # Content type filter
                    if filters.get('content_type'):
                        query = query.filter(DocumentVector.content_type == filters['content_type'])

                    # BAB filter
                    if filters.get('bab'):
                        query = query.filter(DocumentVector.bab_number == str(filters['bab']))

                    # Pasal filter
                    if filters.get('pasal'):
                        query = query.filter(DocumentVector.pasal_number == str(filters['pasal']))

                    # Subject area filter
                    if filters.get('subject'):
                        query = query.filter(DocumentVector.subject_areas.ilike(f"%{filters['subject']}%"))

                # Order by similarity and limit
                results = query.order_by(text("similarity DESC")).limit(limit).all()

                # Apply similarity threshold in Python if needed
                if score_threshold > 0:
                    filtered_results = []
                    for vector_doc, similarity in results:
                        if float(similarity) >= score_threshold:
                            filtered_results.append((vector_doc, similarity))
                    results = filtered_results

                # Convert to VectorRecord format
                records = []
                for vector_doc, similarity in results:
                    metadata = vector_doc.to_search_result(float(similarity))

                    record = VectorRecord(
                        id=str(vector_doc.id),
                        vector=[],  # Don't return vectors to save bandwidth
                        metadata=metadata,
                        score=float(similarity)
                    )
                    records.append(record)

                search_time = time.time() - start_time

                # Log search for analytics (optional)
                self._log_search(query_vector, filters, len(records), search_time * 1000, session)

                return SearchResult(
                    records=records,
                    total_found=len(records),
                    search_time=search_time
                )

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return SearchResult(records=[], total_found=0, search_time=time.time() - start_time)

    def delete_by_filter(self, filters: Dict[str, Any]) -> bool:
        """Delete vectors by filter criteria."""
        try:
            with self.SessionLocal() as session:
                query = session.query(DocumentVector)

                # Apply filters
                if filters.get('doc_type'):
                    query = query.filter(DocumentVector.doc_type == filters['doc_type'])

                if filters.get('document_id'):
                    query = query.filter(DocumentVector.document_id == uuid.UUID(filters['document_id']))

                if filters.get('doc_year'):
                    query = query.filter(DocumentVector.doc_year == filters['doc_year'])

                # Delete matching records
                deleted_count = query.delete(synchronize_session=False)
                session.commit()

                logger.info(f"✅ Deleted {deleted_count} vectors matching filters")
                return True

        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
            return False

    def clear_collection(self, confirm: bool = False) -> bool:
        """Clear all vectors. Use with extreme caution!"""
        if not confirm:
            logger.warning("Clear requires confirm=True")
            return False

        try:
            with self.SessionLocal() as session:
                deleted_count = session.query(DocumentVector).delete()
                session.commit()

                logger.info(f"🗑️ Cleared collection - deleted {deleted_count} vectors")
                return True

        except Exception as e:
            logger.error(f"❌ Clear error: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            with self.SessionLocal() as session:
                # Count vectors
                total_vectors = session.query(DocumentVector).count()

                # Count by document type
                type_counts = session.execute(text("""
                    SELECT doc_type, COUNT(*) as count
                    FROM document_vectors
                    GROUP BY doc_type
                    ORDER BY count DESC
                """)).fetchall()

                # Get table size
                size_result = session.execute(text("""
                    SELECT pg_total_relation_size('document_vectors') as size_bytes,
                           pg_size_pretty(pg_total_relation_size('document_vectors')) as size_pretty
                """)).fetchone()

                return {
                    'name': 'document_vectors',
                    'points_count': total_vectors,
                    'vectors_count': total_vectors,
                    'status': 'active',
                    'disk_data_size': size_result.size_bytes if size_result else 0,
                    'disk_data_size_pretty': size_result.size_pretty if size_result else "0 bytes",
                    'document_types': {row.doc_type: row.count for row in type_counts}
                }

        except Exception as e:
            logger.error(f"❌ Collection info error: {e}")
            return {'name': 'document_vectors', 'points_count': 0, 'status': 'error'}

    def health_check(self) -> bool:
        """Simple health check."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False

    def _log_search(self, query_vector: List[float], filters: Optional[Dict],
                   results_count: int, duration_ms: float, session: Session):
        """Log search for analytics (optional)."""
        try:
            # Create query vector hash for deduplication
            vector_str = ','.join(map(str, query_vector[:10]))  # First 10 dimensions for hash
            query_hash = hashlib.sha256(vector_str.encode()).hexdigest()[:16]

            log_entry = VectorSearchLog(
                query_vector_hash=query_hash,
                filters_used=str(filters) if filters else '',
                results_found=results_count,
                search_duration_ms=int(duration_ms)
            )

            session.add(log_entry)
            # Don't commit here - let parent transaction handle it

        except Exception as e:
            # Don't fail search if logging fails
            logger.debug(f"Search logging failed: {e}")


# Compatibility functions for existing interfaces
def setup_pgvector_for_legal_docs(connection_string: Optional[str] = None) -> PGVectorService:
    """Setup pgvector service for legal documents."""
    return PGVectorService(connection_string=connection_string)


def store_chunked_documents(service: PGVectorService,
                          chunked_documents: List[Dict[str, Any]],
                          document_metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Store chunked documents with vectors."""
    return service.store_vectors(chunked_documents, document_metadata)


def search_legal_documents(service: PGVectorService,
                         query_vector: List[float],
                         doc_type: Optional[str] = None,
                         citation_filter: Optional[str] = None,
                         year_filter: Optional[int] = None,
                         limit: int = 10) -> SearchResult:
    """Search legal documents with common filters."""
    filters = {}
    if doc_type:
        filters['doc_type'] = doc_type
    if citation_filter:
        filters['citation'] = citation_filter
    if year_filter:
        filters['doc_year'] = year_filter

    return service.search_vectors(
        query_vector=query_vector,
        limit=limit,
        filters=filters if filters else None
    )
