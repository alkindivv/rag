"""
Vector Storage Model
Production-ready pgvector model for semantic search.

Stores vectors with essential metadata for accurate semantic search.
Uses UUID5 for consistency with document storage.
Simple, clean implementation following KISS principles.
"""

from sqlalchemy import Column, String, Integer, Text, DateTime, Index, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime
from typing import Dict, Any, List

Base = declarative_base()


class DocumentVector(Base):
    """
    Vector storage for semantic search of legal document chunks.
    Essential metadata only - optimized for search performance.
    """
    __tablename__ = "document_vectors"

    # Primary identifier - UUID5 for consistency
    id = Column(UUID(as_uuid=True), primary_key=True)

    # Vector embedding (Gemini text-embedding-004)
    embedding = Column(Vector(768), nullable=False)

    # Document reference (UUID5 matching legal_documents.id)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Content information
    content_text = Column(Text, nullable=False)
    content_type = Column(String(50), nullable=False, index=True)  # 'pasal', 'ayat', 'bab', 'full_doc'

    # Legal hierarchy path for precise citations
    hierarchy_path = Column(String(500), nullable=True, index=True)

    # Essential metadata for search filtering
    doc_type = Column(String(20), nullable=False, index=True)  # UU, PP, Perpres
    doc_year = Column(Integer, nullable=False, index=True)
    doc_number = Column(String(100), nullable=False, index=True)
    doc_status = Column(String(50), nullable=False, index=True)  # Berlaku, Dicabut

    # Subject classification for domain filtering
    subject_areas = Column(String(500), index=True)  # Comma-separated for simple filtering

    # Legal structure identifiers
    bab_number = Column(String(20), index=True)  # "I", "II", "III"
    pasal_number = Column(String(20), index=True)  # "1", "2", "3"
    ayat_number = Column(String(20), index=True)  # "1", "2", "3"

    # Content metrics
    token_count = Column(Integer, default=0)
    char_count = Column(Integer, default=0)

    # Processing metadata
    embedding_model = Column(String(100), default='text-embedding-004')
    embedding_version = Column(String(20), default='v1')

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Optimized indexes for semantic search and filtering
    __table_args__ = (
        # HNSW index for vector similarity search (cosine distance)
        Index(
            'idx_document_vector_embedding_cosine',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),

        # Composite indexes for filtered search
        Index('idx_doc_vector_type_year', 'doc_type', 'doc_year'),
        Index('idx_doc_vector_status_type', 'doc_status', 'doc_type'),
        Index('idx_doc_vector_document_content', 'document_id', 'content_type'),

        # Legal structure navigation
        Index('idx_doc_vector_hierarchy', 'bab_number', 'pasal_number', 'ayat_number'),

        # Subject filtering (GIN for flexible text search)
        Index('idx_doc_vector_subjects', 'subject_areas', postgresql_using='gin',
              postgresql_ops={'subject_areas': 'gin_trgm_ops'}),

        # Hierarchy path search
        Index('idx_doc_vector_hierarchy_path', 'hierarchy_path', postgresql_using='gin',
              postgresql_ops={'hierarchy_path': 'gin_trgm_ops'}),
    )

    def __repr__(self):
        return f"<DocumentVector(id={self.id}, doc_type={self.doc_type}, hierarchy='{self.hierarchy_path[:50]}...')>"

    def to_search_result(self, similarity_score: float = 0.0) -> Dict[str, Any]:
        """Convert to search result format"""
        return {
            'id': str(self.id),
            'score': similarity_score,
            'content': self.content_text,
            'citation': self.hierarchy_path,
            'doc_type': self.doc_type,
            'doc_year': self.doc_year,
            'doc_number': self.doc_number,
            'doc_status': self.doc_status,
            'content_type': self.content_type,
            'subject_areas': self.subject_areas.split(',') if self.subject_areas else [],
            'bab': self.bab_number,
            'pasal': self.pasal_number,
            'ayat': self.ayat_number,
            'token_count': self.token_count,
            'char_count': self.char_count
        }

    @classmethod
    def generate_consistent_id(cls, document_id: uuid.UUID, content_hash: str) -> uuid.UUID:
        """Generate UUID5 for consistent vector IDs"""
        namespace = uuid.UUID('6ba7b811-9dad-11d1-80b4-00c04fd430c8')  # Custom namespace for vectors
        unique_key = f"{document_id}-{content_hash}"
        return uuid.uuid5(namespace, unique_key)

    @classmethod
    def from_chunk_data(cls, chunk_data: Dict[str, Any], document_metadata: Dict[str, Any]) -> 'DocumentVector':
        """Create DocumentVector from chunked content with document metadata"""

        # Generate consistent UUID5
        content_hash = str(hash(chunk_data.get('content', '')))
        vector_id = cls.generate_consistent_id(
            uuid.UUID(document_metadata['doc_id']),
            content_hash
        )

        # Parse hierarchy for structured fields
        hierarchy = chunk_data.get('citation', '')
        bab_num = pasal_num = ayat_num = None

        # Extract BAB number
        if 'BAB' in hierarchy:
            try:
                bab_part = hierarchy.split('BAB')[1].split('›')[0].strip()
                bab_num = bab_part.split()[0] if bab_part else None
            except:
                pass

        # Extract Pasal number
        if 'Pasal' in hierarchy:
            try:
                pasal_part = hierarchy.split('Pasal')[1].split('›')[0].strip()
                pasal_num = pasal_part.split()[0] if pasal_part else None
            except:
                pass

        # Extract Ayat number
        if 'Ayat' in hierarchy:
            try:
                ayat_part = hierarchy.split('Ayat')[1].strip()
                ayat_num = ayat_part.replace('(', '').replace(')', '').split()[0] if ayat_part else None
            except:
                pass

        # Join subject areas for simple search
        subjects = document_metadata.get('subject_areas', [])
        subject_str = ','.join(subjects) if isinstance(subjects, list) else str(subjects)

        content_text = chunk_data.get('content', '')

        return cls(
            id=vector_id,
            embedding=chunk_data.get('vector', chunk_data.get('embedding')),
            document_id=uuid.UUID(document_metadata['doc_id']),
            content_text=content_text,
            content_type=chunk_data.get('chunk_type', 'pasal'),
            hierarchy_path=hierarchy,
            doc_type=document_metadata.get('doc_type', ''),
            doc_year=int(document_metadata.get('doc_year', 0)),
            doc_number=str(document_metadata.get('doc_number', '')),
            doc_status=document_metadata.get('doc_status', 'Berlaku'),
            subject_areas=subject_str,
            bab_number=bab_num,
            pasal_number=pasal_num,
            ayat_number=ayat_num,
            token_count=chunk_data.get('token_count', 0),
            char_count=len(content_text),
            embedding_model=chunk_data.get('model_used', 'text-embedding-004'),
            embedding_version=chunk_data.get('embedding_version', 'v1')
        )


class VectorSearchLog(Base):
    """
    Optional: Log search queries for analytics and improvement.
    Simple table for tracking search patterns.
    """
    __tablename__ = "vector_search_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Query information
    query_text = Column(Text)
    query_vector_hash = Column(String(64))  # Hash of query vector for deduplication

    # Search parameters
    filters_used = Column(String(500))  # JSON string of filters
    limit_requested = Column(Integer, default=10)

    # Results metadata
    results_found = Column(Integer, default=0)
    search_duration_ms = Column(Integer, default=0)

    # User context (optional)
    user_session = Column(String(100), index=True)

    # Timestamp
    searched_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_search_log_session_time', 'user_session', 'searched_at'),
        Index('idx_search_log_query_hash', 'query_vector_hash'),
    )

    def __repr__(self):
        return f"<VectorSearchLog(id={self.id}, results={self.results_found}, duration={self.search_duration_ms}ms)>"
