"""
Document Storage Model
Production-ready PostgreSQL model for complete legal document metadata.

Stores all crawler results with full metadata for legal documents.
Simple, clean implementation following KISS principles.
"""

from sqlalchemy import Column, String, Integer, Text, Date, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.sql import func
import uuid
from datetime import datetime, date
from typing import Optional, List, Dict, Any

Base = declarative_base()


class LegalDocument(Base):
    """
    Complete legal document storage with all crawler metadata.
    Single table for all document types: UU, PP, Perpres, Permenkes, etc.
    """
    __tablename__ = "legal_documents"

    # Primary identifier - UUID5 for consistency
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Basic document identification
    title = Column(Text, nullable=False, index=True)
    number = Column(String(100), nullable=False, index=True)
    form = Column(String(20), nullable=False, index=True)  # UU, PP, Perpres, etc.
    form_short = Column(String(10), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)

    # Source and classification
    source = Column(String(100), nullable=False, index=True)  # BPK, etc.
    type = Column(String(100), nullable=False)  # "Peraturan Perundang-undangan"
    teu = Column(String(255))  # "Indonesia, Pemerintah Pusat"

    # Geographic and administrative info
    place_enacted = Column(String(255))  # Jakarta, etc.
    language = Column(String(100), default="Bahasa Indonesia")
    location = Column(String(255))  # "Pemerintah Pusat"
    field = Column(String(255), index=True)  # "HUKUM UMUM"

    # Important dates
    date_enacted = Column(Date)
    date_promulgated = Column(Date)
    date_effective = Column(Date)

    # Status and classification
    status = Column(String(50), nullable=False, default="Berlaku", index=True)

    # Subject areas (JSONB array for better indexing)
    subject = Column(JSONB, default=list)  # ["PERTAHANAN DAN KEAMANAN", "MILITER"]

    # Legal relationships (JSONB arrays for better indexing)
    amends = Column(JSONB, default=list)  # Documents this amends
    revokes = Column(JSONB, default=list)  # Documents this revokes
    amended_by = Column(JSONB, default=list)  # Documents that amend this
    revoked_by = Column(JSONB, default=list)  # Documents that revoke this
    revokes_partially = Column(JSONB, default=list)
    revoked_partially_by = Column(JSONB, default=list)
    established_by = Column(JSONB, default=list)

    # URLs and file paths
    detail_url = Column(Text)  # Source detail page
    source_url = Column(Text)  # Original source URL
    pdf_url = Column(Text)  # PDF download URL
    uji_materi_pdf_url = Column(Text)  # Judicial review PDF URL
    pdf_path = Column(Text)  # Local PDF file path

    # Document content
    content = Column(Text)  # Full document text content

    # Full-text search vector (prepared for future FTS)
    content_vector = Column(TSVECTOR)

    # Processing metadata
    content_length = Column(Integer)  # Character count
    processing_status = Column(String(50), default="pending", index=True)
    error_message = Column(Text)  # Processing errors if any

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True))

    # Performance indexes
    __table_args__ = (
        # Composite indexes for common queries
        Index('idx_legal_doc_form_year', 'form', 'year'),
        Index('idx_legal_doc_source_status', 'source', 'status'),
        Index('idx_legal_doc_field_year', 'field', 'year'),

        # Subject search (GIN index for JSONB)
        Index('idx_legal_doc_subject', 'subject', postgresql_using='gin'),

        # Legal relationships indexes
        Index('idx_legal_doc_amends', 'amends', postgresql_using='gin'),
        Index('idx_legal_doc_revokes', 'revokes', postgresql_using='gin'),

        # Full-text search index (for future FTS implementation)
        Index('idx_legal_doc_fts', 'content_vector', postgresql_using='gin'),

        # Content search
        Index('idx_legal_doc_content_text', 'content', postgresql_using='gin',
              postgresql_ops={'content': 'gin_trgm_ops'}),
    )

    def __repr__(self):
        return f"<LegalDocument(id={self.id}, form={self.form}, number={self.number}, year={self.year}, title='{self.title[:50]}...')>"

    @property
    def document_identifier(self) -> str:
        """Generate standard document identifier: UU No. 3 Tahun 2025"""
        return f"{self.form} No. {self.number} Tahun {self.year}"

    @property
    def full_title(self) -> str:
        """Get full document title with identifier"""
        return f"{self.document_identifier} - {self.title}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': str(self.id),
            'title': self.title,
            'number': self.number,
            'form': self.form,
            'form_short': self.form_short,
            'year': self.year,
            'source': self.source,
            'type': self.type,
            'teu': self.teu,
            'place_enacted': self.place_enacted,
            'language': self.language,
            'location': self.location,
            'field': self.field,
            'date_enacted': self.date_enacted.isoformat() if self.date_enacted else None,
            'date_promulgated': self.date_promulgated.isoformat() if self.date_promulgated else None,
            'date_effective': self.date_effective.isoformat() if self.date_effective else None,
            'status': self.status,
            'subject': self.subject,
            'amends': self.amends,
            'revokes': self.revokes,
            'amended_by': self.amended_by,
            'revoked_by': self.revoked_by,
            'revokes_partially': self.revokes_partially,
            'revoked_partially_by': self.revoked_partially_by,
            'established_by': self.established_by,
            'detail_url': self.detail_url,
            'source_url': self.source_url,
            'pdf_url': self.pdf_url,
            'uji_materi_pdf_url': self.uji_materi_pdf_url,
            'pdf_path': self.pdf_path,
            'content_length': self.content_length,
            'processing_status': self.processing_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None
        }

    def get_vector_metadata(self) -> Dict[str, Any]:
        """Get essential metadata for vector storage"""
        return {
            'doc_id': str(self.id),
            'doc_type': self.form,
            'doc_number': self.number,
            'doc_year': self.year,
            'doc_title': self.title,
            'doc_status': self.status,
            'doc_field': self.field,
            'doc_source': self.source,
            'doc_identifier': self.document_identifier,
            'subject_areas': self.subject
        }

    @classmethod
    def from_crawler_data(cls, data: Dict[str, Any]) -> 'LegalDocument':
        """Create LegalDocument from crawler JSON data"""

        # Parse dates safely
        def parse_date(date_str: Optional[str]) -> Optional[date]:
            if not date_str:
                return None
            try:
                # Handle various date formats
                if isinstance(date_str, str):
                    # Try common Indonesian date format: "26 Maret 2025"
                    month_map = {
                        'Januari': '01', 'Februari': '02', 'Maret': '03', 'April': '04',
                        'Mei': '05', 'Juni': '06', 'Juli': '07', 'Agustus': '08',
                        'September': '09', 'Oktober': '10', 'November': '11', 'Desember': '12'
                    }
                    for indo_month, num_month in month_map.items():
                        if indo_month in date_str:
                            parts = date_str.split()
                            if len(parts) == 3:
                                day = parts[0].zfill(2)
                                year = parts[2]
                                return datetime.strptime(f"{year}-{num_month}-{day}", "%Y-%m-%d").date()
                return None
            except:
                return None

        # Generate UUID5 for consistency
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace
        doc_key = f"{data.get('form', '')}-{data.get('number', '')}-{data.get('year', '')}"
        doc_id = uuid.uuid5(namespace, doc_key)

        return cls(
            id=doc_id,
            title=data.get('title', ''),
            number=str(data.get('number', '')),
            form=data.get('form', ''),
            form_short=data.get('form_short', data.get('form', '')),
            year=int(data.get('year', 0)) if data.get('year') else None,
            source=data.get('source', ''),
            type=data.get('type', ''),
            teu=data.get('teu', ''),
            place_enacted=data.get('place_enacted', ''),
            language=data.get('language', 'Bahasa Indonesia'),
            location=data.get('location', ''),
            field=data.get('field', ''),
            date_enacted=parse_date(data.get('date_enacted')),
            date_promulgated=parse_date(data.get('date_promulgated')),
            date_effective=parse_date(data.get('date_effective')),
            status=data.get('status', 'Berlaku'),
            subject=data.get('subject', []),
            amends=data.get('amends', []),
            revokes=data.get('revokes', []),
            amended_by=data.get('amended_by', []),
            revoked_by=data.get('revoked_by', []),
            revokes_partially=data.get('revokes_partially', []),
            revoked_partially_by=data.get('revoked_partially_by', []),
            established_by=data.get('established_by', []),
            detail_url=data.get('detail_url', ''),
            source_url=data.get('source_url', ''),
            pdf_url=data.get('pdf_url', ''),
            uji_materi_pdf_url=data.get('uji_materi_pdf_url'),
            pdf_path=data.get('pdf_path', ''),
            content=data.get('content', ''),
            content_length=len(data.get('content', '')) if data.get('content') else 0
        )
