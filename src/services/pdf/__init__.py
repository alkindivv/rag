"""
PDF Services Module
Simplified PDF processing for pasal-level contextual embeddings

This module provides streamlined PDF processing capabilities specifically designed for
Indonesian legal documents, focusing on pasal-level extraction with verse breakdown
for contextual embedding approach.

Components:
- PDF Orchestrator: Coordinates different PDF extraction methods
- Enhanced Structure Detector: Flexible boolean flags approach for structure detection
- Pasal Metadata Extractor: Simplified pasal-level extraction with verse breakdown
- Various PDF Extractors: PyMuPDF, PDFPlumber, PyPDF, OCR methods

Features:
- KISS principle: Simplified architecture focused on pasal-level extraction
- Maintains high accuracy for Indonesian legal documents
- Supports all major PDF extraction methods with fallback strategies
- Pasal-level aggregation with comprehensive verse breakdown metadata
- Optimized for contextual embedding pipeline
"""

# Core PDF processing imports
from .pdf_orchestrator import PDFOrchestrator
from .extractor import ExtractionResult, UnifiedPDFExtractor

# Export all components
__all__ = [
    # Main orchestrator
    'PDFOrchestrator',
    'ExtractionResult',
    'UnifiedPDFExtractor'
]

# Version information
__version__ = '2.0.0'
__author__ = 'Refactored Architecture Team'
__description__ = 'Simplified PDF processing for pasal-level contextual embeddings'

# Supported document types for Indonesian legal documents
SUPPORTED_DOCUMENT_TYPES = [
    'undang_undang',
    'peraturan_pemerintah',
    'peraturan_presiden',
    'peraturan_menteri',
    'keputusan_presiden',
    'instruksi_presiden',
    'surat_edaran',
    'ratifikasi',
    'peraturan_daerah'
]

# Supported extraction methods
SUPPORTED_EXTRACTION_METHODS = [
    'pymupdf',
    'pdfplumber',
    'pypdf'
]

# Pasal-level structure elements
PASAL_STRUCTURE_ELEMENTS = [
    'BAB',     # Chapter (for context)
    'PASAL',   # Article (main unit)
    'AYAT',    # Verse (breakdown within pasal)
    'HURUF',   # Letter enumeration (a, b, c)
    'ANGKA'    # Number enumeration (1, 2, 3)
]

# Performance targets for simplified architecture
PERFORMANCE_TARGETS = {
    'pasal_extraction_accuracy': 0.95,
    'verse_breakdown_accuracy': 0.90,
    'structure_confidence_threshold': 0.70,
    'max_processing_time': 30.0,
    'test_coverage': 0.95,
    'max_file_lines': 300
}

def get_module_info():
    """Get comprehensive module information"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'supported_document_types': SUPPORTED_DOCUMENT_TYPES,
        'supported_extraction_methods': SUPPORTED_EXTRACTION_METHODS,
        'pasal_structure_elements': PASAL_STRUCTURE_ELEMENTS,
        'performance_targets': PERFORMANCE_TARGETS,
        'components': len(__all__),
        'architecture': 'simplified_pasal_level'
    }
