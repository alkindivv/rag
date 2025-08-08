"""
Services Package
Modular services for legal document processing, AI, crawling, and embedding operations.
"""

# PDF Processing Services
from .pdf import *

# AI Services
from .ai import *

# Crawler Services
from .crawler import *

# Embedding Services
from .embedding import *

__all__ = [
    # PDF Services
    'PyMuPDFExtractor',
    'PDFPlumberExtractor',
    'PyPDFExtractor',
    'OCRExtractor',
    # 'StructureDetector',
    'PDFOrchestrator',

    # AI Services
    'SummarizerService',
    'QAEngine',
    'AnalyzerService',
    'ComparatorService',
    'AIOrchestrator',

    # Crawler Services
    'MetadataExtractor',
    'WebScraper',
    'FileDownloader',
    'CrawlerOrchestrator',

    # Embedding Services
    'RAGOrchestrator',
    'RAGResponse',
    'DocumentChunker',
    'DocumentChunk',
    'ContextManager',
    'RAGContext',
    'RAGEngine'
]
