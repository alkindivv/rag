"""
PDF Processing Configuration
Simplified configuration for pasal-level contextual PDF processing

This module provides streamlined configuration for legal document PDF processing,
focusing on pasal-level extraction with verse breakdown for contextual embeddings.

Author: Refactored Architecture
Purpose: Centralized PDF configuration following KISS principles
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class ExtractionStrategy(Enum):
    """PDF extraction strategies."""
    FASTEST = "fastest"
    HIGHEST_QUALITY = "highest_quality"
    FALLBACK_CASCADE = "fallback_cascade"
    PARALLEL_BEST = "parallel_best"
    OCR_ONLY = "ocr_only"
    HYBRID = "hybrid"


class ExtractionMethod(Enum):
    """Available extraction methods."""
    PYMUPDF = "pymupdf"
    PDFPLUMBER = "pdfplumber"
    PYPDF = "pypdf"
    OCR = "ocr"


@dataclass
class PDFConfig:
    """Configuration for PDF processing services"""

    # Extraction Strategy
    default_strategy: ExtractionStrategy = ExtractionStrategy.FALLBACK_CASCADE
    quality_preference: List[ExtractionMethod] = None

    # Processing Limits
    max_workers: int = 3
    timeout_seconds: int = 300
    min_confidence_threshold: float = 50.0
    max_text_length: int = 1000000
    min_pasal_length: int = 10

    # OCR Settings
    ocr_language: str = "ind+eng"
    ocr_dpi: int = 300
    ocr_timeout: int = 60

    # Structure Detection
    enable_structure_analysis: bool = True
    structure_confidence_threshold: float = 0.7

    # Pasal-Level Settings
    enable_pasal_aggregation: bool = True
    enable_verse_breakdown: bool = True
    enable_sub_element_extraction: bool = True
    max_verses_per_pasal: int = 50

    # Performance Settings
    enable_parallel_processing: bool = True
    enable_caching: bool = True
    fallback_to_ocr: bool = True

    # Debug Settings
    enable_debug_logging: bool = False
    save_intermediate_results: bool = False
    log_extraction_stats: bool = True

    def __post_init__(self):
        if self.quality_preference is None:
            self.quality_preference = [
                ExtractionMethod.PYMUPDF,
                ExtractionMethod.PYPDF,
                ExtractionMethod.PDFPLUMBER,
                ExtractionMethod.OCR
            ]

    @classmethod
    def from_env(cls) -> 'PDFConfig':
        """Create configuration from environment variables"""
        return cls(
            default_strategy=ExtractionStrategy(
                os.getenv('PDF_DEFAULT_STRATEGY', 'fallback_cascade')
            ),
            max_workers=int(os.getenv('PDF_MAX_WORKERS', '3')),
            timeout_seconds=int(os.getenv('PDF_TIMEOUT_SECONDS', '300')),
            min_confidence_threshold=float(os.getenv('PDF_MIN_CONFIDENCE', '50.0')),
            max_text_length=int(os.getenv('PDF_MAX_TEXT_LENGTH', '1000000')),
            min_pasal_length=int(os.getenv('PDF_MIN_PASAL_LENGTH', '10')),
            ocr_language=os.getenv('PDF_OCR_LANGUAGE', 'ind+eng'),
            ocr_dpi=int(os.getenv('PDF_OCR_DPI', '300')),
            ocr_timeout=int(os.getenv('PDF_OCR_TIMEOUT', '60')),
            enable_structure_analysis=os.getenv('PDF_ENABLE_STRUCTURE_ANALYSIS', 'true').lower() == 'true',
            structure_confidence_threshold=float(os.getenv('PDF_STRUCTURE_CONFIDENCE', '0.7')),
            enable_pasal_aggregation=os.getenv('PDF_ENABLE_PASAL_AGGREGATION', 'true').lower() == 'true',
            enable_verse_breakdown=os.getenv('PDF_ENABLE_VERSE_BREAKDOWN', 'true').lower() == 'true',
            enable_sub_element_extraction=os.getenv('PDF_ENABLE_SUB_ELEMENT_EXTRACTION', 'true').lower() == 'true',
            max_verses_per_pasal=int(os.getenv('PDF_MAX_VERSES_PER_PASAL', '50')),
            enable_parallel_processing=os.getenv('PDF_ENABLE_PARALLEL', 'true').lower() == 'true',
            enable_caching=os.getenv('PDF_ENABLE_CACHING', 'true').lower() == 'true',
            fallback_to_ocr=os.getenv('PDF_FALLBACK_TO_OCR', 'true').lower() == 'true',
            enable_debug_logging=os.getenv('PDF_DEBUG_LOGGING', 'false').lower() == 'true',
            save_intermediate_results=os.getenv('PDF_SAVE_INTERMEDIATE', 'false').lower() == 'true',
            log_extraction_stats=os.getenv('PDF_LOG_STATS', 'true').lower() == 'true'
        )


@dataclass
class PasalMetadataConfig:
    """Configuration for pasal-level metadata extraction"""

    # Aggregation Settings
    combine_all_verses: bool = True
    preserve_verse_numbering: bool = True
    include_sub_elements: bool = True

    # Hierarchy Context
    include_bab_context: bool = True
    include_document_metadata: bool = True

    # Content Processing
    clean_ocr_artifacts: bool = True
    normalize_whitespace: bool = True
    extract_legal_keywords: bool = True

    # Quality Control
    min_verse_length: int = 5
    max_pasal_content_length: int = 10000
    require_verse_numbering: bool = False

    # Sub-element Detection
    detect_huruf_elements: bool = True
    detect_angka_elements: bool = True
    extract_huruf_keywords: bool = True

    @classmethod
    def from_env(cls) -> 'PasalMetadataConfig':
        """Create configuration from environment variables"""
        return cls(
            combine_all_verses=os.getenv('PASAL_COMBINE_VERSES', 'true').lower() == 'true',
            preserve_verse_numbering=os.getenv('PASAL_PRESERVE_NUMBERING', 'true').lower() == 'true',
            include_sub_elements=os.getenv('PASAL_INCLUDE_SUB_ELEMENTS', 'true').lower() == 'true',
            include_bab_context=os.getenv('PASAL_INCLUDE_BAB_CONTEXT', 'true').lower() == 'true',
            include_document_metadata=os.getenv('PASAL_INCLUDE_DOC_METADATA', 'true').lower() == 'true',
            clean_ocr_artifacts=os.getenv('PASAL_CLEAN_OCR', 'true').lower() == 'true',
            normalize_whitespace=os.getenv('PASAL_NORMALIZE_WHITESPACE', 'true').lower() == 'true',
            extract_legal_keywords=os.getenv('PASAL_EXTRACT_KEYWORDS', 'true').lower() == 'true',
            min_verse_length=int(os.getenv('PASAL_MIN_VERSE_LENGTH', '5')),
            max_pasal_content_length=int(os.getenv('PASAL_MAX_CONTENT_LENGTH', '10000')),
            require_verse_numbering=os.getenv('PASAL_REQUIRE_NUMBERING', 'false').lower() == 'true',
            detect_huruf_elements=os.getenv('PASAL_DETECT_HURUF', 'true').lower() == 'true',
            detect_angka_elements=os.getenv('PASAL_DETECT_ANGKA', 'true').lower() == 'true',
            extract_huruf_keywords=os.getenv('PASAL_EXTRACT_HURUF_KEYWORDS', 'true').lower() == 'true'
        )


# Singleton instances
_pdf_config_instance: Optional[PDFConfig] = None
_pasal_config_instance: Optional[PasalMetadataConfig] = None


def get_pdf_config() -> PDFConfig:
    """Get singleton instance of PDF configuration"""
    global _pdf_config_instance
    if _pdf_config_instance is None:
        _pdf_config_instance = PDFConfig.from_env()
    return _pdf_config_instance


def get_pasal_metadata_config() -> PasalMetadataConfig:
    """Get singleton instance of pasal metadata configuration"""
    global _pasal_config_instance
    if _pasal_config_instance is None:
        _pasal_config_instance = PasalMetadataConfig.from_env()
    return _pasal_config_instance


# Export constants
SUPPORTED_EXTRACTION_METHODS = [method.value for method in ExtractionMethod]
SUPPORTED_EXTRACTION_STRATEGIES = [strategy.value for strategy in ExtractionStrategy]

# Default test configuration
DEFAULT_TEST_CONFIG = PDFConfig(
    default_strategy=ExtractionStrategy.FASTEST,
    timeout_seconds=60,
    ocr_dpi=150,  # Lower DPI for faster testing
    ocr_timeout=30,
    max_text_length=100000,
    enable_debug_logging=True,
    save_intermediate_results=True
)

DEFAULT_TEST_PASAL_CONFIG = PasalMetadataConfig(
    max_pasal_content_length=5000,  # Smaller for testing
    min_verse_length=3,
    require_verse_numbering=False,
    extract_legal_keywords=True
)
