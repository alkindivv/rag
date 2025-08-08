"""
Chunking Configuration
Centralized configuration for adaptive legal document chunking.

This module provides configuration for the Context7-inspired adaptive chunking
system, replacing complex pasal metadata configuration with simple but powerful
settings optimized for contextual embedding and search.
"""

import os
from typing import Dict, Any, List
from enum import Enum


class ChunkingMode(Enum):
    """Chunking operation modes."""
    ADAPTIVE = "adaptive"           # Context7-inspired adaptive chunking
    PASAL_ONLY = "pasal_only"      # Strict pasal-level chunking
    HIERARCHICAL = "hierarchical"   # Full hierarchy preservation
    FIXED_SIZE = "fixed_size"      # Fixed token size chunks


class SplittingStrategy(Enum):
    """Strategies for handling oversized chunks."""
    AYAT_SPLIT = "ayat_split"       # Split by ayat boundaries
    HURUF_SPLIT = "huruf_split"     # Split by huruf boundaries
    SENTENCE_SPLIT = "sentence_split"  # Split by sentence boundaries
    FORCE_SPLIT = "force_split"     # Force split with overlap


class ChunkingConfig:
    """
    Configuration for adaptive legal document chunking.

    Provides simple but powerful configuration for Context7-inspired
    adaptive chunking optimized for Indonesian legal documents.
    """

    # === CORE CHUNKING SETTINGS ===

    # Token limits
    DEFAULT_MAX_TOKENS = int(os.getenv("CHUNKING_MAX_TOKENS", "1500"))
    MAX_CHUNK_TOKENS = int(os.getenv("CHUNKING_MAX_CHUNK_TOKENS", "2000"))
    MIN_CHUNK_TOKENS = int(os.getenv("CHUNKING_MIN_CHUNK_TOKENS", "50"))

    # Overlap settings
    DEFAULT_OVERLAP = int(os.getenv("CHUNKING_DEFAULT_OVERLAP", "100"))
    MAX_OVERLAP = int(os.getenv("CHUNKING_MAX_OVERLAP", "300"))

    # === ADAPTIVE CHUNKING SETTINGS ===

    # Primary chunking unit (always PASAL for legal documents)
    PRIMARY_UNIT = "pasal"

    # Hierarchy levels for context building
    HIERARCHY_LEVELS = [
        "buku",      # Book
        "bab",       # Chapter
        "bagian",    # Section
        "paragraf",  # Paragraph
        "pasal",     # Article (primary)
        "ayat",      # Verse
        "huruf",     # Letter
        "angka"      # Number
    ]

    # === SPLITTING THRESHOLDS ===

    # When to trigger different splitting strategies
    AYAT_SPLIT_THRESHOLD = int(os.getenv("CHUNKING_AYAT_SPLIT_THRESHOLD", "1200"))
    HURUF_SPLIT_THRESHOLD = int(os.getenv("CHUNKING_HURUF_SPLIT_THRESHOLD", "800"))
    SENTENCE_SPLIT_THRESHOLD = int(os.getenv("CHUNKING_SENTENCE_SPLIT_THRESHOLD", "1500"))

    # === CONTEXT BUILDING SETTINGS ===

    # Maximum semantic keywords per chunk
    MAX_SEMANTIC_KEYWORDS = int(os.getenv("CHUNKING_MAX_KEYWORDS", "10"))

    # Context quality thresholds
    MIN_CONTEXT_QUALITY = float(os.getenv("CHUNKING_MIN_CONTEXT_QUALITY", "0.3"))
    TARGET_CONTEXT_QUALITY = float(os.getenv("CHUNKING_TARGET_CONTEXT_QUALITY", "0.8"))

    # === CITATION SETTINGS ===

    # Citation formats
    CITATION_FORMATS = {
        "formal": "Berdasarkan {citation}",
        "conversational": "Menurut {citation}",
        "academic": "Sebagaimana diatur dalam {citation}",
        "short": "{citation}"
    }

    # Document type abbreviations
    DOCUMENT_ABBREVIATIONS = {
        "undang-undang": "UU",
        "peraturan pemerintah": "PP",
        "peraturan presiden": "Perpres",
        "peraturan menteri": "Permen",
        "keputusan presiden": "Keppres",
        "instruksi presiden": "Inpres",
        "peraturan daerah": "Perda",
        "keputusan menteri": "Kepmen",
        "surat edaran": "SE"
    }

    # === SEMANTIC KEYWORD SETTINGS ===

    # Legal domain categories for keyword extraction
    LEGAL_DOMAINS = {
        "hukum_pidana": {
            "keywords": ["pidana", "sanksi", "hukuman", "pelanggaran", "kejahatan", "tindak pidana"],
            "weight": 1.5
        },
        "hukum_perdata": {
            "keywords": ["perdata", "kontrak", "perjanjian", "ganti rugi", "wanprestasi"],
            "weight": 1.3
        },
        "hukum_administrasi": {
            "keywords": ["administrasi", "perizinan", "pelayanan publik", "birokrasi"],
            "weight": 1.2
        },
        "hukum_tata_negara": {
            "keywords": ["konstitusi", "pemerintahan", "kekuasaan", "kedaulatan"],
            "weight": 1.4
        },
        "hak_asasi": {
            "keywords": ["hak asasi", "kebebasan", "martabat", "kemanusiaan"],
            "weight": 1.6
        }
    }

    # Legal entity patterns
    LEGAL_ENTITIES = [
        "pemerintah", "menteri", "presiden", "wakil presiden", "dpr", "dpd", "mpr",
        "mahkamah", "pengadilan", "kejaksaan", "kepolisian"
    ]

    # Legal concept patterns
    LEGAL_CONCEPTS = [
        "hak", "kewajiban", "tanggung jawab", "wewenang", "kekuasaan", "kebijakan",
        "peraturan", "ketentuan", "prosedur", "mekanisme"
    ]

    # === PERFORMANCE SETTINGS ===

    # Processing limits
    MAX_PROCESSING_TIME = int(os.getenv("CHUNKING_MAX_PROCESSING_TIME", "300"))  # 5 minutes
    ENABLE_PARALLEL_PROCESSING = os.getenv("CHUNKING_ENABLE_PARALLEL", "true").lower() == "true"

    # Memory limits
    MAX_MEMORY_USAGE = int(os.getenv("CHUNKING_MAX_MEMORY_MB", "1024"))  # 1GB

    # === DEBUG AND LOGGING SETTINGS ===

    # Debug flags
    ENABLE_DEBUG_LOGGING = os.getenv("CHUNKING_DEBUG_LOGGING", "false").lower() == "true"
    LOG_CHUNK_STATISTICS = os.getenv("CHUNKING_LOG_STATS", "true").lower() == "true"
    SAVE_INTERMEDIATE_RESULTS = os.getenv("CHUNKING_SAVE_INTERMEDIATE", "false").lower() == "true"

    # Validation settings
    ENABLE_CHUNK_VALIDATION = os.getenv("CHUNKING_ENABLE_VALIDATION", "true").lower() == "true"
    VALIDATE_CITATIONS = os.getenv("CHUNKING_VALIDATE_CITATIONS", "true").lower() == "true"

    @classmethod
    def get_chunking_config(cls, mode: ChunkingMode = ChunkingMode.ADAPTIVE) -> Dict[str, Any]:
        """
        Get configuration for specific chunking mode.

        Args:
            mode: Chunking mode to configure

        Returns:
            Configuration dictionary for the specified mode
        """
        base_config = {
            "max_tokens": cls.DEFAULT_MAX_TOKENS,
            "overlap": cls.DEFAULT_OVERLAP,
            "min_tokens": cls.MIN_CHUNK_TOKENS,
            "primary_unit": cls.PRIMARY_UNIT,
            "hierarchy_levels": cls.HIERARCHY_LEVELS
        }

        if mode == ChunkingMode.ADAPTIVE:
            base_config.update({
                "enable_adaptive_splitting": True,
                "splitting_strategies": [
                    SplittingStrategy.AYAT_SPLIT,
                    SplittingStrategy.HURUF_SPLIT,
                    SplittingStrategy.SENTENCE_SPLIT
                ],
                "context_building": True,
                "semantic_keywords": True,
                "citation_building": True
            })

        elif mode == ChunkingMode.PASAL_ONLY:
            base_config.update({
                "enable_adaptive_splitting": False,
                "max_tokens": cls.MAX_CHUNK_TOKENS,  # Allow larger chunks
                "context_building": True,
                "semantic_keywords": True,
                "citation_building": True
            })

        elif mode == ChunkingMode.HIERARCHICAL:
            base_config.update({
                "preserve_full_hierarchy": True,
                "include_all_levels": True,
                "context_building": True,
                "semantic_keywords": True,
                "citation_building": True
            })

        elif mode == ChunkingMode.FIXED_SIZE:
            base_config.update({
                "fixed_size": True,
                "ignore_hierarchy": True,
                "context_building": False,
                "semantic_keywords": False,
                "citation_building": False
            })

        return base_config

    @classmethod
    def get_splitting_config(cls, strategy: SplittingStrategy) -> Dict[str, Any]:
        """
        Get configuration for specific splitting strategy.

        Args:
            strategy: Splitting strategy to configure

        Returns:
            Configuration dictionary for the splitting strategy
        """
        if strategy == SplittingStrategy.AYAT_SPLIT:
            return {
                "pattern": r'\((\d+)\)',
                "preserve_markers": True,
                "min_split_size": 100,
                "threshold": cls.AYAT_SPLIT_THRESHOLD
            }

        elif strategy == SplittingStrategy.HURUF_SPLIT:
            return {
                "pattern": r'^([a-z])\.\s+',
                "preserve_markers": True,
                "min_split_size": 50,
                "threshold": cls.HURUF_SPLIT_THRESHOLD
            }

        elif strategy == SplittingStrategy.SENTENCE_SPLIT:
            return {
                "pattern": r'(?<=[.!?])\s+',
                "preserve_markers": False,
                "min_split_size": 30,
                "threshold": cls.SENTENCE_SPLIT_THRESHOLD,
                "enable_overlap": True
            }

        elif strategy == SplittingStrategy.FORCE_SPLIT:
            return {
                "force_split": True,
                "target_size": cls.DEFAULT_MAX_TOKENS,
                "overlap": cls.DEFAULT_OVERLAP,
                "preserve_words": True
            }

        return {}

    @classmethod
    def get_context_config(cls) -> Dict[str, Any]:
        """Get configuration for context building."""
        return {
            "max_keywords": cls.MAX_SEMANTIC_KEYWORDS,
            "min_quality": cls.MIN_CONTEXT_QUALITY,
            "target_quality": cls.TARGET_CONTEXT_QUALITY,
            "legal_domains": cls.LEGAL_DOMAINS,
            "legal_entities": cls.LEGAL_ENTITIES,
            "legal_concepts": cls.LEGAL_CONCEPTS,
            "citation_formats": cls.CITATION_FORMATS,
            "document_abbreviations": cls.DOCUMENT_ABBREVIATIONS
        }

    @classmethod
    def get_performance_config(cls) -> Dict[str, Any]:
        """Get performance-related configuration."""
        return {
            "max_processing_time": cls.MAX_PROCESSING_TIME,
            "enable_parallel": cls.ENABLE_PARALLEL_PROCESSING,
            "max_memory_mb": cls.MAX_MEMORY_USAGE,
            "enable_debug": cls.ENABLE_DEBUG_LOGGING,
            "log_stats": cls.LOG_CHUNK_STATISTICS,
            "save_intermediate": cls.SAVE_INTERMEDIATE_RESULTS,
            "enable_validation": cls.ENABLE_CHUNK_VALIDATION,
            "validate_citations": cls.VALIDATE_CITATIONS
        }

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate chunking configuration.

        Args:
            config: Configuration to validate

        Returns:
            Dictionary with validation results (errors and warnings)
        """
        errors = []
        warnings = []

        # Validate token limits
        max_tokens = config.get("max_tokens", cls.DEFAULT_MAX_TOKENS)
        if max_tokens < cls.MIN_CHUNK_TOKENS:
            errors.append(f"max_tokens ({max_tokens}) below minimum ({cls.MIN_CHUNK_TOKENS})")
        elif max_tokens > cls.MAX_CHUNK_TOKENS:
            warnings.append(f"max_tokens ({max_tokens}) above recommended maximum ({cls.MAX_CHUNK_TOKENS})")

        # Validate overlap
        overlap = config.get("overlap", cls.DEFAULT_OVERLAP)
        if overlap >= max_tokens:
            errors.append(f"overlap ({overlap}) must be less than max_tokens ({max_tokens})")
        elif overlap > cls.MAX_OVERLAP:
            warnings.append(f"overlap ({overlap}) above recommended maximum ({cls.MAX_OVERLAP})")

        # Validate hierarchy levels
        hierarchy_levels = config.get("hierarchy_levels", [])
        if not hierarchy_levels:
            warnings.append("No hierarchy levels specified")

        required_level = "pasal"
        if required_level not in hierarchy_levels:
            errors.append(f"Required hierarchy level '{required_level}' not found")

        return {
            "errors": errors,
            "warnings": warnings
        }


# Export configuration classes
__all__ = [
    'ChunkingMode',
    'SplittingStrategy',
    'ChunkingConfig'
]
