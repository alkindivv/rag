"""
AI Configuration Module
Centralized configuration for all AI services and components

This module provides configuration classes for AI services, including
model settings, prompt configurations, caching parameters, and
Indonesian legal document optimization settings.

Author: Refactored Architecture
Purpose: Single responsibility configuration management for AI services
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging


class ModelProvider(Enum):
    """Supported AI model providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"
    LOCAL = "local"


class SummaryType(Enum):
    """Types of document summaries."""
    BRIEF = "brief"
    DETAILED = "detailed"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    LEGAL = "legal"


class AnalysisType(Enum):
    """Types of AI analysis available."""
    SUMMARIZE = "summarize"
    STRUCTURE = "structure"
    DEFINITIONS = "definitions"
    SANCTIONS = "sanctions"
    QUESTION_ANSWER = "question_answer"
    COMPARE = "compare"
    EXPLAIN = "explain"
    DOCUMENT_INTELLIGENCE = "document_intelligence"
    LEGAL_ANALYSIS = "legal_analysis"
    CONCEPT_EXTRACTION = "concept_extraction"


@dataclass
class ModelConfig:
    """Configuration for AI model settings."""
    provider: ModelProvider = ModelProvider.GEMINI
    model_name: str = "gemini-2.0-flash-lite"
    api_key: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 8192
    top_p: float = 0.95
    top_k: int = 40
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("GEMINI_API_KEY")


@dataclass
class CacheConfig:
    """Configuration for AI response caching."""
    enabled: bool = True
    max_size: int = 1000
    ttl_hours: int = 24
    use_persistent_cache: bool = False
    cache_directory: str = "./cache/ai_responses"
    compression_enabled: bool = True
    eviction_policy: str = "lru"  # lru, fifo, random


@dataclass
class PromptConfig:
    """Configuration for prompt management."""
    language: str = "id"  # Indonesian
    max_prompt_length: int = 32000
    template_directory: str = "./templates/ai_prompts"
    use_dynamic_templates: bool = True
    legal_context_enabled: bool = True
    include_examples: bool = True
    prompt_optimization: bool = True


@dataclass
class SummarizerConfig:
    """Configuration for document summarizer service."""
    max_input_length: int = 50000
    default_summary_type: SummaryType = SummaryType.DETAILED
    max_summary_length: int = 1000
    min_summary_length: int = 100
    include_key_points: bool = True
    max_key_points: int = 10
    preserve_structure: bool = True
    legal_focus: bool = True
    confidence_threshold: float = 0.7


@dataclass
class QAEngineConfig:
    """Configuration for question-answering engine."""
    max_question_length: int = 500
    max_context_length: int = 10000
    max_answer_length: int = 2000
    confidence_threshold: float = 0.6
    use_context_ranking: bool = True
    max_context_sources: int = 5
    include_citations: bool = True
    legal_domain_focus: bool = True
    multi_hop_reasoning: bool = True


@dataclass
class AnalyzerConfig:
    """Configuration for content analyzer service."""
    max_content_length: int = 100000
    structure_detection: bool = True
    definition_extraction: bool = True
    sanction_identification: bool = True
    concept_mapping: bool = True
    min_confidence_score: float = 0.5
    max_concepts_per_document: int = 50
    legal_pattern_matching: bool = True
    hierarchy_analysis: bool = True


@dataclass
class ComparatorConfig:
    """Configuration for document comparator service."""
    max_document_length: int = 50000
    similarity_threshold: float = 0.3
    difference_threshold: float = 0.1
    semantic_comparison: bool = True
    structural_comparison: bool = True
    legal_concept_comparison: bool = True
    max_comparison_pairs: int = 10
    detailed_diff_analysis: bool = True
    confidence_scoring: bool = True


@dataclass
class ConceptExtractorConfig:
    """Configuration for legal concept extractor."""
    max_input_length: int = 75000
    min_concept_importance: float = 0.4
    max_concepts_extracted: int = 100
    relationship_mapping: bool = True
    context_window_size: int = 200
    legal_taxonomy_matching: bool = True
    importance_scoring: bool = True
    concept_categorization: bool = True
    related_terms_extraction: bool = True


@dataclass
class OrchestratorConfig:
    """Configuration for AI orchestrator."""
    max_concurrent_requests: int = 5
    load_balancing: bool = True
    fallback_strategy: str = "cascade"  # cascade, parallel, best_effort
    performance_monitoring: bool = True
    auto_scaling: bool = False
    health_check_interval: int = 60
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_batch_processing: bool = True
    batch_size: int = 10
    parallel_processing: bool = True
    max_workers: int = 3
    memory_optimization: bool = True
    response_compression: bool = True
    streaming_responses: bool = False
    timeout_monitoring: bool = True


@dataclass
class LegalOptimizationConfig:
    """Configuration for Indonesian legal document optimization."""
    language_model: str = "id"
    legal_terminology_boost: bool = True
    pasal_structure_recognition: bool = True
    ayat_analysis: bool = True
    bab_hierarchy_mapping: bool = True
    sanction_pattern_matching: bool = True
    definition_extraction_patterns: List[str] = field(default_factory=lambda: [
        r"yang dimaksud dengan",
        r"dalam.*ini.*adalah",
        r"pengertian.*meliputi"
    ])
    legal_entity_recognition: bool = True
    citation_analysis: bool = True
    regulatory_reference_mapping: bool = True


class AIConfig:
    """
    Main AI configuration class that aggregates all AI service configurations.

    Provides centralized access to all AI-related settings and supports
    environment-based configuration overrides.
    """

    def __init__(self,
                 model_config: Optional[ModelConfig] = None,
                 cache_config: Optional[CacheConfig] = None,
                 prompt_config: Optional[PromptConfig] = None):
        """
        Initialize AI configuration.

        Args:
            model_config: Model configuration (uses defaults if None)
            cache_config: Cache configuration (uses defaults if None)
            prompt_config: Prompt configuration (uses defaults if None)
        """
        self.model = model_config or ModelConfig()
        self.cache = cache_config or CacheConfig()
        self.prompt = prompt_config or PromptConfig()

        # Service-specific configurations
        self.summarizer = SummarizerConfig()
        self.qa_engine = QAEngineConfig()
        self.analyzer = AnalyzerConfig()
        self.comparator = ComparatorConfig()
        self.concept_extractor = ConceptExtractorConfig()
        self.orchestrator = OrchestratorConfig()
        self.performance = PerformanceConfig()
        self.legal_optimization = LegalOptimizationConfig()

        # Apply environment overrides
        self._apply_environment_overrides()

        # Validate configuration
        self._validate_configuration()

    def _apply_environment_overrides(self) -> None:
        """Apply configuration overrides from environment variables."""
        # Model configuration overrides
        if os.getenv("AI_MODEL_NAME"):
            self.model.model_name = os.getenv("AI_MODEL_NAME")

        if os.getenv("AI_TEMPERATURE"):
            self.model.temperature = float(os.getenv("AI_TEMPERATURE"))

        if os.getenv("AI_MAX_TOKENS"):
            self.model.max_tokens = int(os.getenv("AI_MAX_TOKENS"))

        # Cache configuration overrides
        if os.getenv("AI_CACHE_ENABLED"):
            self.cache.enabled = os.getenv("AI_CACHE_ENABLED").lower() == "true"

        if os.getenv("AI_CACHE_TTL_HOURS"):
            self.cache.ttl_hours = int(os.getenv("AI_CACHE_TTL_HOURS"))

        # Performance configuration overrides
        if os.getenv("AI_MAX_WORKERS"):
            self.performance.max_workers = int(os.getenv("AI_MAX_WORKERS"))

        if os.getenv("AI_BATCH_SIZE"):
            self.performance.batch_size = int(os.getenv("AI_BATCH_SIZE"))

    def _validate_configuration(self) -> None:
        """Validate configuration values."""
        # Validate model configuration
        if self.model.temperature < 0 or self.model.temperature > 2:
            raise ValueError("Model temperature must be between 0 and 2")

        if self.model.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")

        # Validate cache configuration
        if self.cache.max_size <= 0:
            raise ValueError("Cache max size must be positive")

        if self.cache.ttl_hours <= 0:
            raise ValueError("Cache TTL must be positive")

        # Validate performance configuration
        if self.performance.max_workers <= 0:
            raise ValueError("Max workers must be positive")

        if self.performance.batch_size <= 0:
            raise ValueError("Batch size must be positive")

    def get_model_config_dict(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            "provider": self.model.provider.value,
            "model_name": self.model.model_name,
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
            "top_p": self.model.top_p,
            "top_k": self.model.top_k,
            "timeout_seconds": self.model.timeout_seconds
        }

    def get_service_config(self, service_name: str) -> Any:
        """
        Get configuration for specific service.

        Args:
            service_name: Name of the service (summarizer, qa_engine, etc.)

        Returns:
            Configuration object for the specified service
        """
        service_configs = {
            "summarizer": self.summarizer,
            "qa_engine": self.qa_engine,
            "analyzer": self.analyzer,
            "comparator": self.comparator,
            "concept_extractor": self.concept_extractor,
            "orchestrator": self.orchestrator
        }

        if service_name not in service_configs:
            raise ValueError(f"Unknown service: {service_name}")

        return service_configs[service_name]

    def is_legal_optimization_enabled(self) -> bool:
        """Check if Indonesian legal document optimization is enabled."""
        return (
            self.legal_optimization.legal_terminology_boost and
            self.legal_optimization.pasal_structure_recognition and
            self.legal_optimization.language_model == "id"
        )

    def get_cache_key_prefix(self, service_name: str) -> str:
        """Generate cache key prefix for service."""
        return f"ai_{service_name}_{self.model.model_name}"

    def get_timeout_config(self, service_name: str) -> int:
        """Get timeout configuration for specific service."""
        base_timeout = self.model.timeout_seconds

        # Service-specific timeout multipliers
        multipliers = {
            "summarizer": 1.5,
            "qa_engine": 1.0,
            "analyzer": 2.0,
            "comparator": 2.5,
            "concept_extractor": 2.0,
            "orchestrator": 1.0
        }

        multiplier = multipliers.get(service_name, 1.0)
        return int(base_timeout * multiplier)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": {
                "provider": self.model.provider.value,
                "model_name": self.model.model_name,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens
            },
            "cache": {
                "enabled": self.cache.enabled,
                "max_size": self.cache.max_size,
                "ttl_hours": self.cache.ttl_hours
            },
            "performance": {
                "max_workers": self.performance.max_workers,
                "batch_size": self.performance.batch_size,
                "parallel_processing": self.performance.parallel_processing
            },
            "legal_optimization": {
                "language_model": self.legal_optimization.language_model,
                "pasal_structure_recognition": self.legal_optimization.pasal_structure_recognition,
                "legal_terminology_boost": self.legal_optimization.legal_terminology_boost
            }
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AIConfig':
        """Create configuration from dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        cache_config = CacheConfig(**config_dict.get("cache", {}))
        prompt_config = PromptConfig(**config_dict.get("prompt", {}))

        return cls(
            model_config=model_config,
            cache_config=cache_config,
            prompt_config=prompt_config
        )

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"AIConfig("
            f"model={self.model.model_name}, "
            f"cache_enabled={self.cache.enabled}, "
            f"legal_optimization={self.is_legal_optimization_enabled()}"
            f")"
        )


# Default configuration instance
default_ai_config = AIConfig()

# Configuration factory functions
def create_development_config() -> AIConfig:
    """Create configuration optimized for development."""
    model_config = ModelConfig(
        temperature=0.5,
        max_tokens=4096,
        timeout_seconds=15
    )

    cache_config = CacheConfig(
        enabled=True,
        max_size=100,
        ttl_hours=1
    )

    return AIConfig(model_config, cache_config)


def create_production_config() -> AIConfig:
    """Create configuration optimized for production."""
    model_config = ModelConfig(
        temperature=0.3,
        max_tokens=8192,
        timeout_seconds=30,
        retry_attempts=3
    )

    cache_config = CacheConfig(
        enabled=True,
        max_size=2000,
        ttl_hours=24,
        use_persistent_cache=True
    )

    return AIConfig(model_config, cache_config)


def create_testing_config() -> AIConfig:
    """Create configuration optimized for testing."""
    model_config = ModelConfig(
        model_name="mock-model",
        temperature=0.0,
        max_tokens=1024,
        timeout_seconds=5
    )

    cache_config = CacheConfig(
        enabled=False
    )

    return AIConfig(model_config, cache_config)
