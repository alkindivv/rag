"""
AI Orchestrator Service
Coordinates and manages all AI services with intelligent routing and optimization

This module provides centralized coordination of AI services including summarization,
question-answering, analysis, and comparison. Features load balancing, caching,
circuit breaker patterns, and Indonesian legal document optimization.

Author: Refactored Architecture
Purpose: Single responsibility AI service coordination and optimization
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import json

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from src.config.ai_config import AIConfig, AnalysisType, SummaryType
from src.services.ai.summarizer import SummarizerService, SummaryRequest, SummaryResult
from src.services.ai.qa_engine import QAEngine, QARequest, QAResult
from src.services.ai.analyzer import AnalyzerService, AnalysisRequest, AnalysisResult
from src.services.ai.comparator import ComparatorService, ComparisonRequest, ComparisonResult


class ServiceType(Enum):
    """Types of AI services available."""
    SUMMARIZER = "summarizer"
    QA_ENGINE = "qa_engine"
    ANALYZER = "analyzer"
    COMPARATOR = "comparator"


class RoutingStrategy(Enum):
    """Strategies for routing requests to services."""
    ROUND_ROBIN = "round_robin"
    LOAD_BASED = "load_based"
    PERFORMANCE_BASED = "performance_based"
    RANDOM = "random"


class ServiceHealth(Enum):
    """Health status of services."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"


@dataclass
class ServiceMetrics:
    """Metrics for individual services."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: float = 0.0
    health_status: ServiceHealth = ServiceHealth.HEALTHY
    circuit_breaker_open: bool = False
    circuit_breaker_failures: int = 0


@dataclass
class OrchestratorRequest:
    """Request to the AI orchestrator."""
    service_type: ServiceType
    request_data: Dict[str, Any]
    priority: int = 1
    timeout: Optional[float] = None
    fallback_enabled: bool = True
    cache_enabled: bool = True


@dataclass
class OrchestratorResult:
    """Result from AI orchestrator."""
    success: bool
    service_used: ServiceType
    result_data: Dict[str, Any]
    processing_time: float
    cached: bool = False
    fallback_used: bool = False
    metadata: Dict[str, Any] = None
    error: str = ""

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CircuitBreaker:
    """Circuit breaker for service resilience."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds before attempting to reset circuit
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open

    def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e

    def on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"

    def on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class AIOrchestrator:
    """
    AI services orchestrator.

    Provides centralized coordination, load balancing, caching, and resilience
    patterns for all AI services with Indonesian legal document optimization.
    """

    def __init__(self, config: Optional[AIConfig] = None):
        """
        Initialize AI orchestrator.

        Args:
            config: AI configuration (uses default if None)
        """
        self.config = config or AIConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize services
        self.services = {}
        self.service_metrics = {}
        self.circuit_breakers = {}
        self._initialize_services()

        # Routing and load balancing
        self.routing_strategy = RoutingStrategy(self.config.orchestrator.fallback_strategy)
        self.service_instances = defaultdict(list)

        # Cache for results
        self.result_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Overall orchestrator statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'service_distribution': {st.value: 0 for st in ServiceType}
        }

        self.logger.info("AI Orchestrator initialized successfully")

    def _initialize_services(self) -> None:
        """Initialize all AI services."""
        try:
            # Initialize services with shared config
            self.services[ServiceType.SUMMARIZER] = SummarizerService(self.config)
            self.services[ServiceType.QA_ENGINE] = QAEngine(self.config)
            self.services[ServiceType.ANALYZER] = AnalyzerService(self.config)
            self.services[ServiceType.COMPARATOR] = ComparatorService(self.config)

            # Initialize metrics and circuit breakers for each service
            for service_type in ServiceType:
                self.service_metrics[service_type] = ServiceMetrics()
                self.circuit_breakers[service_type] = CircuitBreaker(
                    failure_threshold=self.config.orchestrator.circuit_breaker_threshold,
                    reset_timeout=60
                )

            self.logger.info(f"Initialized {len(self.services)} AI services")

        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            raise RuntimeError(f"Service initialization failed: {e}")

    async def process_request(self, request: OrchestratorRequest) -> OrchestratorResult:
        """
        Process request through appropriate AI service.

        Args:
            request: OrchestratorRequest with service type and data

        Returns:
            OrchestratorResult with processed data
        """
        start_time = time.time()
        self.stats['total_requests'] += 1

        try:
            # Check cache first if enabled
            if request.cache_enabled:
                cached_result = self._check_cache(request)
                if cached_result:
                    self.cache_hits += 1
                    processing_time = time.time() - start_time
                    self._update_average_response_time(processing_time)
                    return cached_result

            self.cache_misses += 1

            # Route request to appropriate service
            result = await self._route_request(request)

            # Cache result if successful
            if result.success and request.cache_enabled:
                self._cache_result(request, result)

            # Update statistics
            self.stats['successful_requests'] += 1
            self.stats['service_distribution'][request.service_type.value] += 1

            processing_time = time.time() - start_time
            result.processing_time = processing_time
            self._update_average_response_time(processing_time)

            return result

        except Exception as e:
            self.stats['failed_requests'] += 1
            processing_time = time.time() - start_time
            error_msg = f"Request processing failed: {str(e)}"
            self.logger.error(error_msg)

            return OrchestratorResult(
                success=False,
                service_used=request.service_type,
                result_data={},
                processing_time=processing_time,
                error=error_msg
            )

    async def _route_request(self, request: OrchestratorRequest) -> OrchestratorResult:
        """Route request to appropriate service with resilience patterns."""
        service_type = request.service_type

        # Check service health
        if not self._is_service_healthy(service_type):
            if request.fallback_enabled:
                return await self._handle_fallback(request)
            else:
                raise Exception(f"Service {service_type.value} is unavailable")

        # Get service instance
        service = self.services[service_type]
        circuit_breaker = self.circuit_breakers[service_type]

        try:
            # Call service with circuit breaker protection
            if service_type == ServiceType.SUMMARIZER:
                result_data = await circuit_breaker.call(
                    self._call_summarizer, service, request.request_data
                )
            elif service_type == ServiceType.QA_ENGINE:
                result_data = await circuit_breaker.call(
                    self._call_qa_engine, service, request.request_data
                )
            elif service_type == ServiceType.ANALYZER:
                result_data = await circuit_breaker.call(
                    self._call_analyzer, service, request.request_data
                )
            elif service_type == ServiceType.COMPARATOR:
                result_data = await circuit_breaker.call(
                    self._call_comparator, service, request.request_data
                )
            else:
                raise ValueError(f"Unknown service type: {service_type}")

            # Update service metrics
            self._update_service_metrics(service_type, success=True)

            return OrchestratorResult(
                success=True,
                service_used=service_type,
                result_data=result_data,
                processing_time=0.0  # Will be set by caller
            )

        except Exception as e:
            # Update service metrics
            self._update_service_metrics(service_type, success=False)

            if request.fallback_enabled:
                return await self._handle_fallback(request)
            else:
                raise e

    async def _call_summarizer(self, service: SummarizerService, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call summarizer service."""
        summary_request = SummaryRequest(
            text=request_data.get('text', ''),
            summary_type=SummaryType(request_data.get('summary_type', 'detailed')),
            max_length=request_data.get('max_length'),
            include_key_points=request_data.get('include_key_points', True),
            preserve_structure=request_data.get('preserve_structure', True),
            legal_focus=request_data.get('legal_focus', True)
        )

        result = await service.summarize(summary_request)
        return {
            'success': result.success,
            'summary': result.summary,
            'key_points': result.key_points,
            'confidence': result.confidence,
            'word_count': result.word_count,
            'metadata': result.metadata,
            'error': result.error
        }

    async def _call_qa_engine(self, service: QAEngine, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call QA engine service."""
        qa_request = QARequest(
            question=request_data.get('question', ''),
            context=request_data.get('context', ''),
            include_citations=request_data.get('include_citations', True),
            legal_focus=request_data.get('legal_focus', True)
        )

        result = await service.answer_question(qa_request)
        return {
            'success': result.success,
            'answer': result.answer,
            'question_type': result.question_type.value,
            'confidence': result.confidence,
            'citations': [
                {
                    'text': c.text,
                    'source_section': c.source_section,
                    'relevance_score': c.relevance_score
                } for c in result.citations
            ],
            'follow_up_questions': result.follow_up_questions,
            'metadata': result.metadata,
            'error': result.error
        }

    async def _call_analyzer(self, service: AnalyzerService, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call analyzer service."""
        analysis_request = AnalysisRequest(
            text=request_data.get('text', ''),
            analysis_types=[AnalysisType(at) for at in request_data.get('analysis_types', ['structure'])],
            include_relationships=request_data.get('include_relationships', True),
            legal_focus=request_data.get('legal_focus', True)
        )

        result = await service.analyze(analysis_request)
        return {
            'success': result.success,
            'structure_elements': [
                {
                    'element_type': e.element_type,
                    'number': e.number,
                    'title': e.title,
                    'level': e.level,
                    'confidence': e.confidence
                } for e in result.structure_elements
            ],
            'definitions': [
                {
                    'term': d.term,
                    'definition': d.definition,
                    'confidence': d.confidence,
                    'importance_score': d.importance_score
                } for d in result.definitions
            ],
            'sanctions': [
                {
                    'sanction_type': s.sanction_type,
                    'description': s.description,
                    'severity': s.severity,
                    'confidence': s.confidence
                } for s in result.sanctions
            ],
            'concepts': [
                {
                    'term': c.term,
                    'category': c.category,
                    'importance_score': c.importance_score,
                    'confidence': c.confidence
                } for c in result.concepts
            ],
            'overall_confidence': result.overall_confidence,
            'metadata': result.metadata,
            'error': result.error
        }

    async def _call_comparator(self, service: ComparatorService, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call comparator service."""
        comparison_request = ComparisonRequest(
            document1=request_data.get('document1', ''),
            document2=request_data.get('document2', ''),
            document1_id=request_data.get('document1_id', 'doc1'),
            document2_id=request_data.get('document2_id', 'doc2'),
            include_differences=request_data.get('include_differences', True),
            include_similarities=request_data.get('include_similarities', True),
            legal_focus=request_data.get('legal_focus', True)
        )

        result = await service.compare(comparison_request)
        return {
            'success': result.success,
            'overall_similarity_score': result.overall_similarity_score,
            'similarity_level': result.similarity_level.value,
            'similarities': [
                {
                    'content': s.content,
                    'similarity_score': s.similarity_score,
                    'category': s.category,
                    'confidence': s.confidence
                } for s in result.similarities
            ],
            'differences': [
                {
                    'doc1_content': d.doc1_content,
                    'doc2_content': d.doc2_content,
                    'difference_type': d.difference_type,
                    'significance': d.significance,
                    'confidence': d.confidence
                } for d in result.differences
            ],
            'summary': result.summary,
            'metadata': result.metadata,
            'error': result.error
        }

    async def _handle_fallback(self, request: OrchestratorRequest) -> OrchestratorResult:
        """Handle fallback when primary service fails."""
        try:
            # Simple fallback strategy - try to provide basic response
            fallback_result = {
                'success': True,
                'message': f"Service {request.service_type.value} unavailable, fallback response provided",
                'fallback': True
            }

            return OrchestratorResult(
                success=True,
                service_used=request.service_type,
                result_data=fallback_result,
                processing_time=0.0,
                fallback_used=True
            )

        except Exception as e:
            raise Exception(f"Fallback also failed: {e}")

    def _is_service_healthy(self, service_type: ServiceType) -> bool:
        """Check if service is healthy."""
        metrics = self.service_metrics[service_type]
        circuit_breaker = self.circuit_breakers[service_type]

        return (
            metrics.health_status in [ServiceHealth.HEALTHY, ServiceHealth.DEGRADED] and
            circuit_breaker.state != "open"
        )

    def _update_service_metrics(self, service_type: ServiceType, success: bool) -> None:
        """Update metrics for a service."""
        metrics = self.service_metrics[service_type]
        circuit_breaker = self.circuit_breakers[service_type]

        metrics.total_requests += 1
        metrics.last_request_time = time.time()

        if success:
            metrics.successful_requests += 1
            circuit_breaker.on_success()
        else:
            metrics.failed_requests += 1
            circuit_breaker.on_failure()

        # Update health status based on recent performance
        if metrics.total_requests > 0:
            success_rate = metrics.successful_requests / metrics.total_requests
            if success_rate >= 0.9:
                metrics.health_status = ServiceHealth.HEALTHY
            elif success_rate >= 0.7:
                metrics.health_status = ServiceHealth.DEGRADED
            else:
                metrics.health_status = ServiceHealth.UNHEALTHY

        metrics.circuit_breaker_open = circuit_breaker.state == "open"
        metrics.circuit_breaker_failures = circuit_breaker.failure_count

    def _check_cache(self, request: OrchestratorRequest) -> Optional[OrchestratorResult]:
        """Check if result is cached."""
        try:
            cache_key = self._generate_cache_key(request)

            if cache_key in self.result_cache:
                cached_data, timestamp = self.result_cache[cache_key]

                # Check if cache is still valid (24 hours)
                if time.time() - timestamp < self.config.cache.ttl_hours * 3600:
                    cached_result = OrchestratorResult(**cached_data)
                    cached_result.cached = True
                    return cached_result
                else:
                    # Remove expired cache entry
                    del self.result_cache[cache_key]

        except Exception as e:
            self.logger.warning(f"Cache check failed: {e}")

        return None

    def _cache_result(self, request: OrchestratorRequest, result: OrchestratorResult) -> None:
        """Cache result for future use."""
        try:
            if not self.config.cache.enabled:
                return

            cache_key = self._generate_cache_key(request)

            # Implement LRU eviction if cache is full
            if len(self.result_cache) >= self.config.cache.max_size:
                # Remove oldest entry
                oldest_key = min(self.result_cache.keys(),
                               key=lambda k: self.result_cache[k][1])
                del self.result_cache[oldest_key]

            # Convert result to dict for caching
            cache_data = {
                'success': result.success,
                'service_used': result.service_used,
                'result_data': result.result_data,
                'processing_time': result.processing_time,
                'metadata': result.metadata,
                'error': result.error
            }

            self.result_cache[cache_key] = (cache_data, time.time())

        except Exception as e:
            self.logger.warning(f"Result caching failed: {e}")

    def _generate_cache_key(self, request: OrchestratorRequest) -> str:
        """Generate cache key for request."""
        import hashlib

        key_data = {
            'service_type': request.service_type.value,
            'request_data': request.request_data
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _update_average_response_time(self, response_time: float) -> None:
        """Update average response time."""
        current_avg = self.stats['average_response_time']
        total_requests = self.stats['total_requests']

        self.stats['average_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )

        # Update cache hit rate
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            self.stats['cache_hit_rate'] = self.cache_hits / total_cache_requests

    # Convenience methods for different AI operations
    async def summarize(self, text: str, summary_type: str = "detailed", **kwargs) -> OrchestratorResult:
        """Convenience method for document summarization."""
        request = OrchestratorRequest(
            service_type=ServiceType.SUMMARIZER,
            request_data={
                'text': text,
                'summary_type': summary_type,
                **kwargs
            }
        )
        return await self.process_request(request)

    async def answer_question(self, question: str, context: str, **kwargs) -> OrchestratorResult:
        """Convenience method for question answering."""
        request = OrchestratorRequest(
            service_type=ServiceType.QA_ENGINE,
            request_data={
                'question': question,
                'context': context,
                **kwargs
            }
        )
        return await self.process_request(request)

    async def analyze_document(self, text: str, analysis_types: List[str] = None, **kwargs) -> OrchestratorResult:
        """Convenience method for document analysis."""
        if analysis_types is None:
            analysis_types = ['structure', 'definitions']

        request = OrchestratorRequest(
            service_type=ServiceType.ANALYZER,
            request_data={
                'text': text,
                'analysis_types': analysis_types,
                **kwargs
            }
        )
        return await self.process_request(request)

    async def compare_documents(self, doc1: str, doc2: str, doc1_id: str = "doc1",
                              doc2_id: str = "doc2", **kwargs) -> OrchestratorResult:
        """Convenience method for document comparison."""
        request = OrchestratorRequest(
            service_type=ServiceType.COMPARATOR,
            request_data={
                'document1': doc1,
                'document2': doc2,
                'document1_id': doc1_id,
                'document2_id': doc2_id,
                **kwargs
            }
        )
        return await self.process_request(request)

    def get_service_health(self) -> Dict[str, Any]:
        """Get health status of all services."""
        health_report = {}

        for service_type, metrics in self.service_metrics.items():
            health_report[service_type.value] = {
                'health_status': metrics.health_status.value,
                'total_requests': metrics.total_requests,
                'success_rate': (
                    metrics.successful_requests / metrics.total_requests
                    if metrics.total_requests > 0 else 0
                ),
                'average_response_time': metrics.average_response_time,
                'circuit_breaker_open': metrics.circuit_breaker_open,
                'last_request_time': metrics.last_request_time
            }

        return health_report

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        success_rate = (
            self.stats['successful_requests'] / self.stats['total_requests']
            if self.stats['total_requests'] > 0 else 0
        )

        return {
            **self.stats,
            'success_rate': success_rate,
            'total_services': len(self.services),
            'healthy_services': sum(
                1 for metrics in self.service_metrics.values()
                if metrics.health_status == ServiceHealth.HEALTHY
            ),
            'cache_enabled': self.config.cache.enabled,
            'cache_size': len(self.result_cache)
        }

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'service_distribution': {st.value: 0 for st in ServiceType}
        }

        for service_type in ServiceType:
            self.service_metrics[service_type] = ServiceMetrics()

        self.cache_hits = 0
        self.cache_misses = 0
        self.result_cache.clear()

    def get_orchestrator_info(self) -> Dict[str, Any]:
        """Get information about the orchestrator."""
        return {
            'name': 'AI Services Orchestrator',
            'description': 'Centralized coordination of AI services for Indonesian legal documents',
            'available_services': [st.value for st in ServiceType],
            'features': [
                'Service coordination and routing',
                'Load balancing and health monitoring',
                'Circuit breaker patterns',
                'Intelligent caching',
                'Fallback strategies',
                'Performance monitoring',
                'Indonesian legal optimization'
            ],
            'configuration': {
                'max_concurrent_requests': self.config.orchestrator.max_concurrent_requests,
                'circuit_breaker_enabled': self.config.orchestrator.circuit_breaker_enabled,
                'cache_enabled': self.config.cache.enabled,
                'cache_ttl_hours': self.config.cache.ttl_hours,
                'routing_strategy': self.routing_strategy.value
            },
            'service_health': self.get_service_health(),
            'statistics': self.get_orchestrator_stats()
        }

    def __repr__(self) -> str:
        """String representation of orchestrator."""
        healthy_services = sum(
            1 for metrics in self.service_metrics.values()
            if metrics.health_status == ServiceHealth.HEALTHY
        )
        return f"AIOrchestrator(services={len(self.services)}, healthy={healthy_services})"
