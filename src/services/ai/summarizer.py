"""
Summarizer Service
Specialized AI service for legal document summarization

This module provides intelligent summarization capabilities specifically optimized
for Indonesian legal documents, supporting multiple summary types and formats.

Author: Refactored Architecture
Purpose: Single responsibility document summarization using AI
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from src.config.ai_config import AIConfig, SummaryType
from src.utils.prompt_manager import PromptManager, PromptType


@dataclass
class SummaryRequest:
    """Request for document summarization."""
    text: str
    summary_type: SummaryType = SummaryType.DETAILED
    max_length: Optional[int] = None
    include_key_points: bool = True
    preserve_structure: bool = True
    legal_focus: bool = True
    custom_instructions: Optional[str] = None


@dataclass
class SummaryResult:
    """Result of document summarization."""
    success: bool
    summary: str
    summary_type: SummaryType
    key_points: List[str]
    confidence: float
    processing_time: float
    word_count: int
    metadata: Dict[str, Any]
    error: str = ""
    cached: bool = False


class SummarizerService:
    """
    AI-powered document summarizer service.

    Provides intelligent summarization capabilities for Indonesian legal documents
    with support for multiple summary types and customizable parameters.
    """

    def __init__(self, config: Optional[AIConfig] = None):
        """
        Initialize summarizer service.

        Args:
            config: AI configuration (uses default if None)
        """
        self.config = config or AIConfig()
        self.prompt_manager = PromptManager()
        self.logger = logging.getLogger(__name__)

        # Initialize AI model if available
        if GEMINI_AVAILABLE:
            self._initialize_model()
        else:
            self.logger.warning("Gemini AI not available. Install google-generativeai package.")
            self.model = None

        # Service statistics
        self.stats = {
            'total_requests': 0,
            'successful_summaries': 0,
            'cache_hits': 0,
            'average_processing_time': 0.0,
            'summary_type_counts': {st.value: 0 for st in SummaryType}
        }

    def _initialize_model(self) -> None:
        """Initialize Gemini AI model."""
        try:
            if self.config.model.api_key:
                genai.configure(api_key=self.config.model.api_key)

                generation_config = {
                    "temperature": self.config.model.temperature,
                    "top_p": self.config.model.top_p,
                    "top_k": self.config.model.top_k,
                    "max_output_tokens": self.config.model.max_tokens,
                }

                self.model = genai.GenerativeModel(
                    model_name=self.config.model.model_name,
                    generation_config=generation_config
                )

                self.logger.info(f"Initialized Gemini model: {self.config.model.model_name}")
            else:
                self.logger.error("No API key provided for Gemini")
                self.model = None

        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model: {e}")
            self.model = None

    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        """
        Generate summary for the given text.

        Args:
            request: SummaryRequest with text and parameters

        Returns:
            SummaryResult with generated summary and metadata
        """
        start_time = time.time()
        self.stats['total_requests'] += 1

        # Validate input
        validation_result = self._validate_input(request)
        if not validation_result.success:
            return validation_result

        if not GEMINI_AVAILABLE or not self.model:
            return SummaryResult(
                success=False,
                summary="",
                summary_type=request.summary_type,
                key_points=[],
                confidence=0.0,
                processing_time=0.0,
                word_count=0,
                metadata={},
                error="AI model not available"
            )

        try:
            # Generate summary using AI
            summary_text, confidence = await self._generate_summary_with_ai(request)

            # Extract key points if requested
            key_points = []
            if request.include_key_points and summary_text:
                key_points = await self._extract_key_points(summary_text, request.text)

            # Calculate metrics
            processing_time = time.time() - start_time
            word_count = len(summary_text.split()) if summary_text else 0

            # Update statistics
            if summary_text:
                self.stats['successful_summaries'] += 1
                self.stats['summary_type_counts'][request.summary_type.value] += 1

            self._update_average_processing_time(processing_time)

            return SummaryResult(
                success=bool(summary_text),
                summary=summary_text,
                summary_type=request.summary_type,
                key_points=key_points,
                confidence=confidence,
                processing_time=processing_time,
                word_count=word_count,
                metadata={
                    'input_length': len(request.text),
                    'compression_ratio': word_count / len(request.text.split()) if request.text else 0,
                    'model_used': self.config.model.model_name,
                    'legal_focus': request.legal_focus,
                    'preserve_structure': request.preserve_structure
                },
                error="" if summary_text else "Failed to generate summary"
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Summarization failed: {str(e)}"
            self.logger.error(error_msg)

            return SummaryResult(
                success=False,
                summary="",
                summary_type=request.summary_type,
                key_points=[],
                confidence=0.0,
                processing_time=processing_time,
                word_count=0,
                metadata={'input_length': len(request.text)},
                error=error_msg
            )

    def _validate_input(self, request: SummaryRequest) -> SummaryResult:
        """Validate summarization request."""
        if not request.text or not request.text.strip():
            return SummaryResult(
                success=False,
                summary="",
                summary_type=request.summary_type,
                key_points=[],
                confidence=0.0,
                processing_time=0.0,
                word_count=0,
                metadata={},
                error="Empty or invalid text provided"
            )

        # Check text length limits
        max_length = self.config.summarizer.max_input_length
        if len(request.text) > max_length:
            return SummaryResult(
                success=False,
                summary="",
                summary_type=request.summary_type,
                key_points=[],
                confidence=0.0,
                processing_time=0.0,
                word_count=0,
                metadata={'input_length': len(request.text)},
                error=f"Text too long. Maximum {max_length} characters allowed."
            )

        # Validation passed
        return SummaryResult(
            success=True,
            summary="",
            summary_type=request.summary_type,
            key_points=[],
            confidence=1.0,
            processing_time=0.0,
            word_count=0,
            metadata={}
        )

    async def _generate_summary_with_ai(self, request: SummaryRequest) -> Tuple[str, float]:
        """Generate summary using AI model."""
        try:
            # Select appropriate prompt based on summary type
            prompt_type = self._get_prompt_type_for_summary(request.summary_type)

            # Get formatted prompt
            prompt = self.prompt_manager.get_prompt(prompt_type, text=request.text)

            # Add custom instructions if provided
            if request.custom_instructions:
                prompt += f"\n\nINSTRUKSI TAMBAHAN:\n{request.custom_instructions}\n"

            # Generate response
            response = await asyncio.create_task(
                self._call_ai_model(prompt)
            )

            if response and len(response.strip()) > 10:
                confidence = self._calculate_confidence(response, request.text)
                return response.strip(), confidence
            else:
                return "", 0.0

        except Exception as e:
            self.logger.error(f"AI summary generation failed: {e}")
            return "", 0.0

    def _get_prompt_type_for_summary(self, summary_type: SummaryType) -> PromptType:
        """Map summary type to prompt type."""
        mapping = {
            SummaryType.BRIEF: PromptType.SUMMARIZE_BRIEF,
            SummaryType.DETAILED: PromptType.SUMMARIZE_DETAILED,
            SummaryType.EXECUTIVE: PromptType.SUMMARIZE_EXECUTIVE,
            SummaryType.TECHNICAL: PromptType.SUMMARIZE_TECHNICAL,
            SummaryType.LEGAL: PromptType.SUMMARIZE_LEGAL
        }
        return mapping.get(summary_type, PromptType.SUMMARIZE_DETAILED)

    async def _call_ai_model(self, prompt: str) -> str:
        """Call AI model with prompt."""
        try:
            response = self.model.generate_content(prompt)
            return response.text if response and response.text else ""
        except Exception as e:
            self.logger.error(f"AI model call failed: {e}")
            return ""

    def _calculate_confidence(self, summary: str, original_text: str) -> float:
        """Calculate confidence score for generated summary."""
        try:
            # Basic confidence metrics
            summary_words = len(summary.split())
            original_words = len(original_text.split())

            # Length appropriateness (not too short, not too long)
            length_ratio = summary_words / original_words if original_words > 0 else 0
            length_score = 1.0 if 0.05 <= length_ratio <= 0.3 else 0.5

            # Content quality indicators
            has_structure = any(keyword in summary.lower() for keyword in [
                'pasal', 'bab', 'ayat', 'ketentuan', 'sanksi', 'tujuan'
            ])
            structure_score = 0.2 if has_structure else 0.0

            # Completeness check
            completeness_score = 0.3 if summary_words >= 50 else summary_words / 50 * 0.3

            # Indonesian language quality
            indonesian_indicators = ['yang', 'dengan', 'dalam', 'pada', 'untuk', 'adalah']
            indonesian_score = sum(1 for word in indonesian_indicators if word in summary.lower()) / len(indonesian_indicators) * 0.2

            # Combine scores
            total_confidence = length_score * 0.4 + structure_score + completeness_score + indonesian_score

            return min(total_confidence, 1.0)

        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence

    async def _extract_key_points(self, summary: str, original_text: str) -> List[str]:
        """Extract key points from summary or original text."""
        try:
            # Simple key point extraction based on patterns
            key_points = []

            # Look for numbered points
            import re
            numbered_points = re.findall(r'(?:^|\n)\s*\d+\.?\s*([^\n]+)', summary, re.MULTILINE)
            key_points.extend(numbered_points[:5])  # Max 5 numbered points

            # Look for bullet points
            bullet_points = re.findall(r'(?:^|\n)\s*[-•]\s*([^\n]+)', summary, re.MULTILINE)
            key_points.extend(bullet_points[:3])  # Max 3 bullet points

            # If no structured points found, extract sentences with legal keywords
            if not key_points:
                sentences = summary.split('. ')
                legal_keywords = ['pasal', 'ketentuan', 'sanksi', 'kewajiban', 'hak', 'prosedur']

                for sentence in sentences[:10]:  # Check first 10 sentences
                    if any(keyword in sentence.lower() for keyword in legal_keywords):
                        key_points.append(sentence.strip())
                        if len(key_points) >= 5:
                            break

            # Clean and limit key points
            cleaned_points = []
            for point in key_points:
                cleaned = point.strip().rstrip('.')
                if cleaned and len(cleaned) > 10:  # Meaningful length
                    cleaned_points.append(cleaned)

            return cleaned_points[:self.config.summarizer.max_key_points]

        except Exception as e:
            self.logger.warning(f"Key point extraction failed: {e}")
            return []

    def _update_average_processing_time(self, processing_time: float) -> None:
        """Update average processing time statistic."""
        current_avg = self.stats['average_processing_time']
        total_requests = self.stats['total_requests']

        # Calculate new average
        self.stats['average_processing_time'] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )

    async def summarize_brief(self, text: str, max_length: int = 200) -> SummaryResult:
        """Generate brief summary (convenience method)."""
        request = SummaryRequest(
            text=text,
            summary_type=SummaryType.BRIEF,
            max_length=max_length,
            include_key_points=True
        )
        return await self.summarize(request)

    async def summarize_detailed(self, text: str, max_length: int = 800) -> SummaryResult:
        """Generate detailed summary (convenience method)."""
        request = SummaryRequest(
            text=text,
            summary_type=SummaryType.DETAILED,
            max_length=max_length,
            include_key_points=True,
            preserve_structure=True
        )
        return await self.summarize(request)

    async def summarize_executive(self, text: str, max_length: int = 400) -> SummaryResult:
        """Generate executive summary (convenience method)."""
        request = SummaryRequest(
            text=text,
            summary_type=SummaryType.EXECUTIVE,
            max_length=max_length,
            include_key_points=True,
            legal_focus=True
        )
        return await self.summarize(request)

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        success_rate = (
            self.stats['successful_summaries'] / self.stats['total_requests']
            if self.stats['total_requests'] > 0 else 0
        )

        return {
            **self.stats,
            'success_rate': success_rate,
            'service_status': 'available' if self.model else 'unavailable',
            'model_name': self.config.model.model_name if self.model else None
        }

    def reset_stats(self) -> None:
        """Reset service statistics."""
        self.stats = {
            'total_requests': 0,
            'successful_summaries': 0,
            'cache_hits': 0,
            'average_processing_time': 0.0,
            'summary_type_counts': {st.value: 0 for st in SummaryType}
        }

    def supports_summary_type(self, summary_type: SummaryType) -> bool:
        """Check if summary type is supported."""
        return summary_type in [
            SummaryType.BRIEF,
            SummaryType.DETAILED,
            SummaryType.EXECUTIVE,
            SummaryType.TECHNICAL,
            SummaryType.LEGAL
        ]

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the summarizer service."""
        return {
            'name': 'Legal Document Summarizer',
            'description': 'AI-powered summarization for Indonesian legal documents',
            'supported_summary_types': [st.value for st in SummaryType],
            'features': [
                'Multiple summary types',
                'Key point extraction',
                'Indonesian legal optimization',
                'Structure preservation',
                'Confidence scoring',
                'Performance monitoring'
            ],
            'configuration': {
                'max_input_length': self.config.summarizer.max_input_length,
                'legal_focus': self.config.summarizer.legal_focus,
                'include_key_points': self.config.summarizer.include_key_points,
                'model_name': self.config.model.model_name
            },
            'status': 'available' if self.model else 'unavailable'
        }

    def __repr__(self) -> str:
        """String representation of summarizer service."""
        return f"SummarizerService(model={self.config.model.model_name}, available={self.model is not None})"
