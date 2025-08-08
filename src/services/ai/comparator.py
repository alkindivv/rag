"""
Comparator Service
Specialized AI service for legal document comparison and similarity analysis

This module provides intelligent document comparison capabilities specifically optimized
for Indonesian legal documents, including similarity scoring, difference analysis,
and structural comparison.

Author: Refactored Architecture
Purpose: Single responsibility document comparison using AI
"""

import logging
import time
import asyncio
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from difflib import SequenceMatcher
import hashlib

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from src.config.ai_config import AIConfig
from src.utils.prompt_manager import PromptManager, PromptType


class ComparisonType(Enum):
    """Types of document comparison."""
    STRUCTURAL = "structural"         # Compare document structure
    SEMANTIC = "semantic"            # Compare meaning and content
    LEGAL_CONCEPTS = "legal_concepts" # Compare legal concepts
    COMPREHENSIVE = "comprehensive"   # Full comparison
    SIMILARITY_ONLY = "similarity_only" # Just similarity score


class SimilarityLevel(Enum):
    """Levels of document similarity."""
    IDENTICAL = "identical"      # 95-100% similar
    VERY_HIGH = "very_high"     # 80-94% similar
    HIGH = "high"               # 60-79% similar
    MODERATE = "moderate"       # 40-59% similar
    LOW = "low"                 # 20-39% similar
    VERY_LOW = "very_low"       # 0-19% similar


@dataclass
class ComparisonRequest:
    """Request for document comparison."""
    document1: str
    document2: str
    document1_id: str
    document2_id: str
    comparison_type: ComparisonType = ComparisonType.COMPREHENSIVE
    include_differences: bool = True
    include_similarities: bool = True
    detailed_analysis: bool = True
    legal_focus: bool = True
    confidence_threshold: float = 0.6


@dataclass
class Similarity:
    """Identified similarity between documents."""
    content: str
    similarity_score: float
    category: str
    source_doc1: str
    source_doc2: str
    confidence: float


@dataclass
class Difference:
    """Identified difference between documents."""
    doc1_content: str
    doc2_content: str
    difference_type: str
    category: str
    significance: str
    confidence: float


@dataclass
class StructuralComparison:
    """Structural comparison result."""
    doc1_structure: Dict[str, int]
    doc2_structure: Dict[str, int]
    structural_similarity: float
    missing_in_doc1: List[str]
    missing_in_doc2: List[str]
    common_elements: List[str]


@dataclass
class ComparisonResult:
    """Result of document comparison."""
    success: bool
    document1_id: str
    document2_id: str
    comparison_type: ComparisonType
    overall_similarity_score: float
    similarity_level: SimilarityLevel
    similarities: List[Similarity]
    differences: List[Difference]
    structural_comparison: Optional[StructuralComparison]
    summary: str
    processing_time: float
    confidence: float
    metadata: Dict[str, Any]
    error: str = ""


class ComparatorService:
    """
    AI-powered document comparator service.

    Provides intelligent comparison capabilities for Indonesian legal documents
    with support for structural, semantic, and legal concept analysis.
    """

    def __init__(self, config: Optional[AIConfig] = None):
        """
        Initialize comparator service.

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

        # Load legal patterns for comparison
        self._load_legal_patterns()

        # Service statistics
        self.stats = {
            'total_comparisons': 0,
            'successful_comparisons': 0,
            'average_similarity_score': 0.0,
            'average_processing_time': 0.0,
            'comparison_type_counts': {ct.value: 0 for ct in ComparisonType},
            'similarity_level_counts': {sl.value: 0 for sl in SimilarityLevel}
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

    def _load_legal_patterns(self) -> None:
        """Load patterns for legal document comparison."""
        self.legal_patterns = {
            # Structure patterns
            'bab': re.compile(r'(?m)^(?:BAB|B[AI]B)\s+([IVXLC]+|\d+)', re.IGNORECASE),
            'pasal': re.compile(r'(?m)^Pasal\s+(\d+[A-Z]?)', re.IGNORECASE),
            'ayat': re.compile(r'\((\d+)\)', re.MULTILINE),

            # Legal concept patterns
            'definitions': re.compile(r'(?i)yang dimaksud dengan|pengertian.*adalah'),
            'sanctions': re.compile(r'(?i)dipidana|denda|sanksi|kurungan|penjara'),
            'obligations': re.compile(r'(?i)wajib|harus|berkewajiban'),
            'rights': re.compile(r'(?i)berhak|hak untuk|dapat.*melakukan'),
            'prohibitions': re.compile(r'(?i)dilarang|tidak boleh|tidak dapat'),

            # Reference patterns
            'law_references': re.compile(r'(?i)undang-undang.*nomor.*tahun'),
            'regulation_references': re.compile(r'(?i)peraturan.*nomor.*tahun'),
            'article_references': re.compile(r'(?i)pasal\s+\d+')
        }

    async def compare(self, request: ComparisonRequest) -> ComparisonResult:
        """
        Compare two legal documents.

        Args:
            request: ComparisonRequest with documents and parameters

        Returns:
            ComparisonResult with comparison analysis
        """
        start_time = time.time()
        self.stats['total_comparisons'] += 1

        # Validate input
        validation_result = self._validate_input(request)
        if not validation_result.success:
            return validation_result

        if not GEMINI_AVAILABLE or not self.model:
            return ComparisonResult(
                success=False,
                document1_id=request.document1_id,
                document2_id=request.document2_id,
                comparison_type=request.comparison_type,
                overall_similarity_score=0.0,
                similarity_level=SimilarityLevel.VERY_LOW,
                similarities=[],
                differences=[],
                structural_comparison=None,
                summary="",
                processing_time=0.0,
                confidence=0.0,
                metadata={},
                error="AI model not available"
            )

        try:
            # Perform different types of comparison based on request
            similarities = []
            differences = []
            structural_comparison = None

            # Calculate basic similarity score
            basic_similarity = self._calculate_basic_similarity(request.document1, request.document2)

            # Structural comparison
            if request.comparison_type in [ComparisonType.STRUCTURAL, ComparisonType.COMPREHENSIVE]:
                structural_comparison = self._compare_structure(request.document1, request.document2)

            # Semantic comparison using AI
            if request.comparison_type in [ComparisonType.SEMANTIC, ComparisonType.COMPREHENSIVE]:
                ai_similarities, ai_differences = await self._compare_with_ai(request)
                similarities.extend(ai_similarities)
                differences.extend(ai_differences)

            # Legal concept comparison
            if request.comparison_type in [ComparisonType.LEGAL_CONCEPTS, ComparisonType.COMPREHENSIVE]:
                concept_similarities, concept_differences = self._compare_legal_concepts(
                    request.document1, request.document2
                )
                similarities.extend(concept_similarities)
                differences.extend(concept_differences)

            # Calculate overall similarity score
            overall_similarity = self._calculate_overall_similarity(
                basic_similarity, similarities, structural_comparison
            )

            # Determine similarity level
            similarity_level = self._get_similarity_level(overall_similarity)

            # Generate summary
            summary = await self._generate_comparison_summary(
                request, overall_similarity, similarities, differences
            )

            # Calculate confidence
            confidence = self._calculate_comparison_confidence(
                similarities, differences, overall_similarity
            )

            processing_time = time.time() - start_time

            # Update statistics
            self.stats['successful_comparisons'] += 1
            self.stats['comparison_type_counts'][request.comparison_type.value] += 1
            self.stats['similarity_level_counts'][similarity_level.value] += 1
            self._update_average_metrics(overall_similarity, processing_time)

            return ComparisonResult(
                success=True,
                document1_id=request.document1_id,
                document2_id=request.document2_id,
                comparison_type=request.comparison_type,
                overall_similarity_score=overall_similarity,
                similarity_level=similarity_level,
                similarities=similarities,
                differences=differences,
                structural_comparison=structural_comparison,
                summary=summary,
                processing_time=processing_time,
                confidence=confidence,
                metadata={
                    'doc1_length': len(request.document1),
                    'doc2_length': len(request.document2),
                    'similarities_found': len(similarities),
                    'differences_found': len(differences),
                    'model_used': self.config.model.model_name
                }
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Document comparison failed: {str(e)}"
            self.logger.error(error_msg)

            return ComparisonResult(
                success=False,
                document1_id=request.document1_id,
                document2_id=request.document2_id,
                comparison_type=request.comparison_type,
                overall_similarity_score=0.0,
                similarity_level=SimilarityLevel.VERY_LOW,
                similarities=[],
                differences=[],
                structural_comparison=None,
                summary="",
                processing_time=processing_time,
                confidence=0.0,
                metadata={},
                error=error_msg
            )

    def _validate_input(self, request: ComparisonRequest) -> ComparisonResult:
        """Validate comparison request."""
        if not request.document1 or not request.document1.strip():
            return ComparisonResult(
                success=False,
                document1_id=request.document1_id,
                document2_id=request.document2_id,
                comparison_type=request.comparison_type,
                overall_similarity_score=0.0,
                similarity_level=SimilarityLevel.VERY_LOW,
                similarities=[],
                differences=[],
                structural_comparison=None,
                summary="",
                processing_time=0.0,
                confidence=0.0,
                metadata={},
                error="Empty or invalid document1 provided"
            )

        if not request.document2 or not request.document2.strip():
            return ComparisonResult(
                success=False,
                document1_id=request.document1_id,
                document2_id=request.document2_id,
                comparison_type=request.comparison_type,
                overall_similarity_score=0.0,
                similarity_level=SimilarityLevel.VERY_LOW,
                similarities=[],
                differences=[],
                structural_comparison=None,
                summary="",
                processing_time=0.0,
                confidence=0.0,
                metadata={},
                error="Empty or invalid document2 provided"
            )

        # Check length limits
        max_length = self.config.comparator.max_document_length
        if len(request.document1) > max_length:
            return ComparisonResult(
                success=False,
                document1_id=request.document1_id,
                document2_id=request.document2_id,
                comparison_type=request.comparison_type,
                overall_similarity_score=0.0,
                similarity_level=SimilarityLevel.VERY_LOW,
                similarities=[],
                differences=[],
                structural_comparison=None,
                summary="",
                processing_time=0.0,
                confidence=0.0,
                metadata={},
                error=f"Document1 too long. Maximum {max_length} characters allowed."
            )

        if len(request.document2) > max_length:
            return ComparisonResult(
                success=False,
                document1_id=request.document1_id,
                document2_id=request.document2_id,
                comparison_type=request.comparison_type,
                overall_similarity_score=0.0,
                similarity_level=SimilarityLevel.VERY_LOW,
                similarities=[],
                differences=[],
                structural_comparison=None,
                summary="",
                processing_time=0.0,
                confidence=0.0,
                metadata={},
                error=f"Document2 too long. Maximum {max_length} characters allowed."
            )

        # Validation passed
        return ComparisonResult(
            success=True,
            document1_id=request.document1_id,
            document2_id=request.document2_id,
            comparison_type=request.comparison_type,
            overall_similarity_score=0.0,
            similarity_level=SimilarityLevel.VERY_LOW,
            similarities=[],
            differences=[],
            structural_comparison=None,
            summary="",
            processing_time=0.0,
            confidence=1.0,
            metadata={}
        )

    def _calculate_basic_similarity(self, doc1: str, doc2: str) -> float:
        """Calculate basic text similarity using sequence matching."""
        try:
            # Use SequenceMatcher for basic similarity
            matcher = SequenceMatcher(None, doc1.lower(), doc2.lower())
            return matcher.ratio()
        except Exception as e:
            self.logger.warning(f"Basic similarity calculation failed: {e}")
            return 0.0

    def _compare_structure(self, doc1: str, doc2: str) -> StructuralComparison:
        """Compare document structure."""
        try:
            # Extract structural elements from both documents
            doc1_structure = self._extract_structure_elements(doc1)
            doc2_structure = self._extract_structure_elements(doc2)

            # Find common and missing elements
            all_elements = set(doc1_structure.keys()).union(set(doc2_structure.keys()))
            common_elements = []
            missing_in_doc1 = []
            missing_in_doc2 = []

            for element in all_elements:
                if element in doc1_structure and element in doc2_structure:
                    common_elements.append(element)
                elif element in doc1_structure:
                    missing_in_doc2.append(element)
                else:
                    missing_in_doc1.append(element)

            # Calculate structural similarity
            if all_elements:
                structural_similarity = len(common_elements) / len(all_elements)
            else:
                structural_similarity = 0.0

            return StructuralComparison(
                doc1_structure=doc1_structure,
                doc2_structure=doc2_structure,
                structural_similarity=structural_similarity,
                missing_in_doc1=missing_in_doc1,
                missing_in_doc2=missing_in_doc2,
                common_elements=common_elements
            )

        except Exception as e:
            self.logger.warning(f"Structure comparison failed: {e}")
            return StructuralComparison(
                doc1_structure={},
                doc2_structure={},
                structural_similarity=0.0,
                missing_in_doc1=[],
                missing_in_doc2=[],
                common_elements=[]
            )

    def _extract_structure_elements(self, text: str) -> Dict[str, int]:
        """Extract structural elements and their counts."""
        structure = {}

        for element_type, pattern in self.legal_patterns.items():
            if element_type in ['bab', 'pasal', 'ayat']:
                matches = pattern.findall(text)
                structure[element_type] = len(matches)

        return structure

    async def _compare_with_ai(self, request: ComparisonRequest) -> Tuple[List[Similarity], List[Difference]]:
        """Compare documents using AI for semantic analysis."""
        similarities = []
        differences = []

        try:
            # Use AI comparison prompt
            prompt = self.prompt_manager.get_prompt(
                PromptType.COMPARE_DOCUMENTS,
                document1=request.document1[:5000],  # Limit for API
                document2=request.document2[:5000]   # Limit for API
            )

            ai_response = await self._call_ai_model(prompt)

            if ai_response:
                similarities, differences = self._parse_ai_comparison_response(ai_response)

        except Exception as e:
            self.logger.warning(f"AI comparison failed: {e}")

        return similarities, differences

    def _compare_legal_concepts(self, doc1: str, doc2: str) -> Tuple[List[Similarity], List[Difference]]:
        """Compare legal concepts between documents."""
        similarities = []
        differences = []

        try:
            # Extract legal concepts from both documents
            doc1_concepts = self._extract_legal_concepts(doc1)
            doc2_concepts = self._extract_legal_concepts(doc2)

            # Find common concepts (similarities)
            for concept_type in doc1_concepts:
                if concept_type in doc2_concepts:
                    # Calculate similarity of concept instances
                    common_instances = set(doc1_concepts[concept_type]).intersection(
                        set(doc2_concepts[concept_type])
                    )

                    if common_instances:
                        for instance in common_instances:
                            similarities.append(Similarity(
                                content=instance,
                                similarity_score=0.9,
                                category=concept_type,
                                source_doc1=f"Found in document 1",
                                source_doc2=f"Found in document 2",
                                confidence=0.8
                            ))

            # Find differences
            for concept_type in doc1_concepts:
                unique_in_doc1 = set(doc1_concepts[concept_type]) - set(doc2_concepts.get(concept_type, []))
                for instance in unique_in_doc1:
                    differences.append(Difference(
                        doc1_content=instance,
                        doc2_content="Not found",
                        difference_type="missing_in_doc2",
                        category=concept_type,
                        significance="medium",
                        confidence=0.7
                    ))

            for concept_type in doc2_concepts:
                unique_in_doc2 = set(doc2_concepts[concept_type]) - set(doc1_concepts.get(concept_type, []))
                for instance in unique_in_doc2:
                    differences.append(Difference(
                        doc1_content="Not found",
                        doc2_content=instance,
                        difference_type="missing_in_doc1",
                        category=concept_type,
                        significance="medium",
                        confidence=0.7
                    ))

        except Exception as e:
            self.logger.warning(f"Legal concept comparison failed: {e}")

        return similarities, differences

    def _extract_legal_concepts(self, text: str) -> Dict[str, List[str]]:
        """Extract legal concepts from text."""
        concepts = {}

        for concept_type, pattern in self.legal_patterns.items():
            if concept_type not in ['bab', 'pasal', 'ayat']:  # Skip structural patterns
                matches = pattern.findall(text)
                if matches:
                    concepts[concept_type] = [match if isinstance(match, str) else match[0] for match in matches]

        return concepts

    async def _call_ai_model(self, prompt: str) -> str:
        """Call AI model with prompt."""
        try:
            response = self.model.generate_content(prompt)
            return response.text if response and response.text else ""
        except Exception as e:
            self.logger.error(f"AI model call failed: {e}")
            return ""

    def _parse_ai_comparison_response(self, response: str) -> Tuple[List[Similarity], List[Difference]]:
        """Parse AI response for similarities and differences."""
        similarities = []
        differences = []

        try:
            # Simple parsing - in production, this would be more sophisticated
            lines = response.split('\n')
            current_section = None

            for line in lines:
                line = line.strip()
                if 'persamaan' in line.lower() or 'similarities' in line.lower():
                    current_section = 'similarities'
                elif 'perbedaan' in line.lower() or 'differences' in line.lower():
                    current_section = 'differences'
                elif line and current_section == 'similarities':
                    similarities.append(Similarity(
                        content=line,
                        similarity_score=0.8,
                        category='ai_identified',
                        source_doc1="AI analysis",
                        source_doc2="AI analysis",
                        confidence=0.7
                    ))
                elif line and current_section == 'differences':
                    differences.append(Difference(
                        doc1_content=line,
                        doc2_content="Different content",
                        difference_type="content_difference",
                        category='ai_identified',
                        significance="medium",
                        confidence=0.7
                    ))

        except Exception as e:
            self.logger.warning(f"AI response parsing failed: {e}")

        return similarities, differences

    def _calculate_overall_similarity(self, basic_similarity: float,
                                    similarities: List[Similarity],
                                    structural_comparison: Optional[StructuralComparison]) -> float:
        """Calculate overall similarity score."""
        try:
            # Weight different factors
            weights = {
                'basic': 0.3,
                'semantic': 0.4,
                'structural': 0.3
            }

            total_score = basic_similarity * weights['basic']

            # Add semantic similarity from AI analysis
            if similarities:
                semantic_score = sum(sim.similarity_score for sim in similarities) / len(similarities)
                total_score += semantic_score * weights['semantic']
            else:
                total_score += basic_similarity * weights['semantic']  # Fallback

            # Add structural similarity
            if structural_comparison:
                total_score += structural_comparison.structural_similarity * weights['structural']
            else:
                total_score += basic_similarity * weights['structural']  # Fallback

            return min(total_score, 1.0)

        except Exception as e:
            self.logger.warning(f"Overall similarity calculation failed: {e}")
            return basic_similarity

    def _get_similarity_level(self, similarity_score: float) -> SimilarityLevel:
        """Determine similarity level from score."""
        if similarity_score >= 0.95:
            return SimilarityLevel.IDENTICAL
        elif similarity_score >= 0.80:
            return SimilarityLevel.VERY_HIGH
        elif similarity_score >= 0.60:
            return SimilarityLevel.HIGH
        elif similarity_score >= 0.40:
            return SimilarityLevel.MODERATE
        elif similarity_score >= 0.20:
            return SimilarityLevel.LOW
        else:
            return SimilarityLevel.VERY_LOW

    async def _generate_comparison_summary(self, request: ComparisonRequest,
                                         similarity_score: float,
                                         similarities: List[Similarity],
                                         differences: List[Difference]) -> str:
        """Generate comparison summary."""
        try:
            similarity_level = self._get_similarity_level(similarity_score)

            summary_parts = [
                f"Perbandingan antara {request.document1_id} dan {request.document2_id}:",
                f"Tingkat kemiripan: {similarity_level.value} ({similarity_score:.1%})",
                f"Persamaan ditemukan: {len(similarities)}",
                f"Perbedaan ditemukan: {len(differences)}"
            ]

            if similarities:
                summary_parts.append("Persamaan utama:")
                for sim in similarities[:3]:  # Top 3 similarities
                    summary_parts.append(f"- {sim.content[:100]}...")

            if differences:
                summary_parts.append("Perbedaan utama:")
                for diff in differences[:3]:  # Top 3 differences
                    summary_parts.append(f"- {diff.difference_type}: {diff.doc1_content[:50]}...")

            return "\n".join(summary_parts)

        except Exception as e:
            self.logger.warning(f"Summary generation failed: {e}")
            return f"Comparison completed with {similarity_score:.1%} similarity"

    def _calculate_comparison_confidence(self, similarities: List[Similarity],
                                       differences: List[Difference],
                                       similarity_score: float) -> float:
        """Calculate confidence score for comparison."""
        try:
            # Base confidence from similarity score
            base_confidence = similarity_score * 0.5

            # Boost confidence based on number of identified elements
            element_boost = min((len(similarities) + len(differences)) / 20, 0.3)

            # Boost confidence based on individual element confidence
            if similarities or differences:
                all_elements = similarities + differences
                avg_element_confidence = sum(
                    getattr(elem, 'confidence', 0.5) for elem in all_elements
                ) / len(all_elements)
                confidence_boost = avg_element_confidence * 0.2
            else:
                confidence_boost = 0.0

            total_confidence = base_confidence + element_boost + confidence_boost
            return min(total_confidence, 1.0)

        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.5

    def _update_average_metrics(self, similarity_score: float, processing_time: float) -> None:
        """Update average similarity and processing time."""
        total_comparisons = self.stats['total_comparisons']

        # Update average similarity
        current_avg_sim = self.stats['average_similarity_score']
        self.stats['average_similarity_score'] = (
            (current_avg_sim * (total_comparisons - 1) + similarity_score) / total_comparisons
        )

        # Update average processing time
        current_avg_time = self.stats['average_processing_time']
        self.stats['average_processing_time'] = (
            (current_avg_time * (total_comparisons - 1) + processing_time) / total_comparisons
        )

    async def compare_similarity_only(self, doc1: str, doc2: str,
                                    doc1_id: str = "doc1", doc2_id: str = "doc2") -> ComparisonResult:
        """Convenience method for similarity-only comparison."""
        request = ComparisonRequest(
            document1=doc1,
            document2=doc2,
            document1_id=doc1_id,
            document2_id=doc2_id,
            comparison_type=ComparisonType.SIMILARITY_ONLY,
            include_differences=False,
            include_similarities=False,
            detailed_analysis=False
        )
        return await self.compare(request)

    async def compare_comprehensive(self, doc1: str, doc2: str,
                                  doc1_id: str = "doc1", doc2_id: str = "doc2") -> ComparisonResult:
        """Convenience method for comprehensive comparison."""
        request = ComparisonRequest(
            document1=doc1,
            document2=doc2,
            document1_id=doc1_id,
            document2_id=doc2_id,
            comparison_type=ComparisonType.COMPREHENSIVE,
            include_differences=True,
            include_similarities=True,
            detailed_analysis=True
        )
        return await self.compare(request)

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        success_rate = (
            self.stats['successful_comparisons'] / self.stats['total_comparisons']
            if self.stats['total_comparisons'] > 0 else 0
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
            'total_comparisons': 0,
            'successful_comparisons': 0,
            'average_similarity_score': 0.0,
            'average_processing_time': 0.0,
            'comparison_type_counts': {ct.value: 0 for ct in ComparisonType},
            'similarity_level_counts': {sl.value: 0 for sl in SimilarityLevel}
        }

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the comparator service."""
        return {
            'name': 'Legal Document Comparator',
            'description': 'AI-powered comparison for Indonesian legal documents',
            'supported_comparison_types': [ct.value for ct in ComparisonType],
            'similarity_levels': [sl.value for sl in SimilarityLevel],
            'features': [
                'Structural comparison',
                'Semantic analysis',
                'Legal concept comparison',
                'Similarity scoring',
                'Difference identification',
                'Indonesian legal optimization',
                'Confidence scoring'
            ],
            'configuration': {
                'max_document_length': self.config.comparator.max_document_length,
                'similarity_threshold': self.config.comparator.similarity_threshold,
                'semantic_comparison': self.config.comparator.semantic_comparison,
                'model_name': self.config.model.model_name
            },
            'status': 'available' if self.model else 'unavailable'
        }

    def __repr__(self) -> str:
        """String representation of comparator service."""
        return f"ComparatorService(model={self.config.model.model_name}, available={self.model is not None})"
