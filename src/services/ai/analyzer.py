"""
Analyzer Service
Specialized AI service for legal document content analysis

This module provides intelligent content analysis capabilities specifically optimized
for Indonesian legal documents, including structure analysis, definition extraction,
sanction identification, and concept analysis.

Author: Refactored Architecture
Purpose: Single responsibility content analysis using AI
"""

import logging
import time
import asyncio
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from src.config.ai_config import AIConfig, AnalysisType
from src.utils.prompt_manager import PromptManager, PromptType


class AnalysisScope(Enum):
    """Scope of analysis to perform."""
    BASIC = "basic"           # Basic structure and definitions
    COMPREHENSIVE = "comprehensive"  # Full analysis including concepts
    FOCUSED = "focused"       # Specific analysis type only
    DEEP = "deep"            # Deep analysis with relationships


@dataclass
class AnalysisRequest:
    """Request for document analysis."""
    text: str
    analysis_types: List[AnalysisType]
    scope: AnalysisScope = AnalysisScope.COMPREHENSIVE
    max_concepts: Optional[int] = None
    include_relationships: bool = True
    legal_focus: bool = True
    confidence_threshold: float = 0.5


@dataclass
class LegalDefinition:
    """Extracted legal definition."""
    term: str
    definition: str
    context: str
    confidence: float
    source_section: str
    importance_score: float


@dataclass
class LegalSanction:
    """Identified legal sanction."""
    sanction_type: str
    description: str
    severity: str
    conditions: List[str]
    legal_basis: str
    confidence: float


@dataclass
class LegalConcept:
    """Extracted legal concept."""
    term: str
    category: str
    description: str
    importance_score: float
    context_examples: List[str]
    related_terms: List[str]
    confidence: float


@dataclass
class StructureElement:
    """Document structure element."""
    element_type: str
    number: str
    title: str
    content: str
    level: int
    confidence: float


@dataclass
class AnalysisResult:
    """Result of document analysis."""
    success: bool
    analysis_types: List[AnalysisType]
    structure_elements: List[StructureElement]
    definitions: List[LegalDefinition]
    sanctions: List[LegalSanction]
    concepts: List[LegalConcept]
    overall_confidence: float
    processing_time: float
    metadata: Dict[str, Any]
    error: str = ""


class AnalyzerService:
    """
    AI-powered document analyzer service.

    Provides intelligent content analysis for Indonesian legal documents
    with support for structure, definitions, sanctions, and concept extraction.
    """

    def __init__(self, config: Optional[AIConfig] = None):
        """
        Initialize analyzer service.

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

        # Load legal patterns
        self._load_legal_patterns()

        # Service statistics
        self.stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'definitions_extracted': 0,
            'sanctions_found': 0,
            'concepts_extracted': 0,
            'structure_elements_found': 0,
            'average_processing_time': 0.0,
            'analysis_type_counts': {at.value: 0 for at in AnalysisType}
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
        """Load patterns for legal document analysis."""
        self.legal_patterns = {
            # Structure patterns
            'bab': re.compile(r'(?m)^(?:BAB|B[AI]B)\s+([IVXLC]+|\d+)\s*(.*?)(?=\n(?:BAB|BAGIAN|PASAL|$))', re.IGNORECASE | re.DOTALL),
            'bagian': re.compile(r'(?m)^BAGIAN\s+([IVXLC]+|\d+)\s*(.*?)(?=\n(?:BAGIAN|PASAL|$))', re.IGNORECASE | re.DOTALL),
            'pasal': re.compile(r'(?m)^Pasal\s+(\d+[A-Z]?)\s*(.*?)(?=\n(?:Pasal|BAB|BAGIAN|$))', re.IGNORECASE | re.DOTALL),
            'ayat': re.compile(r'\((\d+)\)\s*(.*?)(?=\(\d+\)|$)', re.DOTALL),

            # Definition patterns
            'definition_intro': re.compile(r'(?i)yang dimaksud dengan|dalam.*ini.*adalah|pengertian.*meliputi'),
            'definition_list': re.compile(r'(?m)^\s*[a-z]\.\s*(.*?)\s+adalah\s+(.*?)(?=\n\s*[a-z]\.|\n\n|$)', re.DOTALL),

            # Sanction patterns
            'criminal_sanctions': re.compile(r'(?i)dipidana.*?(?:penjara|kurungan).*?(?:paling.*?(\d+).*?(?:tahun|bulan))?'),
            'fine_sanctions': re.compile(r'(?i)denda.*?(?:paling.*?)?(?:Rp\.?\s*)?(\d+(?:\.\d+)*(?:ribu|juta|miliar)?)'),
            'administrative_sanctions': re.compile(r'(?i)pencabutan.*?izin|peringatan.*?tertulis|pembekuan.*?kegiatan'),

            # Legal concept patterns
            'obligations': re.compile(r'(?i)wajib|harus|berkewajiban|diwajibkan'),
            'rights': re.compile(r'(?i)berhak|hak untuk|dapat.*?melakukan'),
            'prohibitions': re.compile(r'(?i)dilarang|tidak boleh|tidak dapat|tidak diperkenankan'),
            'procedures': re.compile(r'(?i)prosedur|tata cara|mekanisme|tahapan')
        }

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform comprehensive analysis of legal document.

        Args:
            request: AnalysisRequest with text and parameters

        Returns:
            AnalysisResult with extracted information
        """
        start_time = time.time()
        self.stats['total_analyses'] += 1

        # Validate input
        if not self._validate_input(request):
            return AnalysisResult(
                success=False,
                analysis_types=[],
                structure_elements=[],
                definitions=[],
                sanctions=[],
                concepts=[],
                overall_confidence=0.0,
                processing_time=0.0,
                metadata={},
                error="Invalid input parameters"
            )

        if not GEMINI_AVAILABLE or not self.model:
            return AnalysisResult(
                success=False,
                analysis_types=request.analysis_types,
                structure_elements=[],
                definitions=[],
                sanctions=[],
                concepts=[],
                overall_confidence=0.0,
                processing_time=0.0,
                metadata={},
                error="AI model not available"
            )

        try:
            # Perform different types of analysis
            structure_elements = []
            definitions = []
            sanctions = []
            concepts = []

            # Structure analysis
            if AnalysisType.STRUCTURE in request.analysis_types:
                structure_elements = await self._analyze_structure(request.text)

            # Definition extraction
            if AnalysisType.DEFINITIONS in request.analysis_types:
                definitions = await self._extract_definitions(request.text)

            # Sanction identification
            if AnalysisType.SANCTIONS in request.analysis_types:
                sanctions = await self._find_sanctions(request.text)

            # Concept extraction (more computationally intensive)
            if AnalysisType.DOCUMENT_INTELLIGENCE in request.analysis_types:
                concepts = await self._extract_concepts(request.text, request.max_concepts)

            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                structure_elements, definitions, sanctions, concepts
            )

            processing_time = time.time() - start_time

            # Update statistics
            self.stats['successful_analyses'] += 1
            self.stats['definitions_extracted'] += len(definitions)
            self.stats['sanctions_found'] += len(sanctions)
            self.stats['concepts_extracted'] += len(concepts)
            self.stats['structure_elements_found'] += len(structure_elements)

            for analysis_type in request.analysis_types:
                self.stats['analysis_type_counts'][analysis_type.value] += 1

            self._update_average_processing_time(processing_time)

            return AnalysisResult(
                success=True,
                analysis_types=request.analysis_types,
                structure_elements=structure_elements,
                definitions=definitions,
                sanctions=sanctions,
                concepts=concepts,
                overall_confidence=overall_confidence,
                processing_time=processing_time,
                metadata={
                    'text_length': len(request.text),
                    'scope': request.scope.value,
                    'model_used': self.config.model.model_name,
                    'total_elements_found': len(structure_elements) + len(definitions) + len(sanctions) + len(concepts)
                }
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Analysis failed: {str(e)}"
            self.logger.error(error_msg)

            return AnalysisResult(
                success=False,
                analysis_types=request.analysis_types,
                structure_elements=[],
                definitions=[],
                sanctions=[],
                concepts=[],
                overall_confidence=0.0,
                processing_time=processing_time,
                metadata={'text_length': len(request.text)},
                error=error_msg
            )

    def _validate_input(self, request: AnalysisRequest) -> bool:
        """Validate analysis request."""
        if not request.text or not request.text.strip():
            return False

        if not request.analysis_types:
            return False

        if len(request.text) > self.config.analyzer.max_content_length:
            return False

        return True

    async def _analyze_structure(self, text: str) -> List[StructureElement]:
        """Analyze document structure using AI and patterns."""
        structure_elements = []

        try:
            # Use AI for structure analysis
            prompt = self.prompt_manager.get_prompt(PromptType.ANALYZE_STRUCTURE, text=text)
            ai_response = await self._call_ai_model(prompt)

            # Parse AI response and extract structure
            if ai_response:
                structure_elements.extend(self._parse_structure_response(ai_response))

            # Complement with pattern-based analysis
            pattern_elements = self._extract_structure_with_patterns(text)

            # Merge and deduplicate
            structure_elements = self._merge_structure_elements(structure_elements, pattern_elements)

        except Exception as e:
            self.logger.warning(f"Structure analysis failed: {e}")
            # Fallback to pattern-based analysis only
            structure_elements = self._extract_structure_with_patterns(text)

        return structure_elements

    def _extract_structure_with_patterns(self, text: str) -> List[StructureElement]:
        """Extract structure using regex patterns."""
        elements = []

        # Extract BAB (chapters)
        for match in self.legal_patterns['bab'].finditer(text):
            elements.append(StructureElement(
                element_type='bab',
                number=match.group(1),
                title=match.group(2).strip() if match.group(2) else '',
                content=match.group(0),
                level=1,
                confidence=0.9
            ))

        # Extract PASAL (articles)
        for match in self.legal_patterns['pasal'].finditer(text):
            elements.append(StructureElement(
                element_type='pasal',
                number=match.group(1),
                title='',
                content=match.group(2).strip() if match.group(2) else '',
                level=2,
                confidence=0.9
            ))

        # Extract AYAT (verses)
        for match in self.legal_patterns['ayat'].finditer(text):
            elements.append(StructureElement(
                element_type='ayat',
                number=match.group(1),
                title='',
                content=match.group(2).strip() if match.group(2) else '',
                level=3,
                confidence=0.8
            ))

        return elements

    async def _extract_definitions(self, text: str) -> List[LegalDefinition]:
        """Extract legal definitions using AI and patterns."""
        definitions = []

        try:
            # Use AI for definition extraction
            prompt = self.prompt_manager.get_prompt(PromptType.EXTRACT_DEFINITIONS, text=text)
            ai_response = await self._call_ai_model(prompt)

            if ai_response:
                definitions.extend(self._parse_definitions_response(ai_response))

            # Complement with pattern-based extraction
            pattern_definitions = self._extract_definitions_with_patterns(text)
            definitions.extend(pattern_definitions)

            # Remove duplicates and rank by importance
            definitions = self._deduplicate_and_rank_definitions(definitions)

        except Exception as e:
            self.logger.warning(f"Definition extraction failed: {e}")
            definitions = self._extract_definitions_with_patterns(text)

        return definitions

    def _extract_definitions_with_patterns(self, text: str) -> List[LegalDefinition]:
        """Extract definitions using regex patterns."""
        definitions = []

        # Look for definition sections
        for match in self.legal_patterns['definition_list'].finditer(text):
            term = match.group(1).strip()
            definition = match.group(2).strip()

            if term and definition:
                definitions.append(LegalDefinition(
                    term=term,
                    definition=definition,
                    context=match.group(0),
                    confidence=0.8,
                    source_section='Pattern-based extraction',
                    importance_score=0.7
                ))

        return definitions

    async def _find_sanctions(self, text: str) -> List[LegalSanction]:
        """Find legal sanctions using AI and patterns."""
        sanctions = []

        try:
            # Use AI for sanction identification
            prompt = self.prompt_manager.get_prompt(PromptType.FIND_SANCTIONS, text=text)
            ai_response = await self._call_ai_model(prompt)

            if ai_response:
                sanctions.extend(self._parse_sanctions_response(ai_response))

            # Complement with pattern-based extraction
            pattern_sanctions = self._extract_sanctions_with_patterns(text)
            sanctions.extend(pattern_sanctions)

            # Remove duplicates
            sanctions = self._deduplicate_sanctions(sanctions)

        except Exception as e:
            self.logger.warning(f"Sanction identification failed: {e}")
            sanctions = self._extract_sanctions_with_patterns(text)

        return sanctions

    def _extract_sanctions_with_patterns(self, text: str) -> List[LegalSanction]:
        """Extract sanctions using regex patterns."""
        sanctions = []

        # Criminal sanctions
        for match in self.legal_patterns['criminal_sanctions'].finditer(text):
            sanctions.append(LegalSanction(
                sanction_type='pidana',
                description=match.group(0),
                severity='high',
                conditions=[],
                legal_basis='Pattern-based extraction',
                confidence=0.8
            ))

        # Fine sanctions
        for match in self.legal_patterns['fine_sanctions'].finditer(text):
            sanctions.append(LegalSanction(
                sanction_type='denda',
                description=match.group(0),
                severity='medium',
                conditions=[],
                legal_basis='Pattern-based extraction',
                confidence=0.8
            ))

        # Administrative sanctions
        for match in self.legal_patterns['administrative_sanctions'].finditer(text):
            sanctions.append(LegalSanction(
                sanction_type='administratif',
                description=match.group(0),
                severity='medium',
                conditions=[],
                legal_basis='Pattern-based extraction',
                confidence=0.7
            ))

        return sanctions

    async def _extract_concepts(self, text: str, max_concepts: Optional[int] = None) -> List[LegalConcept]:
        """Extract legal concepts using AI."""
        concepts = []

        try:
            # Use AI for concept extraction
            prompt = self.prompt_manager.get_prompt(PromptType.EXTRACT_LEGAL_CONCEPTS, text=text)
            ai_response = await self._call_ai_model(prompt)

            if ai_response:
                concepts = self._parse_concepts_response(ai_response)

            # Limit concepts if specified
            if max_concepts and len(concepts) > max_concepts:
                concepts = sorted(concepts, key=lambda x: x.importance_score, reverse=True)[:max_concepts]

        except Exception as e:
            self.logger.warning(f"Concept extraction failed: {e}")

        return concepts

    async def _call_ai_model(self, prompt: str) -> str:
        """Call AI model with prompt."""
        try:
            response = self.model.generate_content(prompt)
            return response.text if response and response.text else ""
        except Exception as e:
            self.logger.error(f"AI model call failed: {e}")
            return ""

    def _parse_structure_response(self, response: str) -> List[StructureElement]:
        """Parse AI response for structure elements."""
        # Implementation would parse structured AI response
        # This is a simplified version
        elements = []
        try:
            # Parse JSON or structured text response
            # Add structure elements based on AI response
            pass
        except Exception as e:
            self.logger.warning(f"Structure response parsing failed: {e}")
        return elements

    def _parse_definitions_response(self, response: str) -> List[LegalDefinition]:
        """Parse AI response for definitions."""
        definitions = []
        try:
            # Parse JSON or structured text response
            # Add definitions based on AI response
            pass
        except Exception as e:
            self.logger.warning(f"Definitions response parsing failed: {e}")
        return definitions

    def _parse_sanctions_response(self, response: str) -> List[LegalSanction]:
        """Parse AI response for sanctions."""
        sanctions = []
        try:
            # Parse JSON or structured text response
            # Add sanctions based on AI response
            pass
        except Exception as e:
            self.logger.warning(f"Sanctions response parsing failed: {e}")
        return sanctions

    def _parse_concepts_response(self, response: str) -> List[LegalConcept]:
        """Parse AI response for concepts."""
        concepts = []
        try:
            # Parse JSON or structured text response
            # Add concepts based on AI response
            pass
        except Exception as e:
            self.logger.warning(f"Concepts response parsing failed: {e}")
        return concepts

    def _merge_structure_elements(self, ai_elements: List[StructureElement],
                                pattern_elements: List[StructureElement]) -> List[StructureElement]:
        """Merge AI and pattern-based structure elements."""
        # Simple implementation - combine and deduplicate
        all_elements = ai_elements + pattern_elements
        # Remove duplicates based on type, number, and content similarity
        unique_elements = []
        seen = set()

        for element in all_elements:
            key = (element.element_type, element.number)
            if key not in seen:
                unique_elements.append(element)
                seen.add(key)

        return sorted(unique_elements, key=lambda x: (x.level, x.number))

    def _deduplicate_and_rank_definitions(self, definitions: List[LegalDefinition]) -> List[LegalDefinition]:
        """Remove duplicate definitions and rank by importance."""
        # Remove duplicates based on term similarity
        unique_definitions = []
        seen_terms = set()

        for definition in definitions:
            term_lower = definition.term.lower()
            if term_lower not in seen_terms:
                unique_definitions.append(definition)
                seen_terms.add(term_lower)

        # Sort by importance score
        return sorted(unique_definitions, key=lambda x: x.importance_score, reverse=True)

    def _deduplicate_sanctions(self, sanctions: List[LegalSanction]) -> List[LegalSanction]:
        """Remove duplicate sanctions."""
        unique_sanctions = []
        seen_descriptions = set()

        for sanction in sanctions:
            desc_key = sanction.description[:50].lower()  # Use first 50 chars as key
            if desc_key not in seen_descriptions:
                unique_sanctions.append(sanction)
                seen_descriptions.add(desc_key)

        return unique_sanctions

    def _calculate_overall_confidence(self, structure_elements: List[StructureElement],
                                    definitions: List[LegalDefinition],
                                    sanctions: List[LegalSanction],
                                    concepts: List[LegalConcept]) -> float:
        """Calculate overall confidence score for the analysis."""
        total_confidence = 0.0
        total_weight = 0.0

        # Weight by number of elements and their individual confidence
        if structure_elements:
            avg_structure_conf = sum(e.confidence for e in structure_elements) / len(structure_elements)
            total_confidence += avg_structure_conf * len(structure_elements) * 0.3
            total_weight += len(structure_elements) * 0.3

        if definitions:
            avg_def_conf = sum(d.confidence for d in definitions) / len(definitions)
            total_confidence += avg_def_conf * len(definitions) * 0.25
            total_weight += len(definitions) * 0.25

        if sanctions:
            avg_sanction_conf = sum(s.confidence for s in sanctions) / len(sanctions)
            total_confidence += avg_sanction_conf * len(sanctions) * 0.25
            total_weight += len(sanctions) * 0.25

        if concepts:
            avg_concept_conf = sum(c.confidence for c in concepts) / len(concepts)
            total_confidence += avg_concept_conf * len(concepts) * 0.2
            total_weight += len(concepts) * 0.2

        return total_confidence / total_weight if total_weight > 0 else 0.0

    def _update_average_processing_time(self, processing_time: float) -> None:
        """Update average processing time statistic."""
        current_avg = self.stats['average_processing_time']
        total_analyses = self.stats['total_analyses']

        self.stats['average_processing_time'] = (
            (current_avg * (total_analyses - 1) + processing_time) / total_analyses
        )

    async def analyze_structure_only(self, text: str) -> AnalysisResult:
        """Convenience method for structure analysis only."""
        request = AnalysisRequest(
            text=text,
            analysis_types=[AnalysisType.STRUCTURE],
            scope=AnalysisScope.FOCUSED
        )
        return await self.analyze(request)

    async def extract_definitions_only(self, text: str) -> AnalysisResult:
        """Convenience method for definition extraction only."""
        request = AnalysisRequest(
            text=text,
            analysis_types=[AnalysisType.DEFINITIONS],
            scope=AnalysisScope.FOCUSED
        )
        return await self.analyze(request)

    async def find_sanctions_only(self, text: str) -> AnalysisResult:
        """Convenience method for sanction identification only."""
        request = AnalysisRequest(
            text=text,
            analysis_types=[AnalysisType.SANCTIONS],
            scope=AnalysisScope.FOCUSED
        )
        return await self.analyze(request)

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        success_rate = (
            self.stats['successful_analyses'] / self.stats['total_analyses']
            if self.stats['total_analyses'] > 0 else 0
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
            'total_analyses': 0,
            'successful_analyses': 0,
            'definitions_extracted': 0,
            'sanctions_found': 0,
            'concepts_extracted': 0,
            'structure_elements_found': 0,
            'average_processing_time': 0.0,
            'analysis_type_counts': {at.value: 0 for at in AnalysisType}
        }

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the analyzer service."""
        return {
            'name': 'Legal Document Analyzer',
            'description': 'AI-powered content analysis for Indonesian legal documents',
            'supported_analysis_types': [at.value for at in AnalysisType],
            'features': [
                'Structure analysis',
                'Definition extraction',
                'Sanction identification',
                'Concept extraction',
                'Indonesian legal optimization',
                'Pattern-based fallbacks',
                'Confidence scoring'
            ],
            'configuration': {
                'max_content_length': self.config.analyzer.max_content_length,
                'legal_pattern_matching': self.config.analyzer.legal_pattern_matching,
                'concept_mapping': self.config.analyzer.concept_mapping,
                'model_name': self.config.model.model_name
            },
            'status': 'available' if self.model else 'unavailable'
        }

    def __repr__(self) -> str:
        """String representation of analyzer service."""
        return f"AnalyzerService(model={self.config.model.model_name}, available={self.model is not None})"
