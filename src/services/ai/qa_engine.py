"""
QA Engine Service
Specialized AI service for question-answering on legal documents

This module provides intelligent question-answering capabilities specifically optimized
for Indonesian legal documents, with context-aware responses and legal domain focus.

Author: Refactored Architecture
Purpose: Single responsibility question-answering using AI
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from src.config.ai_config import AIConfig
from src.utils.prompt_manager import PromptManager, PromptType


class QuestionType(Enum):
    """Types of questions that can be answered."""
    FACTUAL = "factual"           # Direct factual questions
    PROCEDURAL = "procedural"     # How-to and process questions
    LEGAL_INTERPRETATION = "legal_interpretation"  # Legal meaning and interpretation
    COMPLIANCE = "compliance"     # Compliance and requirement questions
    COMPARISON = "comparison"     # Comparing different aspects
    DEFINITION = "definition"     # What is X questions
    CONSEQUENCE = "consequence"   # What happens if questions


@dataclass
class QARequest:
    """Request for question-answering."""
    question: str
    context: str
    question_type: Optional[QuestionType] = None
    max_answer_length: Optional[int] = None
    include_citations: bool = True
    legal_focus: bool = True
    confidence_threshold: float = 0.6
    multi_hop_reasoning: bool = False


@dataclass
class Citation:
    """Citation reference in the answer."""
    text: str
    source_section: str
    relevance_score: float
    start_position: int
    end_position: int


@dataclass
class QAResult:
    """Result of question-answering."""
    success: bool
    question: str
    answer: str
    question_type: QuestionType
    confidence: float
    processing_time: float
    citations: List[Citation]
    context_used: str
    metadata: Dict[str, Any]
    error: str = ""
    requires_clarification: bool = False
    follow_up_questions: List[str] = None

    def __post_init__(self):
        if self.follow_up_questions is None:
            self.follow_up_questions = []


class QAEngine:
    """
    AI-powered question-answering engine.

    Provides intelligent Q&A capabilities for Indonesian legal documents
    with context-aware responses and citation support.
    """

    def __init__(self, config: Optional[AIConfig] = None):
        """
        Initialize QA engine.

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

        # Legal patterns for enhanced processing
        self._load_legal_patterns()

        # Service statistics
        self.stats = {
            'total_questions': 0,
            'successful_answers': 0,
            'average_confidence': 0.0,
            'average_processing_time': 0.0,
            'question_type_counts': {qt.value: 0 for qt in QuestionType},
            'citation_counts': 0
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
        """Load legal patterns for question analysis."""
        self.legal_patterns = {
            'pasal_reference': re.compile(r'pasal\s+(\d+[a-z]?)', re.IGNORECASE),
            'ayat_reference': re.compile(r'\((\d+)\)', re.MULTILINE),
            'bab_reference': re.compile(r'bab\s+([ivxlc]+|\d+)', re.IGNORECASE),
            'definition_pattern': re.compile(r'yang dimaksud dengan|pengertian|definisi', re.IGNORECASE),
            'sanction_pattern': re.compile(r'sanksi|pidana|denda|kurungan|penjara', re.IGNORECASE),
            'procedure_pattern': re.compile(r'bagaimana|cara|prosedur|tahapan|langkah', re.IGNORECASE),
            'requirement_pattern': re.compile(r'syarat|ketentuan|persyaratan|wajib', re.IGNORECASE)
        }

    async def answer_question(self, request: QARequest) -> QAResult:
        """
        Answer question based on provided context.

        Args:
            request: QARequest with question and context

        Returns:
            QAResult with answer and metadata
        """
        start_time = time.time()
        self.stats['total_questions'] += 1

        # Validate input
        validation_result = self._validate_input(request)
        if not validation_result.success:
            return validation_result

        if not GEMINI_AVAILABLE or not self.model:
            return QAResult(
                success=False,
                question=request.question,
                answer="",
                question_type=QuestionType.FACTUAL,
                confidence=0.0,
                processing_time=0.0,
                citations=[],
                context_used="",
                metadata={},
                error="AI model not available"
            )

        try:
            # Analyze question type if not provided
            if request.question_type is None:
                request.question_type = self._analyze_question_type(request.question)

            # Prepare context and generate answer
            processed_context = self._process_context(request.context, request.question)
            answer_text, confidence = await self._generate_answer_with_ai(request, processed_context)

            # Extract citations if requested
            citations = []
            if request.include_citations and answer_text:
                citations = self._extract_citations(answer_text, processed_context)

            # Generate follow-up questions
            follow_up_questions = self._generate_follow_up_questions(request.question, answer_text)

            processing_time = time.time() - start_time

            # Update statistics
            if answer_text:
                self.stats['successful_answers'] += 1
                self.stats['question_type_counts'][request.question_type.value] += 1
                self.stats['citation_counts'] += len(citations)

            self._update_average_metrics(confidence, processing_time)

            return QAResult(
                success=bool(answer_text),
                question=request.question,
                answer=answer_text,
                question_type=request.question_type,
                confidence=confidence,
                processing_time=processing_time,
                citations=citations,
                context_used=processed_context[:500] + "..." if len(processed_context) > 500 else processed_context,
                follow_up_questions=follow_up_questions,
                metadata={
                    'question_length': len(request.question),
                    'context_length': len(request.context),
                    'answer_length': len(answer_text) if answer_text else 0,
                    'model_used': self.config.model.model_name,
                    'legal_focus': request.legal_focus,
                    'citations_found': len(citations)
                },
                error="" if answer_text else "Failed to generate answer",
                requires_clarification=confidence < request.confidence_threshold
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Question answering failed: {str(e)}"
            self.logger.error(error_msg)

            return QAResult(
                success=False,
                question=request.question,
                answer="",
                question_type=request.question_type or QuestionType.FACTUAL,
                confidence=0.0,
                processing_time=processing_time,
                citations=[],
                context_used="",
                metadata={'question_length': len(request.question)},
                error=error_msg
            )

    def _validate_input(self, request: QARequest) -> QAResult:
        """Validate QA request."""
        if not request.question or not request.question.strip():
            return QAResult(
                success=False,
                question="",
                answer="",
                question_type=QuestionType.FACTUAL,
                confidence=0.0,
                processing_time=0.0,
                citations=[],
                context_used="",
                metadata={},
                error="Empty or invalid question provided"
            )

        if not request.context or not request.context.strip():
            return QAResult(
                success=False,
                question=request.question,
                answer="",
                question_type=QuestionType.FACTUAL,
                confidence=0.0,
                processing_time=0.0,
                citations=[],
                context_used="",
                metadata={},
                error="Empty or invalid context provided"
            )

        # Check length limits
        if len(request.question) > self.config.qa_engine.max_question_length:
            return QAResult(
                success=False,
                question=request.question,
                answer="",
                question_type=QuestionType.FACTUAL,
                confidence=0.0,
                processing_time=0.0,
                citations=[],
                context_used="",
                metadata={},
                error=f"Question too long. Maximum {self.config.qa_engine.max_question_length} characters allowed."
            )

        if len(request.context) > self.config.qa_engine.max_context_length:
            return QAResult(
                success=False,
                question=request.question,
                answer="",
                question_type=QuestionType.FACTUAL,
                confidence=0.0,
                processing_time=0.0,
                citations=[],
                context_used="",
                metadata={},
                error=f"Context too long. Maximum {self.config.qa_engine.max_context_length} characters allowed."
            )

        # Validation passed
        return QAResult(
            success=True,
            question=request.question,
            answer="",
            question_type=QuestionType.FACTUAL,
            confidence=1.0,
            processing_time=0.0,
            citations=[],
            context_used="",
            metadata={}
        )

    def _analyze_question_type(self, question: str) -> QuestionType:
        """Analyze and classify question type."""
        question_lower = question.lower()

        # Definition questions
        if any(word in question_lower for word in ['apa itu', 'pengertian', 'definisi', 'yang dimaksud']):
            return QuestionType.DEFINITION

        # Procedural questions
        if any(word in question_lower for word in ['bagaimana', 'cara', 'prosedur', 'tahapan', 'langkah']):
            return QuestionType.PROCEDURAL

        # Consequence questions
        if any(word in question_lower for word in ['akibat', 'konsekuensi', 'sanksi', 'jika', 'apabila']):
            return QuestionType.CONSEQUENCE

        # Compliance questions
        if any(word in question_lower for word in ['wajib', 'harus', 'syarat', 'ketentuan', 'persyaratan']):
            return QuestionType.COMPLIANCE

        # Comparison questions
        if any(word in question_lower for word in ['perbedaan', 'persamaan', 'dibandingkan', 'versus']):
            return QuestionType.COMPARISON

        # Legal interpretation questions
        if any(word in question_lower for word in ['makna', 'arti', 'interpretasi', 'maksud', 'berlaku']):
            return QuestionType.LEGAL_INTERPRETATION

        # Default to factual
        return QuestionType.FACTUAL

    def _process_context(self, context: str, question: str) -> str:
        """Process and optimize context for the question."""
        try:
            # Truncate context if too long
            max_context = self.config.qa_engine.max_context_length
            if len(context) > max_context:
                # Try to find relevant sections
                context = self._find_relevant_context_sections(context, question, max_context)

            return context

        except Exception as e:
            self.logger.warning(f"Context processing failed: {e}")
            return context[:self.config.qa_engine.max_context_length]

    def _find_relevant_context_sections(self, context: str, question: str, max_length: int) -> str:
        """Find most relevant sections of context for the question."""
        try:
            # Split context into sentences
            sentences = context.split('. ')
            question_words = set(question.lower().split())

            # Score sentences by relevance
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                sentence_words = set(sentence.lower().split())
                overlap = len(question_words.intersection(sentence_words))

                # Boost score for legal references
                legal_boost = 0
                for pattern in self.legal_patterns.values():
                    if pattern.search(sentence):
                        legal_boost += 1

                score = overlap + legal_boost * 2
                scored_sentences.append((score, i, sentence))

            # Sort by score and select top sentences
            scored_sentences.sort(reverse=True, key=lambda x: x[0])

            selected_text = ""
            for score, idx, sentence in scored_sentences:
                if len(selected_text + sentence) < max_length:
                    selected_text += sentence + ". "
                else:
                    break

            return selected_text.strip()

        except Exception as e:
            self.logger.warning(f"Relevant context extraction failed: {e}")
            return context[:max_length]

    async def _generate_answer_with_ai(self, request: QARequest, context: str) -> Tuple[str, float]:
        """Generate answer using AI model."""
        try:
            # Get formatted prompt
            prompt = self.prompt_manager.get_prompt(
                PromptType.QUESTION_ANSWER,
                context=context,
                question=request.question
            )

            # Add question type specific instructions
            if request.question_type:
                prompt += f"\n\nTIPE PERTANYAAN: {request.question_type.value}\n"

            # Generate response
            response = await asyncio.create_task(
                self._call_ai_model(prompt)
            )

            if response and len(response.strip()) > 10:
                confidence = self._calculate_answer_confidence(response, request.question, context)
                return response.strip(), confidence
            else:
                return "", 0.0

        except Exception as e:
            self.logger.error(f"AI answer generation failed: {e}")
            return "", 0.0

    async def _call_ai_model(self, prompt: str) -> str:
        """Call AI model with prompt."""
        try:
            response = self.model.generate_content(prompt)
            return response.text if response and response.text else ""
        except Exception as e:
            self.logger.error(f"AI model call failed: {e}")
            return ""

    def _calculate_answer_confidence(self, answer: str, question: str, context: str) -> float:
        """Calculate confidence score for generated answer."""
        try:
            confidence_score = 0.0

            # Length appropriateness
            answer_words = len(answer.split())
            if 10 <= answer_words <= 200:
                confidence_score += 0.3
            elif answer_words > 5:
                confidence_score += 0.1

            # Relevant content indicators
            question_words = set(question.lower().split())
            answer_words_set = set(answer.lower().split())
            overlap = len(question_words.intersection(answer_words_set))
            if overlap > 0:
                confidence_score += min(overlap / len(question_words), 0.3)

            # Legal reference presence
            legal_refs = 0
            for pattern in self.legal_patterns.values():
                if pattern.search(answer):
                    legal_refs += 1
            confidence_score += min(legal_refs * 0.1, 0.2)

            # Context relevance
            context_words = set(context.lower().split())
            context_overlap = len(answer_words_set.intersection(context_words))
            if context_overlap > 5:
                confidence_score += 0.2

            # Indonesian language quality
            indonesian_indicators = ['yang', 'dengan', 'dalam', 'pada', 'untuk', 'adalah', 'dari']
            indonesian_count = sum(1 for word in indonesian_indicators if word in answer.lower())
            confidence_score += min(indonesian_count / len(indonesian_indicators), 0.1)

            return min(confidence_score, 1.0)

        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence

    def _extract_citations(self, answer: str, context: str) -> List[Citation]:
        """Extract citations from answer based on context."""
        citations = []

        try:
            # Find legal references in answer
            for pattern_name, pattern in self.legal_patterns.items():
                matches = pattern.finditer(answer)
                for match in matches:
                    # Find corresponding text in context
                    match_text = match.group(0)
                    context_matches = pattern.finditer(context)

                    for context_match in context_matches:
                        if context_match.group(0).lower() == match_text.lower():
                            # Extract surrounding context
                            start = max(0, context_match.start() - 50)
                            end = min(len(context), context_match.end() + 50)
                            citation_text = context[start:end].strip()

                            citation = Citation(
                                text=citation_text,
                                source_section=match_text,
                                relevance_score=0.8,  # High relevance for exact matches
                                start_position=context_match.start(),
                                end_position=context_match.end()
                            )
                            citations.append(citation)

                            if len(citations) >= self.config.qa_engine.max_context_sources:
                                break

                if len(citations) >= self.config.qa_engine.max_context_sources:
                    break

        except Exception as e:
            self.logger.warning(f"Citation extraction failed: {e}")

        return citations

    def _generate_follow_up_questions(self, original_question: str, answer: str) -> List[str]:
        """Generate relevant follow-up questions."""
        follow_ups = []

        try:
            question_lower = original_question.lower()
            answer_lower = answer.lower()

            # Based on question type, suggest follow-ups
            if 'definisi' in question_lower or 'pengertian' in question_lower:
                follow_ups.append("Bagaimana implementasi praktis dari definisi ini?")
                follow_ups.append("Apa contoh penerapan dalam kasus nyata?")

            if 'sanksi' in answer_lower or 'pidana' in answer_lower:
                follow_ups.append("Apa saja faktor yang mempengaruhi berat ringannya sanksi?")
                follow_ups.append("Bagaimana prosedur pengenaan sanksi ini?")

            if 'prosedur' in question_lower or 'bagaimana' in question_lower:
                follow_ups.append("Apa saja dokumen yang diperlukan untuk prosedur ini?")
                follow_ups.append("Berapa lama waktu yang dibutuhkan untuk proses ini?")

            if 'syarat' in answer_lower or 'ketentuan' in answer_lower:
                follow_ups.append("Apa konsekuensi jika syarat tidak dipenuhi?")
                follow_ups.append("Apakah ada pengecualian untuk ketentuan ini?")

            # Limit to 3 follow-up questions
            return follow_ups[:3]

        except Exception as e:
            self.logger.warning(f"Follow-up question generation failed: {e}")
            return []

    def _update_average_metrics(self, confidence: float, processing_time: float) -> None:
        """Update average confidence and processing time."""
        total_questions = self.stats['total_questions']

        # Update average confidence
        current_avg_conf = self.stats['average_confidence']
        self.stats['average_confidence'] = (
            (current_avg_conf * (total_questions - 1) + confidence) / total_questions
        )

        # Update average processing time
        current_avg_time = self.stats['average_processing_time']
        self.stats['average_processing_time'] = (
            (current_avg_time * (total_questions - 1) + processing_time) / total_questions
        )

    async def ask(self, question: str, context: str, question_type: Optional[QuestionType] = None) -> QAResult:
        """Convenience method for asking questions."""
        request = QARequest(
            question=question,
            context=context,
            question_type=question_type,
            include_citations=True,
            legal_focus=True
        )
        return await self.answer_question(request)

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        success_rate = (
            self.stats['successful_answers'] / self.stats['total_questions']
            if self.stats['total_questions'] > 0 else 0
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
            'total_questions': 0,
            'successful_answers': 0,
            'average_confidence': 0.0,
            'average_processing_time': 0.0,
            'question_type_counts': {qt.value: 0 for qt in QuestionType},
            'citation_counts': 0
        }

    def supports_question_type(self, question_type: QuestionType) -> bool:
        """Check if question type is supported."""
        return question_type in QuestionType

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the QA engine."""
        return {
            'name': 'Legal Document QA Engine',
            'description': 'AI-powered question-answering for Indonesian legal documents',
            'supported_question_types': [qt.value for qt in QuestionType],
            'features': [
                'Context-aware responses',
                'Citation extraction',
                'Question type classification',
                'Follow-up question generation',
                'Indonesian legal optimization',
                'Confidence scoring',
                'Multi-hop reasoning support'
            ],
            'configuration': {
                'max_question_length': self.config.qa_engine.max_question_length,
                'max_context_length': self.config.qa_engine.max_context_length,
                'confidence_threshold': self.config.qa_engine.confidence_threshold,
                'include_citations': self.config.qa_engine.include_citations,
                'model_name': self.config.model.model_name
            },
            'status': 'available' if self.model else 'unavailable'
        }

    def __repr__(self) -> str:
        """String representation of QA engine."""
        return f"QAEngine(model={self.config.model.model_name}, available={self.model is not None})"
