"""
Legal LLM Service for generating answers based on legal context
"""

import time
from typing import List, Dict, Any, Optional

from src.services.llm.factory import LLMFactory
from src.services.llm.prompt.prompt import ANSWER_WITH_CITATIONS, join_contexts
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LegalLLMService:
    """Service for generating legal answers using LLM"""
    
    def __init__(self):
        """Initialize LLM service"""
        self.provider = LLMFactory.create_provider()
        self.max_context_length = 8000
        
    async def generate_answer(
        self,
        query: str,
        context: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate legal answer based on search context
        
        Args:
            query: User question
            context: Search results as context
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Dict with answer, sources, and metadata
        """
        start_time = time.time()
        
        try:
            # Prepare context
            context_text = self._prepare_context(context)
            
            # Build prompt
            prompt = self._build_prompt(query, context_text)
            
            # Generate answer using LLM
            answer = await self._generate_llm_response(prompt, temperature, max_tokens)
            
            # Calculate confidence based on relevance
            confidence = self._calculate_confidence(context)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "answer": answer,
                "sources": context[:5],  # Top 5 sources
                "confidence": confidence,
                "duration_ms": duration_ms,
                "model_used": self.provider.get_model_info()['model']
            }
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "answer": "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda.",
                "sources": [],
                "confidence": 0.0,
                "duration_ms": (time.time() - start_time) * 1000,
                "error": str(e)
            }
    
    def _prepare_context(self, context: List[Dict[str, Any]]) -> str:
        """Prepare search results as context for LLM"""
        if not context:
            return "Tidak ditemukan dokumen terkait."
        
        context_parts = []
        for idx, item in enumerate(context[:5]):  # Top 5 results
            citation = item.get('citation', '')
            text = item.get('text', '')[:1000]  # Truncate long texts
            
            context_parts.append(f"""
Sumber {idx+1}: {citation}
Isi: {text}
""")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM using template"""
        return ANSWER_WITH_CITATIONS.format(
            question=query,
            contexts=context
        )
    
    async def _generate_llm_response(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate LLM response using configured provider"""
        try:
            response = await self.provider.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "Maaf, terjadi kesalahan dalam menghasilkan jawaban. Silakan coba lagi."
        
    def _calculate_confidence(self, context: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on context quality"""
        if not context:
            return 0.0
        
        # Simple confidence based on number and relevance of sources
        base_score = min(len(context) * 0.2, 1.0)
        
        # Boost for exact matches
        exact_matches = sum(1 for c in context if c.get('score', 0) > 0.8)
        boost = exact_matches * 0.1
        
        return min(base_score + boost, 1.0)
    
    def _extract_citations(self, answer: str, context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract citations from answer and context"""
        citations = []
        
        # Extract citations from context
        for item in context:
            if 'unit_type' in item and 'number_label' in item:
                citation = {
                    "pasal": item.get('pasal_number', ''),
                    "ayat": item.get('ayat_number', ''),
                    "text": item.get('content', '')[:100] + "...",
                    "document_title": item.get('document_title', ''),
                    "unit_type": item.get('unit_type', ''),
                    "number_label": item.get('number_label', '')
                }
                citations.append(citation)
        
        return citations
