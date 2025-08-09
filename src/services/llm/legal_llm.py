"""
Legal LLM Service for generating answers based on legal context
"""

import time
from typing import List, Dict, Any, Optional

from src.services.llm.factory import LLMFactory
from src.services.llm.prompt.prompt import ANSWER_WITH_CITATIONS, SUMMARIZE_UNITS, join_contexts
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
            # Build verbatim block from the most relevant units
            verbatim_block, summary_context = self._build_verbatim_and_summary_context(query, context)
            
            # Build prompt for summary constrained to the same units
            summarize_prompt = SUMMARIZE_UNITS.format(contexts=summary_context)
            summary = await self._generate_llm_response(summarize_prompt, temperature, max_tokens)
            
            # Compose final answer: verbatim first, then summary
            answer = (
                "Teks asli (kutipan hukum):\n\n" + verbatim_block.strip() + "\n\n"
                "Ringkasan:\n\n" + (summary.strip() or "-")
            )
            
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
    
    def _build_verbatim_and_summary_context(self, query: str, context: List[Dict[str, Any]]) -> (str, str):
        """Construct verbatim legal text block and a summary context limited to same units.
        Preference order: explicit sources, then fts/vector. Group by unit_id to avoid duplicates.
        """
        if not context:
            return ("Tidak ditemukan dalam konteks yang diberikan.", "(kosong)")

        # Prioritize explicit results
        def score_item(it: Dict[str, Any]) -> int:
            st = (it.get("source_type") or "").lower()
            if st.startswith("explicit"):
                return 3
            if st.startswith("fts"):
                return 2
            if st.startswith("vector"):
                return 1
            return 0

        # Deduplicate by unit_id, keep best scored first 5
        seen = set()
        picked: List[Dict[str, Any]] = []
        for it in sorted(context, key=lambda x: (score_item(x), x.get("score", 0)), reverse=True):
            uid = it.get("unit_id") or f"{it.get('document',{})}:{it.get('citation','')}"
            if uid in seen:
                continue
            txt = (it.get("text") or "").strip()
            if not txt:
                continue
            seen.add(uid)
            picked.append(it)
            if len(picked) >= 5:
                break

        if not picked:
            return ("Tidak ditemukan dalam konteks yang diberikan.", "(kosong)")

        # Build verbatim block and summary context (more compact for LLM)
        verbatim_parts: List[str] = []
        summary_rows: List[str] = []
        for it in picked:
            cit = it.get("citation") or it.get("citation_string") or "(tanpa citation)"
            txt = (it.get("text") or "").strip()
            verbatim_parts.append(f"{cit}\n{txt}")
            # For summary, keep a shorter slice per unit to reduce tokens but preserve content
            stxt = txt if len(txt) <= 2000 else txt[:2000] + "…"
            summary_rows.append(f"- {cit}\n{stxt}")

        verbatim_block = "\n\n".join(verbatim_parts)
        summary_context = "\n\n".join(summary_rows)
        return verbatim_block, summary_context
    
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
