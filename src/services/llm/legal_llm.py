"""
Legal LLM Service for generating answers based on legal context
"""

import time
from typing import List, Dict, Any, Optional

from src.services.search.vector_search import SearchResult
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
        context: List[SearchResult],
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
            # Analyze question type for better targeting
            question_type = self._analyze_question_type(query)

            # Build context focused on the specific question
            focused_context = self._build_focused_context(query, context, question_type)

            # Generate direct answer using improved prompt
            direct_prompt = ANSWER_WITH_CITATIONS.format(
                question=query,
                contexts=focused_context
            )

            answer = await self._generate_llm_response(direct_prompt, temperature, max_tokens)

            # If answer is too generic, try to get more specific response
            if self._is_answer_too_generic(answer, query):
                specific_prompt = self._build_specific_prompt(query, context, question_type)
                answer = await self._generate_llm_response(specific_prompt, temperature, max_tokens)

            # Calculate confidence based on relevance
            confidence = self._calculate_confidence(context)

            duration_ms = (time.time() - start_time) * 1000

            return {
                "answer": answer,
                "sources": [result.to_dict() for result in context[:5]],  # Top 5 sources as dicts
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

    def _analyze_question_type(self, query: str) -> str:
        """Analyze question type to provide targeted responses."""
        query_lower = query.lower()

        # Enhanced question type detection with more patterns
        if any(phrase in query_lower for phrase in ["pasal apa", "undang-undang apa", "uu apa", "peraturan apa", "diatur dalam", "berdasarkan pasal"]):
            return "specific_law_reference"
        elif any(phrase in query_lower for phrase in ["definisi", "pengertian", "arti", "maksud", "yang dimaksud", "adalah"]):
            return "definition"
        elif any(phrase in query_lower for phrase in ["sanksi", "pidana", "hukuman", "denda", "penalti", "ancaman"]):
            return "sanctions"
        elif any(phrase in query_lower for phrase in ["bagaimana", "cara", "prosedur", "mekanisme", "tata cara", "langkah"]):
            return "procedure"
        elif any(phrase in query_lower for phrase in ["siapa", "pihak", "lembaga", "instansi", "kewenangan", "tanggung jawab"]):
            return "authority"
        elif any(phrase in query_lower for phrase in ["kapan", "waktu", "jangka", "batas waktu", "periode"]):
            return "temporal"
        elif any(phrase in query_lower for phrase in ["berapa", "jumlah", "besaran", "tarif", "biaya"]):
            return "quantitative"
        else:
            return "general"

    def _build_focused_context(self, query: str, context: List[SearchResult], question_type: str) -> str:
        """Build context focused on answering the specific question."""
        if not context:
            return "Tidak ada konteks yang ditemukan."

        # Filter and rank results based on question type
        relevant_results = []
        for result in context[:7]:  # Increased from 5 to 7 for better coverage
            content = (result.content or "").strip()
            citation = result.citation_string or "(tanpa citation)"

            # Score relevance based on question type
            relevance_score = self._calculate_content_relevance(content, query, question_type)

            relevant_results.append({
                'result': result,
                'relevance': relevance_score,
                'citation': citation,
                'content': content,
                'search_score': result.score  # Include original search score
            })

        # Sort by combined relevance and search score
        relevant_results.sort(key=lambda x: (x['relevance'] * 0.7 + x['search_score'] * 0.3), reverse=True)

        if not relevant_results:
            return "Tidak ditemukan informasi yang relevan dengan pertanyaan."

        # Always include at least the top 3 results for better context (increased from 2)
        min_results = min(3, len(relevant_results))
        top_results = relevant_results[:min_results]

        # Include additional results with more generous threshold
        for item in relevant_results[min_results:]:
            if item['relevance'] > 0.3 or item['search_score'] > 0.7:  # More generous inclusion
                top_results.append(item)

        # Limit to max 6 results to avoid context overflow
        top_results = top_results[:6]

        # Build focused context with better formatting
        context_parts = []
        for i, item in enumerate(top_results, 1):
            # Add relevance indicator for debugging
            context_parts.append(f"[{i}] {item['citation']}\n{item['content']}")

        return "\n\n".join(context_parts)

    def _calculate_content_relevance(self, content: str, query: str, question_type: str) -> float:
        """Calculate how relevant content is to the specific question."""
        content_lower = content.lower()
        query_lower = query.lower()

        # Expanded stop words for better filtering
        stop_words = {"apa", "dalam", "tentang", "dan", "atau", "yang", "adalah", "dari", "di", "ke", "untuk",
                     "dengan", "pada", "oleh", "ini", "itu", "akan", "dapat", "harus", "telah", "sudah", "juga", "serta"}

        # Extract key terms from query
        query_terms = set(term for term in query_lower.split() if term not in stop_words and len(term) > 2)
        content_words = content_lower.split()
        content_terms = set(content_words)

        if not query_terms:
            return 0.4  # Base score for empty query terms

        # Base relevance from term overlap - more generous scoring
        overlap = len(query_terms.intersection(content_terms))
        base_score = 0.4 if overlap > 0 else 0.3  # More generous base scores

        # Enhanced term density calculation
        if len(query_terms) > 0:
            term_density = overlap / len(query_terms)
            base_score += term_density * 0.4

        # Position-based relevance (early mentions are more important)
        position_bonus = 0
        for term in query_terms:
            if term in content_words[:50]:  # First 50 words
                position_bonus += 0.1
        base_score += min(position_bonus, 0.3)

        # Enhanced question type-specific boosting
        if question_type == "specific_law_reference":
            if any(term in content_lower for term in ["pasal", "ayat", "huruf", "angka", "uu", "undang-undang", "peraturan"]):
                base_score += 0.3
        elif question_type == "definition":
            if any(term in content_lower for term in ["adalah", "merupakan", "dimaksud", "definisi", "pengertian", "berarti"]):
                base_score += 0.4
        elif question_type == "sanctions":
            if any(term in content_lower for term in ["sanksi", "pidana", "denda", "hukuman", "penalti", "ancaman"]):
                base_score += 0.3
        elif question_type == "procedure":
            if any(term in content_lower for term in ["prosedur", "tata cara", "mekanisme", "langkah", "tahap"]):
                base_score += 0.3
        elif question_type == "authority":
            if any(term in content_lower for term in ["kewenangan", "tanggung jawab", "lembaga", "instansi", "pihak"]):
                base_score += 0.3
        elif question_type == "temporal":
            if any(term in content_lower for term in ["waktu", "jangka", "periode", "tanggal", "bulan", "tahun"]):
                base_score += 0.3
        elif question_type == "quantitative":
            if any(term in content_lower for term in ["jumlah", "besaran", "tarif", "biaya", "nilai", "rupiah"]):
                base_score += 0.3
        elif question_type == "general":
            # For general questions, be more inclusive
            base_score += 0.2

        # Semantic similarity bonus for related terms
        semantic_bonus = self._calculate_semantic_similarity(query_terms, content_terms)
        base_score += semantic_bonus * 0.2

        return min(base_score, 1.0)

    def _calculate_semantic_similarity(self, query_terms: set, content_terms: set) -> float:
        """Calculate simple semantic similarity based on related terms."""
        # Simple semantic groups for Indonesian legal terms
        semantic_groups = [
            {"hukum", "peraturan", "undang-undang", "uu", "norma", "ketentuan"},
            {"sanksi", "pidana", "denda", "hukuman", "penalti", "ancaman"},
            {"kewenangan", "wewenang", "otoritas", "kuasa", "hak"},
            {"tanggung jawab", "kewajiban", "tugas", "fungsi"},
            {"lembaga", "instansi", "organisasi", "badan", "dinas"},
            {"prosedur", "tata cara", "mekanisme", "proses", "tahap"},
            {"lingkungan", "ekosistem", "alam", "hijau", "konservasi"},
            {"ekonomi", "finansial", "keuangan", "investasi", "modal"}
        ]

        similarity_score = 0
        for group in semantic_groups:
            query_in_group = len(query_terms.intersection(group))
            content_in_group = len(content_terms.intersection(group))
            if query_in_group > 0 and content_in_group > 0:
                similarity_score += min(query_in_group, content_in_group) * 0.1

        return min(similarity_score, 0.5)

    def _build_specific_prompt(self, query: str, context: List[SearchResult], question_type: str) -> str:
        """Build a specific prompt based on question type."""
        focused_context = self._build_focused_context(query, context, question_type)

        # Enhanced prompts for different question types
        if question_type == "specific_law_reference":
            return f"""Pertanyaan: {query}

Konteks hukum:
{focused_context}

Instruksi: Identifikasi secara spesifik pasal dan undang-undang yang mengatur topik yang ditanyakan.
Sebutkan nama UU dan nomor pasal yang tepat. Jangan memberikan informasi umum.
Format jawaban: "Berdasarkan [UU/PP] [Nomor]/[Tahun], Pasal [X] ayat ([Y])..."

Jawaban:"""

        elif question_type == "definition":
            return f"""Pertanyaan: {query}

Konteks hukum:
{focused_context}

Instruksi: Berikan definisi yang tepat berdasarkan konteks hukum yang tersedia.
Kutip secara eksplisit sumber definisi dengan format citation yang diberikan.
Fokus pada definisi resmi dari peraturan perundang-undangan.

Jawaban:"""

        elif question_type == "sanctions":
            return f"""Pertanyaan: {query}

Konteks hukum:
{focused_context}

Instruksi: Identifikasi jenis sanksi, besaran denda, atau ancaman pidana yang relevan.
Sebutkan secara spesifik pasal yang mengatur sanksi tersebut.
Jika ada sanksi administratif dan pidana, sebutkan keduanya.

Jawaban:"""

        elif question_type == "procedure":
            return f"""Pertanyaan: {query}

Konteks hukum:
{focused_context}

Instruksi: Jelaskan tahapan atau prosedur yang dimaksud berdasarkan konteks hukum.
Urutkan langkah-langkah secara sistematis jika ada.
Kutip pasal yang mengatur setiap tahap prosedur.

Jawaban:"""

        # Default to enhanced general prompt
        return ANSWER_WITH_CITATIONS.format(question=query, contexts=focused_context)

    def _is_answer_too_generic(self, answer: str, query: str) -> bool:
        """Check if answer is too generic for the specific question asked."""
        query_lower = query.lower()
        answer_lower = answer.lower()

        # Enhanced generic answer detection
        if any(phrase in query_lower for phrase in ["pasal apa", "undang-undang apa", "diatur dalam"]):
            if not any(term in answer_lower for term in ["pasal", "uu", "undang-undang", "peraturan"]):
                return True

        if any(phrase in query_lower for phrase in ["definisi", "pengertian", "arti"]):
            if not any(term in answer_lower for term in ["adalah", "merupakan", "dimaksud", "definisi"]):
                return True

        if any(phrase in query_lower for phrase in ["sanksi", "pidana", "denda"]):
            if not any(term in answer_lower for term in ["sanksi", "pidana", "denda", "hukuman"]):
                return True

        # Check for overly short answers
        if len(answer.split()) < 10:
            return True

        # Check for generic phrases that indicate unhelpful answers
        generic_phrases = [
            "tidak dapat memberikan informasi",
            "silakan konsultasi",
            "informasi lebih lanjut",
            "mohon maaf tidak dapat",
            "berdasarkan konteks yang terbatas"
        ]

        if any(phrase in answer_lower for phrase in generic_phrases):
            return True

        return False

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

    def _calculate_confidence(self, context: List[SearchResult]) -> float:
        """Calculate confidence score based on context quality"""
        if not context:
            return 0.0

        # Simple confidence based on number and relevance of sources
        base_score = min(len(context) * 0.2, 1.0)

        # Boost for exact matches
        exact_matches = sum(1 for c in context if c.score > 0.8)
        boost = exact_matches * 0.1

        return min(base_score + boost, 1.0)

    def _extract_citations(self, answer: str, context: List[SearchResult]) -> List[Dict[str, str]]:
        """Extract citations from answer and context"""
        citations = []

        # Extract citations from context
        for result in context:
            if result.unit_type:
                citation = {
                    "pasal": getattr(result, 'pasal_number', ''),
                    "ayat": getattr(result, 'ayat_number', ''),
                    "text": (result.content or '')[:100] + "...",
                    "document_title": getattr(result, 'document_title', ''),
                    "unit_type": result.unit_type,
                    "citation_string": result.citation_string or ''
                }
                citations.append(citation)

        return citations
