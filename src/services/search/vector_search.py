"""
Vector-only search service for Legal RAG system.

Implements dense semantic search with citation parsing for explicit queries.
Replaces hybrid FTS+vector approach with pure vector search optimized for
Indonesian legal documents.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import text
from sqlalchemy.orm import Session

from ...config.settings import settings
from ...db.models import LegalDocument, LegalUnit, DocumentVector, DocForm, DocStatus
from ...db.session import get_db_session
from ...services.citation import CitationMatch, is_explicit_citation, get_best_citation_match
from ...services.embedding.embedder import JinaV4Embedder
from ...services.search.query_optimizer import QueryOptimizationService, get_query_optimizer
from ...utils.logging import get_logger, log_timing

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Individual search result with metadata."""

    id: str
    content: str
    citation_string: str
    score: float
    unit_type: str
    unit_id: str
    doc_form: Optional[str] = None
    doc_year: Optional[int] = None
    doc_number: Optional[str] = None
    hierarchy_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert SearchResult to dictionary for API/LLM compatibility."""
        return {
            "id": self.id,
            "content": self.content,
            "text": self.content,  # Alias for compatibility
            "citation": self.citation_string,
            "citation_string": self.citation_string,
            "score": self.score,
            "unit_type": self.unit_type,
            "unit_id": self.unit_id,
            "doc_form": self.doc_form,
            "doc_year": self.doc_year,
            "doc_number": self.doc_number,
            "hierarchy_path": self.hierarchy_path,
            "metadata": self.metadata or {}
        }


@dataclass
class SearchFilters:
    """Search filters for narrowing results."""

    doc_forms: Optional[List[str]] = None
    doc_years: Optional[List[int]] = None
    doc_numbers: Optional[List[str]] = None
    doc_status: Optional[List[str]] = None
    bab_numbers: Optional[List[str]] = None
    pasal_numbers: Optional[List[str]] = None


class VectorSearchService:
    """
    Vector-only search service for Indonesian legal documents.

    Implements:
    1. Citation parsing for explicit queries (direct SQL lookup)
    2. Contextual vector search with embedding similarity
    3. Optional reranking with cross-encoder
    """

    def __init__(
        self,
        embedder: Optional[JinaV4Embedder] = None,
        default_k: int = 15,
        min_citation_confidence: float = 0.60,
        query_optimizer: Optional[QueryOptimizationService] = None
    ):
        """
        Initialize vector search service.

        Args:
            embedder: Optional JinaV4Embedder instance
            default_k: Default number of results to retrieve
            min_citation_confidence: Minimum confidence for citation matching
            query_optimizer: Optional query optimizer instance
        """
        self.embedder = embedder or JinaV4Embedder(default_dims=384)
        self.default_k = default_k
        self.min_citation_confidence = min_citation_confidence
        self.query_optimizer = query_optimizer or get_query_optimizer()

        logger.info("Initialized VectorSearchService with dense semantic search and query optimization")

    def _detect_legal_keywords(self, query: str) -> bool:
        """
        Simple heuristic to detect queries with legal keywords that might benefit
        from optimized search strategies.

        Args:
            query: Search query text

        Returns:
            True if query contains legal keywords
        """
        legal_keywords = [
            'pasal', 'ayat', 'huruf', 'angka', 'bab', 'bagian',
            'sanksi', 'pidana', 'denda', 'hukuman', 'pelanggaran',
            'definisi', 'pengertian', 'ketentuan', 'peraturan',
            'undang', 'perpres', 'permen', 'perda', 'pojk',
            'tanggung jawab', 'kewajiban', 'hak', 'wewenang'
        ]

        query_lower = query.lower()
        keyword_count = sum(1 for keyword in legal_keywords if keyword in query_lower)

        # Return True if query contains 2+ legal keywords or specific legal patterns
        return keyword_count >= 2 or any(pattern in query_lower for pattern in [
            'uu no', 'uu nomor', 'pp no', 'pp nomor', 'perpres no',
            'pasal', 'ayat', 'sanksi pidana', 'definisi'
        ])

    async def search_async(
        self,
        query: str,
        k: int = None,
        filters: Optional[SearchFilters] = None,
        use_reranking: bool = False,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Async unified search interface with concurrent processing capabilities.

        Args:
            query: Search query text
            k: Number of results to return (defaults to default_k)
            filters: Optional search filters
            use_reranking: Whether to apply reranking
            session_id: Optional session ID for logging

        Returns:
            Search response with results and metadata
        """
        start_time = time.time()
        k = k or self.default_k
        original_query = query

        try:
            # Step 1: Optimize query for better performance (concurrent with citation check)
            optimize_task = asyncio.to_thread(self.query_optimizer.optimize_query, query)
            citation_task = asyncio.to_thread(is_explicit_citation, query, self.min_citation_confidence)

            # Run both operations concurrently
            (optimized_query, query_analysis), is_citation = await asyncio.gather(
                optimize_task, citation_task, return_exceptions=False
            )

            # Use optimized query for processing
            query = optimized_query

            # Step 2: Check for multi-part queries (contextual + citation)
            query_parts = await asyncio.to_thread(self._decompose_multi_part_query, query)

            if len(query_parts) > 1:
                logger.debug(f"Processing async multi-part query with {len(query_parts)} parts: {query}")
                results = await self._handle_multi_part_query_async(query_parts, k, filters, use_reranking)
                search_type = "multi_part_async"
            else:
                # Single query processing
                single_query = query_parts[0] if query_parts else query

                # Step 3: Route to appropriate search handler
                if is_citation:
                    logger.debug(f"Processing explicit citation query: {single_query}")
                    results = await self._handle_explicit_citation_async(single_query, k, filters)
                    search_type = "explicit_citation"
                else:
                    # Legal keyword detection
                    has_legal_keywords = await asyncio.to_thread(self._detect_legal_keywords, single_query)
                    if has_legal_keywords:
                        logger.debug(f"Processing legal keyword query with optimization: {single_query}")

                    logger.debug(f"Processing contextual semantic search: {single_query}")
                    results = await self._handle_contextual_search_async(single_query, k, filters, use_reranking)
                    search_type = "contextual_semantic"

            # Step 3: Build response with metadata
            duration_ms = (time.time() - start_time) * 1000

            response = {
                "results": results,
                "metadata": {
                    "query": original_query,
                    "optimized_query": query,
                    "search_type": search_type,
                    "total_results": len(results),
                    "duration_ms": round(duration_ms, 2),
                    "query_analysis": query_analysis,
                    "query_parts": len(query_parts) if len(query_parts) > 1 else 1
                }
            }

            # Log search for analytics
            await asyncio.to_thread(self._log_search, query, len(results), duration_ms, session_id)

            return response

        except Exception as e:
            logger.error(f"Async search failed for query '{original_query}': {e}")
            return {
                "results": [],
                "metadata": {
                    "query": original_query,
                    "error": str(e),
                    "search_type": "error",
                    "total_results": 0,
                    "duration_ms": (time.time() - start_time) * 1000
                }
            }

    def search(
        self,
        query: str,
        k: int = None,
        filters: Optional[SearchFilters] = None,
        use_reranking: bool = False,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Unified search interface supporting both explicit citations and contextual queries.
        Enhanced with multi-part query decomposition for mixed queries.

        Args:
            query: Search query text
            k: Number of results to return (defaults to default_k)
            filters: Optional search filters
            use_reranking: Whether to apply reranking
            session_id: Optional session ID for logging

        Returns:
            Search response with results and metadata
        """
        start_time = time.time()
        k = k or self.default_k
        original_query = query

        try:
            # Step 1: Optimize query for better performance (KISS approach)
            optimized_query, query_analysis = self.query_optimizer.optimize_query(query)

            # Use optimized query for processing, but keep original for logging
            query = optimized_query

            # Step 2: Check for multi-part queries (contextual + citation)
            query_parts = self._decompose_multi_part_query(query)

            if len(query_parts) > 1:
                logger.debug(f"Processing multi-part query with {len(query_parts)} parts: {query}")
                results = self._handle_multi_part_query(query_parts, k, filters, use_reranking)
                search_type = "multi_part"
            else:
                # Single query processing (existing logic)
                single_query = query_parts[0] if query_parts else query

                # Step 3: Check if query contains explicit citations
                if is_explicit_citation(single_query, self.min_citation_confidence):
                    logger.debug(f"Processing explicit citation query: {single_query}")
                    results = self._handle_explicit_citation(single_query, k, filters)
                    search_type = "explicit_citation"
                else:
                    # Step 4: Optimize based on legal keyword presence
                    has_legal_keywords = self._detect_legal_keywords(single_query)
                    if has_legal_keywords:
                        logger.debug(f"Processing legal keyword query with optimization: {single_query}")
                        # Use slightly more aggressive k for legal keyword queries
                        optimized_k = min(k + 5, k * 2)
                        results = self._handle_contextual_search(single_query, optimized_k, filters, use_reranking)
                        # Trim back to requested k after reranking if needed
                        results = results[:k] if len(results) > k else results
                        search_type = "contextual_semantic_legal"
                    else:
                        logger.debug(f"Processing contextual semantic search: {single_query}")
                        results = self._handle_contextual_search(single_query, k, filters, use_reranking)
                        search_type = "contextual_semantic"

            duration_ms = int((time.time() - start_time) * 1000)

            # Log search for analytics
            self._log_search(query, len(results), duration_ms, search_type, session_id)

            return {
                "results": results,
                "metadata": {
                    "original_query": original_query,
                    "optimized_query": query,
                    "query_type": query_analysis.query_type.value if query_analysis else "unknown",
                    "query_confidence": query_analysis.confidence_score if query_analysis else 0.0,
                    "search_type": search_type,
                    "total_results": len(results),
                    "duration_ms": duration_ms,
                    "filters_applied": filters is not None,
                    "reranking_used": use_reranking and search_type == "contextual_semantic",
                    "optimization_applied": query != original_query
                }
            }

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return {
                "results": [],
                "metadata": {
                    "original_query": original_query,
                    "optimized_query": query,
                    "search_type": "error",
                    "total_results": 0,
                    "duration_ms": int((time.time() - start_time) * 1000),
                    "error": str(e)
                }
            }

    def _handle_explicit_citation(
        self,
        query: str,
        k: int,
        filters: Optional[SearchFilters]
    ) -> List[SearchResult]:
        """
        Handle explicit citation queries with direct SQL lookup.

        Args:
            query: Citation query text
            k: Number of results
            filters: Optional filters

        Returns:
            List of matching legal units
        """
        citation_match = get_best_citation_match(query)
        if not citation_match:
            logger.warning(f"No valid citation parsed from: {query}")
            return []

        logger.debug(f"Parsed citation: {citation_match.to_dict()}")

        with get_db_session() as db:
            return self._lookup_citation_units(db, citation_match, k, filters)

    def _handle_contextual_search(
        self,
        query: str,
        k: int,
        filters: Optional[SearchFilters],
        use_reranking: bool = False
    ) -> List[SearchResult]:
        """
        Handle contextual semantic queries with vector search.

        Args:
            query: Semantic query text
            k: Number of results
            filters: Optional filters
            use_reranking: Whether to apply reranking

        Returns:
            List of semantically similar legal units
        """
        # Step 1: Enhanced query preprocessing
        original_query = query
        normalized_query = self._normalize_query(query)

        # Step 1.5: Query expansion for better context
        expanded_query = self._expand_query_context(normalized_query)

        # Step 2: Embed query with fallback
        query_embedding = self._embed_query(expanded_query)
        if not query_embedding:
            logger.warning(f"Failed to embed expanded query, trying original: {query}")
            query_embedding = self._embed_query(normalized_query)
            if not query_embedding:
                logger.error("Failed to embed query completely")
                return []

        # Step 3: Enhanced vector search with adaptive k
        adaptive_k = min(k * 2, 30)  # Get more results for better filtering
        with get_db_session() as db:
            raw_results = self._vector_search(db, query_embedding, adaptive_k, filters)

        # Step 3.5: Filter and re-rank results based on query context
        filtered_results = self._filter_by_query_relevance(original_query, raw_results)

        # Step 4: Limit to requested k after filtering
        results = filtered_results[:k]

        # Step 5: Optional reranking
        if use_reranking and results:
            results = self._rerank_results(original_query, results)

        return results

    def _normalize_query(self, query: str) -> str:
        """
        Enhanced query normalization for better embedding quality.

        Args:
            query: Raw query text

        Returns:
            Normalized query text
        """
        if not query or not query.strip():
            return ""

        # Lowercasing
        normalized = query.lower().strip()

        # Preserve important legal punctuation before removing
        # Convert common legal abbreviations
        legal_replacements = {
            'uu no.': 'undang undang nomor',
            'pp no.': 'peraturan pemerintah nomor',
            'perpres no.': 'peraturan presiden nomor',
            'permen no.': 'peraturan menteri nomor',
            'pasal ke-': 'pasal',
            'ayat ke-': 'ayat'
        }

        for old, new in legal_replacements.items():
            normalized = normalized.replace(old, new)

        # Remove punctuation but preserve legal structure indicators
        normalized = re.sub(r'[^\w\s()]', ' ', normalized)

        # Preserve parentheses for ayat references
        normalized = re.sub(r'\s*\(\s*', ' (', normalized)
        normalized = re.sub(r'\s*\)\s*', ') ', normalized)

        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)

        # Enhanced Indonesian normalization
        normalized = self._simple_indonesian_normalize(normalized)

        return normalized.strip()

    def _simple_indonesian_normalize(self, text: str) -> str:
        """
        Enhanced Indonesian text normalization for legal documents.
        """
        words = text.split()
        normalized_words = []

        # Legal terms that should not be stemmed
        legal_terms = {
            'pasal', 'ayat', 'huruf', 'angka', 'undang', 'peraturan', 'pemerintah',
            'presiden', 'menteri', 'sanksi', 'pidana', 'administrasi', 'kewenangan',
            'tanggung', 'jawab', 'kewajiban', 'hukum', 'norma', 'ketentuan'
        }

        for word in words:
            original_word = word

            # Skip normalization for legal terms
            if word in legal_terms or len(word) <= 3:
                normalized_words.append(word)
                continue

            # Remove common prefixes (more conservative for legal text)
            prefix_removed = False
            for prefix in ['ber', 'ter', 'me', 'pe', 'di', 'ke', 'se']:
                if word.startswith(prefix) and len(word) > len(prefix) + 3:
                    word = word[len(prefix):]
                    prefix_removed = True
                    break

            # Remove common suffixes (more conservative)
            for suffix in ['kan', 'an', 'nya', 'lah', 'kah']:
                if word.endswith(suffix) and len(word) > len(suffix) + 3:
                    word = word[:-len(suffix)]
                    break

            # Don't over-stem - keep original if result is too short
            if len(word) < 3:
                word = original_word

            normalized_words.append(word)

        return ' '.join(normalized_words)

    async def _embed_query_async(self, query: str) -> List[float]:
        """Async query embedding with caching."""
        try:
            # Use thread pool for embedding to avoid blocking event loop
            embedding = await asyncio.to_thread(self.embedder.embed_query, query)
            return embedding
        except Exception as e:
            logger.error(f"Async query embedding failed: {e}")
            raise

    def _embed_query(self, query: str) -> List[float]:
        """
        Enhanced query embedding with retry logic.

        Args:
            query: Normalized query text

        Returns:
            Query embedding vector or None if failed
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for embedding")
            return None

        try:
            # Add context hint for legal domain
            enhanced_query = f"legal document query: {query}"

            embeddings = self.embedder.embed_texts(
                [enhanced_query],
                task="retrieval.query",
                dims=384
            )

            if embeddings and len(embeddings) > 0 and len(embeddings[0]) == 384:
                return embeddings[0]
            else:
                logger.warning("Invalid embedding dimensions received")
                return None

        except Exception as e:
            logger.error(f"Query embedding failed: {e}")

            # Fallback: try without enhancement
            try:
                logger.info("Trying fallback embedding without enhancement")
                embeddings = self.embedder.embed_texts(
                    [query],
                    task="retrieval.query",
                    dims=384
                )
                return embeddings[0] if embeddings else None
            except Exception as fallback_e:
                logger.error(f"Fallback embedding also failed: {fallback_e}")
                return None

    def _expand_query_context(self, query: str) -> str:
        """
        Expand query with relevant context for better retrieval.

        Args:
            query: Normalized query text

        Returns:
            Expanded query with additional context
        """
        if not query or len(query.strip()) == 0:
            return query

        # Legal domain keywords to add context
        legal_context_terms = {
            'sanksi': 'sanksi pidana administrasi',
            'izin': 'izin perizinan kewenangan',
            'kewajiban': 'kewajiban tanggung jawab',
            'hak': 'hak kewenangan otoritas',
            'prosedur': 'prosedur tata cara mekanisme',
            'lembaga': 'lembaga instansi organisasi',
            'lingkungan': 'lingkungan hidup konservasi',
            'investasi': 'investasi modal penanaman',
            'pajak': 'pajak retribusi pungutan',
            'perdagangan': 'perdagangan ekspor impor'
        }

        expanded_terms = []
        query_words = query.split()

        for word in query_words:
            expanded_terms.append(word)
            # Add contextual terms if found
            if word in legal_context_terms:
                context = legal_context_terms[word]
                # Add only 1-2 most relevant context words to avoid dilution
                context_words = context.split()[:2]
                expanded_terms.extend(context_words)

        # Limit expansion to avoid overwhelming the query
        if len(expanded_terms) > len(query_words) * 2:
            expanded_terms = expanded_terms[:len(query_words) * 2]

        expanded_query = ' '.join(expanded_terms)

        # If expansion made query too long, return original
        if len(expanded_query) > len(query) * 2:
            return query

        return expanded_query

    def _filter_by_query_relevance(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Filter and re-rank results based on query-specific relevance.

        Args:
            query: Original query text
            results: Raw search results from vector search

        Returns:
            Filtered and re-ranked results
        """
        if not results:
            return results

        # Simple query analysis
        query_lower = query.lower()
        is_specific_query = any(term in query_lower for term in
                               ['pasal', 'uu', 'undang-undang', 'peraturan', 'nomor'])

        filtered_results = []

        for result in results:
            content = (result.content or "").lower()
            citation = (result.citation_string or "").lower()

            # Base relevance score from vector similarity
            relevance_score = result.score

            # Boost for specific queries that find matching legal references
            if is_specific_query:
                if any(term in citation for term in ['uu', 'pasal', 'peraturan']):
                    relevance_score += 0.2
                if any(term in content for term in query_lower.split()[:3]):
                    relevance_score += 0.1

            # Content quality filters
            if len(content.strip()) < 20:  # Skip very short content
                relevance_score -= 0.3

            if 'tidak ditemukan' in content or 'tidak ada' in content:
                relevance_score -= 0.2

            # Keep results with decent relevance
            if relevance_score > 0.2:  # Lower threshold for more inclusive results
                # Update the score in result
                result.score = min(relevance_score, 1.0)
                filtered_results.append(result)

        # Sort by updated relevance score
        filtered_results.sort(key=lambda x: x.score, reverse=True)

        logger.debug(f"Filtered {len(results)} -> {len(filtered_results)} results")
        return filtered_results

    def _vector_search(
        self,
        db: Session,
        query_embedding: List[float],
        k: int,
        filters: Optional[SearchFilters]
    ) -> List[SearchResult]:
        """
        Perform vector search using HNSW cosine similarity.

        Args:
            db: Database session
            query_embedding: Query vector
            k: Number of results
            filters: Optional filters

        Returns:
            List of search results ordered by similarity
        """
        # Build base query with HNSW cosine similarity - return pasal citations
        base_query = """
        SELECT
            dv.unit_id,
            dv.doc_form,
            dv.doc_year,
            dv.doc_number,
            dv.hierarchy_path,
            lu.bm25_body AS content,
            lu.citation_string,
            lu.unit_type,
            lu.number_label AS pasal_number,
            ld.doc_title,
            ld.doc_form_short,
            (1 - (dv.embedding <=> :query_vector)) AS similarity_score
        FROM document_vectors dv
        JOIN legal_units lu ON lu.unit_id = dv.unit_id
        JOIN legal_documents ld ON ld.id = dv.document_id
        WHERE dv.content_type = 'pasal'
        AND lu.unit_type = 'PASAL'
        AND ld.doc_status = :doc_status
        """

        # Add filters
        filter_conditions = []
        params = {
            'query_vector': '[' + ','.join(map(str, query_embedding)) + ']',  # Vector string format for pgvector
            'doc_status': DocStatus.BERLAKU.value
        }

        if filters:
            if filters.doc_forms:
                filter_conditions.append("dv.doc_form = ANY(:doc_forms)")
                params['doc_forms'] = filters.doc_forms

            if filters.doc_years:
                filter_conditions.append("dv.doc_year = ANY(:doc_years)")
                params['doc_years'] = filters.doc_years

            if filters.doc_numbers:
                filter_conditions.append("dv.doc_number = ANY(:doc_numbers)")
                params['doc_numbers'] = filters.doc_numbers

            if filters.pasal_numbers:
                filter_conditions.append("dv.pasal_number = ANY(:pasal_numbers)")
                params['pasal_numbers'] = filters.pasal_numbers

        if filter_conditions:
            base_query += " AND " + " AND ".join(filter_conditions)

        # Add ordering and limit
        final_query = base_query + """
        ORDER BY dv.embedding <=> :query_vector
        LIMIT :limit
        """
        params['limit'] = k

        try:
            result = db.execute(text(final_query), params)
            rows = result.fetchall()

            search_results = []
            for row in rows:
                # Build proper citation string for unit
                citation = self._build_unit_citation(
                    row.doc_form_short or row.doc_form,
                    row.doc_number,
                    row.doc_year,
                    row.unit_type or "PASAL",
                    row.pasal_number or row.number_label
                )

                search_results.append(SearchResult(
                    id=row.unit_id,
                    content=row.content or "",
                    citation_string=citation,
                    score=float(row.similarity_score),
                    unit_type=row.unit_type,
                    unit_id=row.unit_id,
                    doc_form=row.doc_form,
                    doc_year=row.doc_year,
                    doc_number=row.doc_number,
                    hierarchy_path=row.hierarchy_path,
                    metadata={
                        'doc_title': row.doc_title,
                        'search_type': 'contextual_semantic_citation',
                        'pasal_number': row.pasal_number
                    }
                ))

            logger.debug(f"Vector search returned {len(search_results)} results")

            # PERFORMANCE MONITORING: Verify HNSW index usage
            if logger.isEnabledFor(logging.DEBUG):
                explain_query = f"EXPLAIN ANALYZE {final_query}"
                try:
                    explain_result = db.execute(text(explain_query), params)
                    query_plan = explain_result.fetchall()

                    # Check if HNSW index is being used
                    plan_text = str(query_plan)
                    index_used = "hnsw" in plan_text.lower()

                    logger.debug(
                        f"HNSW index usage: {'YES' if index_used else 'NO - SEQUENTIAL SCAN!'}"
                    )

                    if not index_used:
                        logger.warning(
                            "PERFORMANCE WARNING: HNSW index not used - falling back to sequential scan! "
                            "Check vector parameter format and index configuration."
                        )

                except Exception as explain_e:
                    logger.debug(f"Could not analyze query plan: {explain_e}")

            return search_results

        except Exception as e:
            logger.error(f"Vector search query failed: {e}")
            return []

    def _lookup_citation_units(
        self,
        db: Session,
        citation: CitationMatch,
        k: int,
        filters: Optional[SearchFilters]
    ) -> List[SearchResult]:
        """
        Direct SQL lookup for citation matches.

        Args:
            db: Database session
            citation: Parsed citation
            k: Number of results
            filters: Optional filters

        Returns:
            List of matching legal units
        """
        # Enhanced query with unit priority scoring for proper legal unit targeting
        base_query = """
        SELECT
            lu.unit_id,
            lu.bm25_body AS content,
            lu.citation_string,
            lu.unit_type,
            lu.number_label,
            lu.hierarchy_path,
            lu.parent_pasal_id,
            lu.parent_ayat_id,
            lu.parent_huruf_id,
            ld.doc_form,
            ld.doc_form_short,
            ld.doc_year,
            ld.doc_number,
            ld.doc_title,
            -- Priority scoring for unit specificity (lower = higher priority)
            CASE lu.unit_type
                WHEN 'HURUF' THEN 1
                WHEN 'ANGKA' THEN 2
                WHEN 'AYAT' THEN 3
                WHEN 'PASAL' THEN 4
                WHEN 'BAGIAN' THEN 5
                WHEN 'BAB' THEN 6
                ELSE 7
            END as unit_priority
        FROM legal_units lu
        JOIN legal_documents ld ON ld.id = lu.document_id
        WHERE ld.doc_status = :doc_status
        """

        params = {'doc_status': DocStatus.BERLAKU.value}
        conditions = []

        # Document-level filters
        if citation.doc_form:
            conditions.append("ld.doc_form = :doc_form")
            params['doc_form'] = citation.doc_form

        if citation.doc_number:
            conditions.append("ld.doc_number = :doc_number")
            params['doc_number'] = citation.doc_number

        if citation.doc_year:
            conditions.append("ld.doc_year = :doc_year")
            params['doc_year'] = citation.doc_year

        # Enhanced unit-level targeting with proper hierarchy
        if citation.huruf_letter and citation.pasal_number:
            # Most specific: Look for exact huruf in specific pasal
            unit_conditions = [
                "(lu.unit_type = 'HURUF' AND lu.number_label = :huruf_letter AND lu.parent_pasal_id LIKE :pasal_pattern)"
            ]
            params['huruf_letter'] = citation.huruf_letter
            params['pasal_pattern'] = f"%pasal-{citation.pasal_number}%"

            # Also include parent pasal for context
            unit_conditions.append(
                "(lu.unit_type = 'PASAL' AND lu.number_label = :pasal_number)"
            )
            params['pasal_number'] = citation.pasal_number

            # Include parent ayat if specified
            if citation.ayat_number:
                unit_conditions.append(
                    "(lu.unit_type = 'AYAT' AND lu.number_label = :ayat_number AND lu.parent_pasal_id LIKE :pasal_pattern)"
                )
                params['ayat_number'] = citation.ayat_number

        elif citation.angka_number and citation.huruf_letter and citation.pasal_number:
            # Very specific: angka in huruf in pasal
            unit_conditions = [
                "(lu.unit_type = 'ANGKA' AND lu.number_label = :angka_number AND lu.parent_huruf_id LIKE :huruf_pattern)",
                "(lu.unit_type = 'HURUF' AND lu.number_label = :huruf_letter AND lu.parent_pasal_id LIKE :pasal_pattern)",
                "(lu.unit_type = 'PASAL' AND lu.number_label = :pasal_number)"
            ]
            params['angka_number'] = citation.angka_number
            params['huruf_letter'] = citation.huruf_letter
            params['huruf_pattern'] = f"%huruf-{citation.huruf_letter}%"
            params['pasal_number'] = citation.pasal_number
            params['pasal_pattern'] = f"%pasal-{citation.pasal_number}%"

        elif citation.ayat_number and citation.pasal_number:
            # Specific ayat in specific pasal
            unit_conditions = [
                "(lu.unit_type = 'AYAT' AND lu.number_label = :ayat_number AND lu.parent_pasal_id LIKE :pasal_pattern)",
                "(lu.unit_type = 'PASAL' AND lu.number_label = :pasal_number)"
            ]
            params['ayat_number'] = citation.ayat_number
            params['pasal_number'] = citation.pasal_number
            params['pasal_pattern'] = f"%pasal-{citation.pasal_number}%"

        elif citation.pasal_number:
            # Pasal level - get pasal and its immediate children
            unit_conditions = [
                "(lu.unit_type = 'PASAL' AND lu.number_label = :pasal_number)",
                "(lu.unit_type = 'AYAT' AND lu.parent_pasal_id LIKE :pasal_pattern)",
                "(lu.unit_type = 'HURUF' AND lu.parent_pasal_id LIKE :pasal_pattern)"
            ]
            params['pasal_number'] = citation.pasal_number
            params['pasal_pattern'] = f"%pasal-{citation.pasal_number}%"

        else:
            # Document level - get key pasal units, NOT chapters
            unit_conditions = [
                "lu.unit_type IN ('PASAL', 'AYAT', 'HURUF')"
            ]

        if 'unit_conditions' in locals():
            conditions.append("(" + " OR ".join(unit_conditions) + ")")

        if conditions:
            base_query += " AND " + " AND ".join(conditions)

        # Enhanced ordering with unit priority for specificity
        final_query = base_query + """
        ORDER BY unit_priority ASC, ld.doc_year DESC,
                 CASE WHEN lu.number_label ~ '^[0-9]+$' THEN lu.number_label::INTEGER ELSE 999 END ASC,
                 lu.number_label ASC
        LIMIT :limit
        """
        params['limit'] = k

        try:
            result = db.execute(text(final_query), params)
            rows = result.fetchall()

            search_results = []
            for row in rows:
                # Build proper citation string based on unit type
                citation_str = self._build_unit_citation(
                    row.doc_form_short or row.doc_form,
                    row.doc_number,
                    row.doc_year,
                    row.unit_type,
                    row.number_label,
                    citation
                )

                search_results.append(SearchResult(
                    id=row.unit_id,
                    content=row.content or "",
                    citation_string=citation_str,
                    score=1.0,  # Exact match
                    unit_type=row.unit_type,
                    unit_id=row.unit_id,
                    doc_form=row.doc_form,
                    doc_year=row.doc_year,
                    doc_number=row.doc_number,
                    hierarchy_path=row.hierarchy_path,
                    metadata={
                        'doc_title': row.doc_title,
                        'search_type': 'explicit_citation',
                        'citation_match': citation.to_dict(),
                        'unit_number': row.number_label,
                        'unit_priority': row.unit_priority if hasattr(row, 'unit_priority') else 7,
                        'exact_match': True
                    }
                ))

            logger.debug(f"Citation lookup returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Citation lookup failed: {e}")
            return []

    async def _handle_contextual_search_async(
        self,
        query: str,
        k: int,
        filters: Optional[SearchFilters],
        use_reranking: bool = False
    ) -> List[SearchResult]:
        """Async version of contextual search with concurrent operations."""
        try:
            # Step 1: Normalize query and generate embedding concurrently
            normalize_task = asyncio.to_thread(self._normalize_query, query)
            embed_task = self._embed_query_async(query)

            # Run both operations concurrently
            normalized_query, query_embedding = await asyncio.gather(
                normalize_task, embed_task, return_exceptions=False
            )

            # Step 2: Vector search in thread pool
            def vector_search():
                with get_db_session() as db:
                    return self._vector_search(db, query_embedding, k, filters)

            vector_results = await asyncio.to_thread(vector_search)

            # Step 3: Apply post-processing concurrently if needed
            if normalized_query != query and use_reranking and vector_results:
                # Run relevance filtering and reranking concurrently
                filter_task = asyncio.to_thread(
                    self._filter_by_query_relevance, query, vector_results
                )
                rerank_task = asyncio.to_thread(
                    self._rerank_results, query, vector_results
                )

                filtered_results, reranked_results = await asyncio.gather(
                    filter_task, rerank_task, return_exceptions=True
                )

                # Use reranked results if both succeeded
                if not isinstance(reranked_results, Exception):
                    vector_results = reranked_results
                elif not isinstance(filtered_results, Exception):
                    vector_results = filtered_results

            elif normalized_query != query:
                vector_results = await asyncio.to_thread(
                    self._filter_by_query_relevance, query, vector_results
                )
            elif use_reranking and vector_results:
                vector_results = await asyncio.to_thread(
                    self._rerank_results, query, vector_results
                )

            logger.debug(f"Async contextual search returned {len(vector_results)} results")
            return vector_results

        except Exception as e:
            logger.error(f"Async contextual search failed: {e}")
            return []

    async def _handle_explicit_citation_async(
        self,
        query: str,
        k: int,
        filters: Optional[SearchFilters]
    ) -> List[SearchResult]:
        """Async version of explicit citation handling."""
        # Parse citation asynchronously
        citation = await asyncio.to_thread(get_best_citation_match, query)
        if not citation:
            logger.warning(f"No citation parsed from query: {query}")
            return []

        # Database lookup using thread pool
        def db_lookup():
            with get_db_session() as db:
                return self._lookup_citation_units(db, citation, k, filters)

        results = await asyncio.to_thread(db_lookup)
        logger.debug(f"Async citation search returned {len(results)} results")
        return results

    def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Rerank results using cross-encoder (placeholder for future implementation).

        Args:
            query: Original query
            results: Initial search results

        Returns:
            Reranked search results
        """
        # Placeholder for reranking implementation
        # In production, implement with jina-reranker-v2-base-multilingual or similar
        logger.debug(f"Reranking {len(results)} results (placeholder implementation)")
        return results

    def _log_search(
        self,
        query: str,
        results_count: int,
        duration_ms: int,
        search_type: str,
        session_id: Optional[str] = None
    ) -> None:
        """Log search for analytics and debugging."""
        logger.info(
            f"Search completed: query='{query[:50]}...', "
            f"type={search_type}, results={results_count}, "
            f"duration={duration_ms}ms, session={session_id}"
        )

    def vector_search_raw(
        self,
        query_text: str,
        k: int = None,
        filters: Optional[SearchFilters] = None
    ) -> List[SearchResult]:
        """
        Raw vector search without citation parsing.

        Useful for when you want to force vector search regardless of query content.

        Args:
            query_text: Query text to embed and search
            k: Number of results
            filters: Optional filters

        Returns:
            List of vector search results
        """
        k = k or self.default_k
        return self._handle_contextual_search(query_text, k, filters, use_reranking=False)

    def get_related_content(
        self,
        unit_id: str,
        k: int = 5,
        filters: Optional[SearchFilters] = None
    ) -> List[SearchResult]:
        """
        Get content related to a specific legal unit using vector similarity.

        Args:
            unit_id: Unit ID to find related content for
            k: Number of related results
            filters: Optional filters

        Returns:
            List of related search results
        """
        with get_db_session() as db:
            # Get the embedding for the source unit
            source_query = text("""
                SELECT embedding, lu.content
                FROM document_vectors dv
                JOIN legal_units lu ON lu.unit_id = dv.unit_id
                WHERE dv.unit_id = :unit_id
                LIMIT 1
            """)

            result = db.execute(source_query, {'unit_id': unit_id})
            row = result.fetchone()

            if not row:
                logger.warning(f"No embedding found for unit_id: {unit_id}")
                return []

            # Use the unit's embedding to find similar content
            return self._vector_search(db, list(row.embedding), k, filters)


    def _build_unit_citation(
        self,
        doc_form: str,
        doc_number: str,
        doc_year: int,
        unit_type: str,
        unit_number: str,
        citation: Optional[CitationMatch] = None
    ) -> str:
        """
        Build standardized citation string for any legal unit type.

        Args:
            doc_form: Document form (UU, PP, etc.)
            doc_number: Document number
            doc_year: Document year
            unit_type: Legal unit type (BAB, PASAL, AYAT, HURUF, ANGKA)
            unit_number: Unit number/label
            citation: Original citation context for enhanced formatting

        Returns:
            Properly formatted citation string
        """
        # Build base document reference
        if all([doc_form, doc_number, doc_year]):
            doc_ref = f"{doc_form} No. {doc_number} Tahun {doc_year}"
        else:
            doc_ref = ""

        # Build unit reference based on type
        unit_ref = ""
        if unit_type == "BAB":
            unit_ref = f"BAB {unit_number}"
        elif unit_type == "BAGIAN":
            unit_ref = f"Bagian {unit_number}"
        elif unit_type == "PASAL":
            unit_ref = f"Pasal {unit_number}"
        elif unit_type == "AYAT":
            unit_ref = f"ayat ({unit_number})"
        elif unit_type == "HURUF":
            unit_ref = f"huruf {unit_number}"
        elif unit_type == "ANGKA":
            unit_ref = f"angka {unit_number}"
        else:
            unit_ref = f"{unit_type} {unit_number}"

        # Enhanced formatting with citation context
        if citation and doc_ref:
            # Build hierarchical citation with proper context
            parts = [doc_ref]

            if citation.pasal_number and unit_type in ["AYAT", "HURUF", "ANGKA"]:
                parts.append(f"Pasal {citation.pasal_number}")

            if citation.ayat_number and unit_type in ["HURUF", "ANGKA"]:
                parts.append(f"ayat ({citation.ayat_number})")

            parts.append(unit_ref)
            return " ".join(parts)

        # Standard formatting
        if doc_ref and unit_ref:
            return f"{doc_ref} {unit_ref}"
        elif unit_ref:
            return unit_ref
        else:
            return f"{unit_type} {unit_number or 'Unknown'}"

    def _decompose_multi_part_query(self, query: str) -> List[str]:
        """
        Decompose query into parts for multi-part processing.

        Detects patterns like:
        - "apa itu X? dan apa isi Pasal Y?"
        - "definisi X dan UU Y Pasal Z"
        - "jelaskan X, kemudian cari Pasal Y"

        Args:
            query: Input query to decompose

        Returns:
            List of query parts (single item if no decomposition needed)
        """
        if not query or len(query.strip()) < 20:
            return [query]

        # Patterns that indicate multi-part queries
        split_patterns = [
            r'\s+dan\s+apa\s+isi\s+',  # "dan apa isi"
            r'\s+dan\s+(?:cari|temukan)\s+',  # "dan cari/temukan"
            r'\?\s*dan\s+',  # "? dan"
            r'\s+kemudian\s+',  # "kemudian"
            r'\s+juga\s+(?:cari|apa)\s+',  # "juga cari/apa"
            r'\s+serta\s+',  # "serta"
        ]

        for pattern in split_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Split at the pattern
                parts = re.split(pattern, query, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2 and all(len(p.strip()) > 5 for p in parts):
                    logger.debug(f"Multi-part query detected, split into: {len(parts)} parts")
                    return [p.strip() for p in parts]

        return [query]

    def _handle_multi_part_query(
        self,
        query_parts: List[str],
        k: int,
        filters: Optional[SearchFilters],
        use_reranking: bool = False
    ) -> List[SearchResult]:
        """
        Handle multi-part queries by processing each part separately and combining results.

        Args:
            query_parts: List of query parts to process
            k: Total number of results desired
            filters: Optional search filters
            use_reranking: Whether to apply reranking

        Returns:
            Combined search results from all parts
        """
        all_results = []
        k_per_part = max(2, k // len(query_parts))  # Distribute k across parts

        for i, part in enumerate(query_parts):
            logger.debug(f"Processing query part {i+1}/{len(query_parts)}: {part}")

            # Determine search type for this part
            if is_explicit_citation(part, self.min_citation_confidence):
                part_results = self._handle_explicit_citation(part, k_per_part, filters)
                logger.debug(f"Part {i+1} processed as explicit citation: {len(part_results)} results")
            else:
                part_results = self._handle_contextual_search(part, k_per_part, filters, use_reranking)
                logger.debug(f"Part {i+1} processed as contextual search: {len(part_results)} results")

            # Add part identifier to metadata for debugging
            for result in part_results:
                if hasattr(result, 'metadata') and result.metadata:
                    result.metadata['query_part'] = i + 1
                    result.metadata['query_part_text'] = part[:50] + "..." if len(part) > 50 else part

            all_results.extend(part_results)

        # Remove duplicates based on unit_id while preserving order
        seen_unit_ids = set()
        deduplicated_results = []

        for result in all_results:
            unit_id = getattr(result, 'unit_id', None) or getattr(result, 'id', None)
            if unit_id and unit_id not in seen_unit_ids:
                seen_unit_ids.add(unit_id)
                deduplicated_results.append(result)
            elif not unit_id:  # Include results without unit_id
                deduplicated_results.append(result)

        # Return top k results
        final_results = deduplicated_results[:k]
        logger.debug(f"Multi-part query completed: {len(final_results)} final results from {len(query_parts)} parts")

        return final_results

    async def _handle_multi_part_query_async(
        self,
        query_parts: List[str],
        k: int,
        filters: Optional[SearchFilters],
        use_reranking: bool = False
    ) -> List[SearchResult]:
        """
        Async version: Handle multi-part queries with concurrent processing.

        Args:
            query_parts: List of query parts to process
            k: Total number of results desired
            filters: Optional search filters
            use_reranking: Whether to apply reranking

        Returns:
            Combined search results from all parts
        """
        k_per_part = max(2, k // len(query_parts))  # Distribute k across parts

        # Create tasks for concurrent processing
        tasks = []
        for i, part in enumerate(query_parts):
            logger.debug(f"Creating async task for query part {i+1}/{len(query_parts)}: {part}")

            # Create task based on query type
            if await asyncio.to_thread(is_explicit_citation, part, self.min_citation_confidence):
                task = self._handle_explicit_citation_async(part, k_per_part, filters)
            else:
                task = self._handle_contextual_search_async(part, k_per_part, filters, use_reranking)

            tasks.append((i + 1, part, task))

        # Execute all tasks concurrently
        all_results = []
        completed_tasks = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)

        for (part_num, part_text, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Async query part {part_num} failed: {result}")
                continue

            logger.debug(f"Async part {part_num} completed: {len(result)} results")

            # Add part identifier to metadata
            for search_result in result:
                if hasattr(search_result, 'metadata') and search_result.metadata:
                    search_result.metadata['query_part'] = part_num
                    search_result.metadata['query_part_text'] = part_text[:50] + "..." if len(part_text) > 50 else part_text
                    search_result.metadata['async_processed'] = True

            all_results.extend(result)

        # Remove duplicates based on unit_id while preserving order
        seen_unit_ids = set()
        deduplicated_results = []

        for result in all_results:
            unit_id = getattr(result, 'unit_id', None) or getattr(result, 'id', None)
            if unit_id and unit_id not in seen_unit_ids:
                seen_unit_ids.add(unit_id)
                deduplicated_results.append(result)
            elif not unit_id:  # Include results without unit_id
                deduplicated_results.append(result)

        # Return top k results
        final_results = deduplicated_results[:k]
        logger.debug(f"Async multi-part query completed: {len(final_results)} final results from {len(query_parts)} parts")

        return final_results


def create_vector_search_service(
    embedder: Optional[JinaV4Embedder] = None,
    **kwargs
) -> VectorSearchService:
    """
    Factory function to create VectorSearchService.

    Args:
        embedder: Optional embedder instance
        **kwargs: Additional arguments for VectorSearchService

    Returns:
        Configured VectorSearchService instance
    """
    return VectorSearchService(embedder=embedder, **kwargs)
