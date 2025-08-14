"""
Vector-only search service for Legal RAG system.

Implements dense semantic search with citation parsing for explicit queries.
Replaces hybrid FTS+vector approach with pure vector search optimized for
Indonesian legal documents.
"""

from __future__ import annotations

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
        min_citation_confidence: float = 0.60
    ):
        """
        Initialize vector search service.

        Args:
            embedder: Optional JinaV4Embedder instance
            default_k: Default number of results to retrieve
            min_citation_confidence: Minimum confidence for citation matching
        """
        self.embedder = embedder or JinaV4Embedder(default_dims=384)
        self.default_k = default_k
        self.min_citation_confidence = min_citation_confidence

        logger.info("Initialized VectorSearchService with dense semantic search only")

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

        try:
            # Step 1: Check if query contains explicit citations
            if is_explicit_citation(query, self.min_citation_confidence):
                logger.debug(f"Processing explicit citation query: {query}")
                results = self._handle_explicit_citation(query, k, filters)
                search_type = "explicit_citation"
            else:
                logger.debug(f"Processing contextual semantic query: {query}")
                results = self._handle_contextual_search(query, k, filters, use_reranking)
                search_type = "contextual_semantic"

            duration_ms = int((time.time() - start_time) * 1000)

            # Log search for analytics
            self._log_search(query, len(results), duration_ms, search_type, session_id)

            return {
                "results": results,
                "metadata": {
                    "query": query,
                    "search_type": search_type,
                    "total_results": len(results),
                    "duration_ms": duration_ms,
                    "filters_applied": filters is not None,
                    "reranking_used": use_reranking and search_type == "contextual_semantic"
                }
            }

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return {
                "results": [],
                "metadata": {
                    "query": query,
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
        # Step 1: Normalize query text
        normalized_query = self._normalize_query(query)

        # Step 2: Embed query
        query_embedding = self._embed_query(normalized_query)
        if not query_embedding:
            logger.error("Failed to embed query")
            return []

        # Step 3: Vector search with HNSW
        with get_db_session() as db:
            results = self._vector_search(db, query_embedding, k, filters)

        # Step 4: Optional reranking
        if use_reranking and results:
            results = self._rerank_results(query, results)

        return results

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query text for better embedding quality.

        Args:
            query: Raw query text

        Returns:
            Normalized query text
        """
        # Lowercasing
        normalized = query.lower().strip()

        # Remove punctuation (keep alphanumeric and spaces)
        normalized = re.sub(r'[^\w\s]', ' ', normalized)

        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)

        # Basic Indonesian stemming/lemmatization (simplified)
        # In production, use proper Indonesian NLP library like Sastrawi
        normalized = self._simple_indonesian_normalize(normalized)

        return normalized.strip()

    def _simple_indonesian_normalize(self, text: str) -> str:
        """
        Simple Indonesian text normalization.

        In production, replace with proper Indonesian lemmatization.
        """
        # Common Indonesian prefix/suffix removal (very basic)
        words = text.split()
        normalized_words = []

        for word in words:
            # Remove common prefixes
            for prefix in ['ber', 'ter', 'me', 'di', 'ke', 'se']:
                if word.startswith(prefix) and len(word) > len(prefix) + 2:
                    word = word[len(prefix):]
                    break

            # Remove common suffixes
            for suffix in ['kan', 'an', 'nya', 'lah']:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    word = word[:-len(suffix)]
                    break

            normalized_words.append(word)

        return ' '.join(normalized_words)

    def _embed_query(self, query: str) -> Optional[List[float]]:
        """
        Embed query using Jina v4 embedder.

        Args:
            query: Normalized query text

        Returns:
            Query embedding vector or None if failed
        """
        try:
            embeddings = self.embedder.embed_texts(
                [query],
                task="retrieval.query",
                dims=384
            )
            return embeddings[0] if embeddings else None

        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return None

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
            'query_vector': query_embedding,
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
                # Build proper citation string for pasal
                citation = self._build_pasal_citation(
                    row.doc_form_short or row.doc_form,
                    row.doc_number,
                    row.doc_year,
                    row.pasal_number
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
        # Build query based on citation specificity - focus on pasal citations
        base_query = """
        SELECT
            lu.unit_id,
            lu.bm25_body AS content,
            lu.citation_string,
            lu.unit_type,
            lu.number_label AS pasal_number,
            lu.hierarchy_path,
            ld.doc_form,
            ld.doc_form_short,
            ld.doc_year,
            ld.doc_number,
            ld.doc_title
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

        # Unit-level filters
        if citation.pasal_number:
            if citation.ayat_number or citation.huruf_letter or citation.angka_number:
                # Looking for specific sub-units
                if citation.ayat_number:
                    conditions.append("(lu.unit_type = 'PASAL' AND lu.number_label = :pasal_number)")
                    conditions.append("OR (lu.unit_type = 'AYAT' AND lu.number_label = :ayat_number AND lu.parent_pasal_id LIKE :pasal_pattern)")
                    params['ayat_number'] = citation.ayat_number
                    params['pasal_pattern'] = f"%pasal-{citation.pasal_number}%"

                if citation.huruf_letter:
                    conditions.append("OR (lu.unit_type = 'HURUF' AND lu.number_label = :huruf_letter)")
                    params['huruf_letter'] = citation.huruf_letter

                if citation.angka_number:
                    conditions.append("OR (lu.unit_type = 'ANGKA' AND lu.number_label = :angka_number)")
                    params['angka_number'] = citation.angka_number
            else:
                # Looking for pasal only
                conditions.append("lu.unit_type = 'PASAL' AND lu.number_label = :pasal_number")

            params['pasal_number'] = citation.pasal_number

        if conditions:
            base_query += " AND (" + " AND ".join(conditions) + ")"

        # Add ordering and limit
        final_query = base_query + """
        ORDER BY ld.doc_year DESC, lu.unit_type, lu.number_label
        LIMIT :limit
        """
        params['limit'] = k

        try:
            result = db.execute(text(final_query), params)
            rows = result.fetchall()

            search_results = []
            for row in rows:
                # Build proper citation string for exact match
                citation_str = self._build_pasal_citation(
                    row.doc_form_short or row.doc_form,
                    row.doc_number,
                    row.doc_year,
                    row.pasal_number or citation.pasal_number
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
                        'pasal_number': row.pasal_number or citation.pasal_number
                    }
                ))

            logger.debug(f"Citation lookup returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Citation lookup failed: {e}")
            return []

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


    def _build_pasal_citation(
        self,
        doc_form: str,
        doc_number: str,
        doc_year: int,
        pasal_number: str
    ) -> str:
        """
        Build standardized pasal citation string.

        Args:
            doc_form: Document form (UU, PP, etc.)
            doc_number: Document number
            doc_year: Document year
            pasal_number: Pasal number

        Returns:
            Formatted citation string
        """
        if not all([doc_form, doc_number, doc_year, pasal_number]):
            return f"Pasal {pasal_number or 'Unknown'}"

        return f"{doc_form} No. {doc_number} Tahun {doc_year} Pasal {pasal_number}"


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
