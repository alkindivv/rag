"""
BM25 Search Service for PostgreSQL Full-Text Search.

Provides keyword-based search using PostgreSQL's FTS with BM25-like ranking
for Legal RAG system. Complements vector search for hybrid retrieval.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Dict, Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from ...config.settings import settings
from ...db.models import DocStatus, UnitType
from ...utils.logging import get_logger, log_timing, log_error
from .vector_search import SearchResult, SearchFilters

logger = get_logger(__name__)


class BM25SearchService:
    """
    BM25-style search service using PostgreSQL Full-Text Search.

    Provides keyword-based retrieval to complement vector search in hybrid mode.
    Uses PostgreSQL's ts_rank for BM25-like relevance scoring.
    """

    def __init__(self):
        """Initialize BM25 search service."""
        self.min_query_length = 2
        self.max_results = 100
        logger.info("BM25SearchService initialized with PostgreSQL FTS")

    async def search_async(
        self,
        query: str,
        k: int = 10,
        filters: Optional[SearchFilters] = None
    ) -> Dict[str, Any]:
        """
        Async wrapper for BM25 search.

        Args:
            query: Search query text
            k: Number of results to return
            filters: Optional search filters

        Returns:
            Dict with results and metadata (consistent with vector search)
        """
        # For now, just call sync version (can be enhanced later with asyncpg)
        return self.search(query, k, filters)

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[SearchFilters] = None
    ) -> Dict[str, Any]:
        """
        Perform BM25 search using PostgreSQL FTS.

        Args:
            query: Search query text
            k: Number of results to return
            filters: Optional search filters

        Returns:
            Dict with results and metadata (consistent with vector search)
        """
        start_time = time.time()

        # Validate query
        if not query or len(query.strip()) < self.min_query_length:
            logger.warning(f"Query too short for BM25 search: '{query}'")
            return []

        # Sanitize and prepare query
        clean_query = self._prepare_fts_query(query.strip())
        if not clean_query:
            logger.warning(f"No valid terms for FTS query: '{query}'")
            return {
                "results": [],
                "metadata": {
                    "query": query,
                    "search_type": "bm25_fts",
                    "total_results": 0,
                    "duration_ms": 0,
                    "error": "No valid terms for FTS query"
                }
            }

        try:
            # Import here to avoid circular imports
            from ...db.session import get_db_session

            with get_db_session() as db:
                results = self._fts_search(db, clean_query, k, filters)

            duration_ms = (time.time() - start_time) * 1000

            self._log_search(query, results, duration_ms)

            # Return dict format consistent with vector search
            return {
                "results": results,
                "metadata": {
                    "query": query,
                    "search_type": "bm25_fts",
                    "total_results": len(results),
                    "duration_ms": round(duration_ms, 2),
                    "fts_query": clean_query
                }
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"BM25 search failed for query '{query}': {e}",
                extra=log_error(e, context={
                    "query": query,
                    "k": k,
                    "duration_ms": duration_ms
                })
            )
            return {
                "results": [],
                "metadata": {
                    "query": query,
                    "search_type": "bm25_fts",
                    "total_results": 0,
                    "duration_ms": duration_ms,
                    "error": str(e)
                }
            }

    def _prepare_fts_query(self, query: str) -> str:
        """
        Prepare query text for PostgreSQL FTS.

        Args:
            query: Raw query text

        Returns:
            Formatted FTS query string
        """
        # Remove special characters that could break FTS
        import re

        # Keep only alphanumeric, spaces, and common punctuation
        cleaned = re.sub(r'[^\w\s\-\.]', ' ', query)

        # Split into terms and filter out short ones
        terms = [term.strip() for term in cleaned.split() if len(term.strip()) >= 2]

        if not terms:
            return ""

        # Join with | for OR search (better recall for multi-word queries)
        # For legal documents, we want to find documents containing any of the terms
        fts_query = " | ".join(terms)

        logger.debug(f"FTS query prepared: '{query}' -> '{fts_query}'")
        return fts_query

    def _fts_search(
        self,
        db: Session,
        fts_query: str,
        k: int,
        filters: Optional[SearchFilters]
    ) -> List[SearchResult]:
        """
        Execute FTS search against PostgreSQL.

        Args:
            db: Database session
            fts_query: Prepared FTS query
            k: Number of results
            filters: Optional filters

        Returns:
            List of SearchResult objects
        """
        # Build FTS query with ts_rank for BM25-like scoring
        base_query = """
        SELECT
            lu.unit_id,
            CASE
                WHEN lu.unit_type = 'PASAL' THEN lu.content
                ELSE lu.bm25_body
            END as content,
            lu.citation_string,
            lu.unit_type,
            lu.number_label,
            lu.hierarchy_path,
            ld.doc_form,
            ld.doc_year,
            ld.doc_number,
            ld.doc_title,
            ld.doc_form_short,
            -- BM25-like scoring using ts_rank_cd with normalization
            CASE
                WHEN lu.unit_type = 'PASAL' THEN
                    ts_rank_cd(
                        to_tsvector('indonesian', lu.content),
                        to_tsquery('indonesian', :fts_query),
                        32
                    )
                ELSE
                    ts_rank_cd(
                        lu.bm25_tsvector,
                        to_tsquery('indonesian', :fts_query),
                        32
                    )
            END as bm25_score
        FROM legal_units lu
        JOIN legal_documents ld ON ld.id = lu.document_id
        WHERE (
            (lu.unit_type = 'PASAL' AND to_tsvector('indonesian', lu.content) @@ to_tsquery('indonesian', :fts_query) AND lu.content IS NOT NULL)
            OR
            (lu.unit_type IN ('AYAT', 'HURUF', 'ANGKA') AND lu.bm25_tsvector @@ to_tsquery('indonesian', :fts_query) AND lu.bm25_body IS NOT NULL)
        )
        AND ld.doc_status = :doc_status
        """

        # Add filters
        filter_conditions = []
        params = {
            'fts_query': fts_query,
            'doc_status': DocStatus.BERLAKU.value
        }

        if filters:
            if filters.doc_forms:
                filter_conditions.append("ld.doc_form = ANY(:doc_forms)")
                params['doc_forms'] = filters.doc_forms

            if filters.doc_years:
                filter_conditions.append("ld.doc_year = ANY(:doc_years)")
                params['doc_years'] = filters.doc_years

            if filters.doc_numbers:
                filter_conditions.append("ld.doc_number = ANY(:doc_numbers)")
                params['doc_numbers'] = filters.doc_numbers

            if filters.pasal_numbers:
                # For BM25, we might need to handle this differently
                filter_conditions.append("lu.number_label = ANY(:pasal_numbers)")
                params['pasal_numbers'] = filters.pasal_numbers

        if filter_conditions:
            base_query += " AND " + " AND ".join(filter_conditions)

        # Add ordering and limit
        final_query = base_query + """
        ORDER BY bm25_score DESC, lu.unit_id
        LIMIT :limit
        """
        params['limit'] = min(k, self.max_results)

        try:
            result = db.execute(text(final_query), params)
            rows = result.fetchall()

            search_results = []
            for row in rows:
                # Build citation if not available
                citation = row.citation_string or self._build_unit_citation(
                    row.doc_form_short or row.doc_form,
                    row.doc_number,
                    row.doc_year,
                    row.unit_type,
                    row.number_label
                )

                search_results.append(SearchResult(
                    id=row.unit_id,
                    content=row.content or "",
                    citation_string=citation,
                    score=float(row.bm25_score) if row.bm25_score else 0.0,
                    unit_type=row.unit_type,
                    unit_id=row.unit_id,
                    doc_form=row.doc_form,
                    doc_year=row.doc_year,
                    doc_number=row.doc_number,
                    hierarchy_path=row.hierarchy_path,
                    metadata={
                        'doc_title': row.doc_title,
                        'search_type': 'bm25_fts',
                        'fts_query': fts_query,
                        'pasal_number': row.number_label
                    }
                ))

            logger.debug(f"BM25 search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"BM25 FTS query failed: {e}")
            return []

    def _build_unit_citation(
        self,
        doc_form: str,
        doc_number: str,
        doc_year: int,
        unit_type: str,
        number_label: str
    ) -> str:
        """
        Build citation string for a legal unit.

        Args:
            doc_form: Document form (UU, PP, etc.)
            doc_number: Document number
            doc_year: Document year
            unit_type: Unit type (PASAL, AYAT, etc.)
            number_label: Unit number

        Returns:
            Formatted citation string
        """
        try:
            # Map unit types to Indonesian terms
            unit_map = {
                "PASAL": "Pasal",
                "AYAT": "ayat",
                "HURUF": "huruf",
                "ANGKA": "angka"
            }

            unit_name = unit_map.get(unit_type, unit_type.lower())

            # Build citation
            citation = f"{doc_form} Nomor {doc_number} Tahun {doc_year}, {unit_name} {number_label}"

            return citation

        except Exception as e:
            logger.warning(f"Failed to build citation: {e}")
            return f"{doc_form} {doc_number}/{doc_year}"

    def _log_search(self, query: str, results: List[SearchResult], duration_ms: float):
        """Log search performance and results."""
        logger.info(
            f"BM25 search completed",
            extra=log_timing(
                "bm25_search",
                duration_ms,
                query_length=len(query),
                results_count=len(results),
                avg_score=sum(r.score for r in results) / len(results) if results else 0
            )
        )

    def get_search_stats(self, db: Session) -> dict:
        """
        Get statistics about BM25 search corpus.

        Args:
            db: Database session

        Returns:
            Dictionary with corpus statistics
        """
        try:
            stats_query = """
            SELECT
                COUNT(*) as total_units,
                COUNT(CASE
                    WHEN unit_type = 'PASAL' AND content IS NOT NULL THEN 1
                    WHEN unit_type IN ('AYAT', 'HURUF', 'ANGKA') AND bm25_tsvector IS NOT NULL THEN 1
                END) as indexed_units,
                AVG(CASE
                    WHEN unit_type = 'PASAL' THEN length(content)
                    ELSE length(bm25_body)
                END) as avg_content_length,
                COUNT(DISTINCT unit_type) as unit_types
            FROM legal_units lu
            JOIN legal_documents ld ON ld.id = lu.document_id
            WHERE ld.doc_status = :doc_status
            """

            result = db.execute(text(stats_query), {'doc_status': DocStatus.BERLAKU.value})
            row = result.fetchone()

            return {
                'total_units': row.total_units,
                'indexed_units': row.indexed_units,
                'indexing_coverage': row.indexed_units / row.total_units if row.total_units > 0 else 0,
                'avg_content_length': float(row.avg_content_length) if row.avg_content_length else 0,
                'unit_types': row.unit_types
            }

        except Exception as e:
            logger.error(f"Failed to get BM25 stats: {e}")
            return {}


def create_bm25_search_service() -> BM25SearchService:
    """Factory function to create BM25 search service."""
    return BM25SearchService()


# Example usage
if __name__ == "__main__":
    # For testing
    service = create_bm25_search_service()
    response = service.search("ekonomi kreatif pelaku", k=5)

    results = response["results"]
    metadata = response["metadata"]

    print(f"Found {len(results)} results:")
    print(f"Search type: {metadata['search_type']}")
    print(f"Duration: {metadata['duration_ms']:.2f}ms")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.citation_string}")
        print(f"   Score: {result.score:.4f}")
        print(f"   Content: {result.content[:100]}...")
        print()
