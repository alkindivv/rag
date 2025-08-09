"""
Hybrid retriever combining FTS and vector search for Legal RAG system.

Handles both explicit queries (specific legal references) and thematic queries
(semantic search across legal content).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..embedding.embedder import JinaEmbedder
from ...utils.logging import get_logger, log_timing

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Individual search result with metadata."""

    id: str
    text: str
    citation_string: str
    score: float
    source_type: str  # 'fts' or 'vector'
    unit_type: str
    unit_id: str
    doc_form: Optional[str] = None
    doc_year: Optional[int] = None
    doc_number: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchFilters:
    """Search filters for narrowing results."""

    doc_forms: Optional[List[str]] = None
    doc_years: Optional[List[int]] = None
    doc_numbers: Optional[List[str]] = None
    unit_types: Optional[List[str]] = None
    bab_numbers: Optional[List[str]] = None
    pasal_numbers: Optional[List[str]] = None


class QueryRouter:
    """Routes queries to appropriate search strategy."""

    # Patterns for explicit legal references
    EXPLICIT_PATTERNS = [
        # Pasal references: "pasal 1", "pasal 1 ayat 2", etc.
        r'pasal\s+(\d+)(?:\s+ayat\s+(\d+))?(?:\s+huruf\s+([a-z]))?(?:\s+angka\s+(\d+))?',

        # Article references: "artikel 1", "article 1", etc.
        r'artikel?\s+(\d+)(?:\s+ayat\s+(\d+))?',

        # Direct document references: "UU 1/2023", "PP No. 1 Tahun 2023"
        r'(UU|PP|PERPU|PERPRES)\s+(?:No\.?\s*)?(\d+)(?:/|\s+[Tt]ahun\s+)(\d{4})',

        # Bab references: "bab 1", "bab I"
        r'bab\s+([IVX]+|\d+)',

        # Ayat, huruf, angka references
        r'ayat\s+(\d+)',
        r'huruf\s+([a-z])',
        r'angka\s+(\d+)',
    ]

    def __init__(self):
        """Initialize query router with compiled patterns."""
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.EXPLICIT_PATTERNS]

    def is_explicit_query(self, query: str) -> bool:
        """
        Determine if query is explicit (specific legal reference).

        Args:
            query: Search query

        Returns:
            True if query contains explicit legal references
        """
        return any(pattern.search(query) for pattern in self.patterns)

    def extract_explicit_references(self, query: str) -> Dict[str, Any]:
        """
        Extract explicit legal references from query.

        Args:
            query: Search query

        Returns:
            Dictionary of extracted references
        """
        references = {}

        for pattern in self.patterns:
            match = pattern.search(query)
            if match:
                groups = match.groups()

                # Parse based on pattern type
                if 'pasal' in pattern.pattern:
                    references['pasal'] = groups[0] if groups[0] else None
                    references['ayat'] = groups[1] if len(groups) > 1 and groups[1] else None
                    references['huruf'] = groups[2] if len(groups) > 2 and groups[2] else None
                    references['angka'] = groups[3] if len(groups) > 3 and groups[3] else None
                elif any(doc_type in pattern.pattern for doc_type in ['UU', 'PP', 'PERPU', 'PERPRES']):
                    references['doc_form'] = groups[0] if groups[0] else None
                    references['doc_number'] = groups[1] if len(groups) > 1 and groups[1] else None
                    references['doc_year'] = int(groups[2]) if len(groups) > 2 and groups[2] else None
                elif 'bab' in pattern.pattern:
                    references['bab'] = groups[0] if groups[0] else None
                elif 'ayat' in pattern.pattern:
                    references['ayat'] = groups[0] if groups[0] else None
                elif 'huruf' in pattern.pattern:
                    references['huruf'] = groups[0] if groups[0] else None
                elif 'angka' in pattern.pattern:
                    references['angka'] = groups[0] if groups[0] else None

        return references


class FTSSearcher:
    """Full-text search on legal units (leaf nodes)."""

    def __init__(self, db: Session):
        """Initialize FTS searcher with database session."""
        self.db = db

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 20
    ) -> List[SearchResult]:
        """
        Search using PostgreSQL full-text search.

        Args:
            query: Search query
            filters: Optional search filters
            limit: Maximum number of results

        Returns:
            List of search results
        """
        # Build base query string
        query_str = """
            SELECT
                lu.id,
                lu.unit_id,
                lu.unit_type,
                lu.bm25_body as text,
                lu.citation_string,
                ld.doc_form,
                ld.doc_year,
                ld.doc_number,
                ts_rank(lu.content_vector, plainto_tsquery('indonesian', :query)) as score
            FROM legal_units lu
            JOIN legal_documents ld ON lu.document_id = ld.id
            WHERE lu.unit_type IN ('AYAT', 'HURUF', 'ANGKA')
              AND lu.content_vector @@ plainto_tsquery('indonesian', :query)
        """

        filters_clause = ""
        query_params = {"query": query, "limit": limit}

        if filters:
            filter_conditions = []

            if filters.doc_forms:
                filter_conditions.append("ld.doc_form::text = ANY(:doc_forms)")
                query_params["doc_forms"] = filters.doc_forms

            if filters.doc_years:
                filter_conditions.append("ld.doc_year = ANY(:doc_years)")
                query_params["doc_years"] = filters.doc_years

            if filters.doc_numbers:
                filter_conditions.append("ld.doc_number = ANY(:doc_numbers)")
                query_params["doc_numbers"] = filters.doc_numbers

            if filter_conditions:
                filters_clause = "AND " + " AND ".join(filter_conditions)

        if filters_clause:
            query_str += f" {filters_clause}"

        query_str += """
            ORDER BY score DESC
            LIMIT :limit
        """

        fts_query = text(query_str)
        results = self.db.execute(fts_query, query_params).fetchall()

        # Convert to SearchResult objects
        search_results = []
        for row in results:
            search_results.append(SearchResult(
                id=str(row.id),
                text=row.text or "",
                citation_string=row.citation_string or "",
                score=float(row.score or 0.0),
                source_type="fts",
                unit_type=row.unit_type,
                unit_id=row.unit_id,
                doc_form=row.doc_form,
                doc_year=row.doc_year,
                doc_number=row.doc_number,
            ))

        logger.debug(f"FTS search returned {len(search_results)} results")
        return search_results


class VectorSearcher:
    """Vector search on document embeddings (pasal level)."""

    def __init__(self, db: Session, embedder: Optional["JinaEmbedder"] = None):
        """Initialize vector searcher with database session and embedder."""
        self.db = db
        if embedder is None:
            from ..embedding.embedder import JinaEmbedder as _JinaEmbedder
            self.embedder = _JinaEmbedder()
        else:
            self.embedder = embedder

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search using vector similarity.

        Args:
            query: Search query
            filters: Optional search filters
            limit: Maximum number of results

        Returns:
            List of search results
        """
        try:
            # Get query embedding
            query_embedding = self.embedder.embed_single(query)
            if getattr(self.embedder, "disabled", False):
                logger.info("Embedder disabled; skipping vector search")
                return []
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []

            # Build base query string
            query_str = """
                SELECT
                    dv.id,
                    dv.unit_id,
                    lu.unit_type,
                    lu.bm25_body as text,
                    lu.citation_string,
                    dv.doc_form,
                    dv.doc_year,
                    dv.doc_number,
                    dv.hierarchy_path,
                    1 - (dv.embedding <=> :query_vector) as score
                FROM document_vectors dv
                LEFT JOIN legal_units lu ON dv.unit_id = lu.unit_id AND lu.unit_type = 'PASAL'
                WHERE 1=1
            """

            filters_clause = ""
            query_params = {"query_vector": query_embedding, "limit": limit}

            if filters:
                filter_conditions = []

                if filters.doc_forms:
                    filter_conditions.append("dv.doc_form::text = ANY(:doc_forms)")
                    query_params["doc_forms"] = filters.doc_forms

                if filters.doc_years:
                    filter_conditions.append("dv.doc_year = ANY(:doc_years)")
                    query_params["doc_years"] = filters.doc_years

                if filters.doc_numbers:
                    filter_conditions.append("dv.doc_number = ANY(:doc_numbers)")
                    query_params["doc_numbers"] = filters.doc_numbers

                if filters.pasal_numbers:
                    filter_conditions.append("dv.pasal_number = ANY(:pasal_numbers)")
                    query_params["pasal_numbers"] = filters.pasal_numbers

                if filter_conditions:
                    filters_clause = "AND " + " AND ".join(filter_conditions)

            if filters_clause:
                query_str += f" {filters_clause}"

            query_str += """
                ORDER BY dv.embedding <=> :query_vector
                LIMIT :limit
            """

            vector_query = text(query_str)
            results = self.db.execute(vector_query, query_params).fetchall()

            # Convert to SearchResult objects
            search_results = []
            for row in results:
                search_results.append(SearchResult(
                    id=str(row.id),
                    text=row.text or "",
                    citation_string=row.citation_string or f"Pasal {row.unit_id}",
                    score=float(row.score or 0.0),
                    source_type="vector",
                    unit_type=row.unit_type or "pasal",
                    unit_id=row.unit_id,
                    doc_form=row.doc_form,
                    doc_year=row.doc_year,
                    doc_number=row.doc_number,
                    metadata={"hierarchy_path": row.hierarchy_path}
                ))

            logger.debug(f"Vector search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []


class ExplicitSearcher:
    """Handles explicit legal reference lookups."""

    def __init__(self, db: Session):
        """Initialize explicit searcher with database session."""
        self.db = db

    def search(
        self,
        references: Dict[str, Any],
        filters: Optional[SearchFilters] = None,
        limit: int = 20
    ) -> List[SearchResult]:
        """
        Search for explicit legal references.

        Args:
            references: Extracted legal references
            filters: Optional search filters
            limit: Maximum number of results

        Returns:
            List of search results
        """
        search_results = []

        # Build query conditions based on references
        conditions = []
        query_params = {"limit": limit}

        # Handle document-level references
        if references.get("doc_form"):
            conditions.append("ld.doc_form = :doc_form")
            query_params["doc_form"] = references["doc_form"]

        if references.get("doc_number"):
            conditions.append("ld.doc_number = :doc_number")
            query_params["doc_number"] = references["doc_number"]

        if references.get("doc_year"):
            conditions.append("ld.doc_year = :doc_year")
            query_params["doc_year"] = references["doc_year"]

        # Handle unit-level references
        unit_conditions = []

        if references.get("pasal"):
            unit_conditions.append("lu.number_label = :pasal AND lu.unit_type = 'PASAL'")
            query_params["pasal"] = references["pasal"]

        if references.get("ayat"):
            unit_conditions.append("lu.number_label = :ayat AND lu.unit_type = 'AYAT'")
            query_params["ayat"] = references["ayat"]

        if references.get("huruf"):
            unit_conditions.append("lu.number_label = :huruf AND lu.unit_type = 'HURUF'")
            query_params["huruf"] = references["huruf"]

        if references.get("angka"):
            unit_conditions.append("lu.number_label = :angka AND lu.unit_type = 'ANGKA'")
            query_params["angka"] = references["angka"]

        # Combine conditions
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        if unit_conditions:
            unit_clause = "(" + " OR ".join(unit_conditions) + ")"
            if where_clause:
                where_clause += f" AND {unit_clause}"
            else:
                where_clause = f"WHERE {unit_clause}"

        # Build and execute query
        explicit_query = text(f"""
            SELECT
                lu.id,
                lu.unit_id,
                lu.unit_type,
                lu.bm25_body as text,
                lu.citation_string,
                ld.doc_form,
                ld.doc_year,
                ld.doc_number,
                1.0 as score
            FROM legal_units lu
            JOIN legal_documents ld ON lu.document_id = ld.id
            {where_clause}
            ORDER BY ld.doc_year DESC, lu.ordinal_int ASC
            LIMIT :limit
        """)

        try:
            results = self.db.execute(explicit_query, query_params).fetchall()

            for row in results:
                search_results.append(SearchResult(
                    id=str(row.id),
                    text=row.text or "",
                    citation_string=row.citation_string or "",
                    score=1.0,  # Explicit matches get perfect score
                    source_type="explicit",
                    unit_type=row.unit_type,
                    unit_id=row.unit_id,
                    doc_form=row.doc_form,
                    doc_year=row.doc_year,
                    doc_number=row.doc_number,
                ))

            logger.debug(f"Explicit search returned {len(search_results)} results")

        except Exception as e:
            logger.error(f"Explicit search failed: {e}")

        return search_results


class HybridRetriever:
    """
    Main hybrid retriever orchestrating FTS, vector, and explicit search.

    Routes queries and combines results from different search methods.
    """

    def __init__(self, embedder: Optional["JinaEmbedder"] = None):
        """Initialize hybrid retriever with optional embedder."""
        if embedder is None:
            from ..embedding.embedder import JinaEmbedder as _JinaEmbedder
            self.embedder = _JinaEmbedder()
        else:
            self.embedder = embedder
        self.router = QueryRouter()

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 20,
        fts_weight: float = 0.4,
        vector_weight: float = 0.6,
        strategy: str = "auto",
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining multiple strategies.

        Args:
            query: Search query
            filters: Optional search filters
            limit: Maximum number of results
            fts_weight: Weight for FTS results in hybrid scoring
            vector_weight: Weight for vector results in hybrid scoring
            strategy: Search strategy ('auto', 'explicit', 'fts', 'vector', 'hybrid')

        Returns:
            List of ranked search results
        """
        import time
        start_time = time.time()

        try:
            from ...db.session import get_db_session
            with get_db_session() as db:
                if strategy == "explicit":
                    return self._explicit_search(db, query, filters, limit)
                if strategy == "fts":
                    return FTSSearcher(db).search(query, filters, limit)
                if strategy == "vector":
                    return VectorSearcher(db, self.embedder).search(query, filters, limit)
                if strategy == "hybrid":
                    return self._thematic_search(
                        db, query, filters, limit, fts_weight, vector_weight
                    )

                # Auto strategy: route based on query type
                if self.router.is_explicit_query(query):
                    return self._explicit_search(db, query, filters, limit)
                return self._thematic_search(
                    db, query, filters, limit, fts_weight, vector_weight
                )

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

        finally:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "Hybrid search completed",
                extra=log_timing("hybrid_search", duration_ms, query_length=len(query))
            )

    def _explicit_search(
        self,
        db: Session,
        query: str,
        filters: Optional[SearchFilters],
        limit: int
    ) -> List[SearchResult]:
        """Handle explicit legal reference queries."""
        logger.info(f"Processing explicit query: {query}")

        # Extract references from query
        references = self.router.extract_explicit_references(query)
        logger.debug(f"Extracted references: {references}")

        # Search using explicit searcher
        explicit_searcher = ExplicitSearcher(db)
        results = explicit_searcher.search(references, filters, limit)

        # If explicit search returns few results, supplement with FTS
        if len(results) < limit // 2:
            fts_searcher = FTSSearcher(db)
            fts_results = fts_searcher.search(query, filters, limit - len(results))
            results.extend(fts_results)

        return self._deduplicate_and_rank(results)[:limit]

    def _thematic_search(
        self,
        db: Session,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
        fts_weight: float,
        vector_weight: float
    ) -> List[SearchResult]:
        """Handle thematic/semantic queries."""
        logger.info(f"Processing thematic query: {query}")

        all_results = []

        # Vector search (pasal level)
        vector_searcher = VectorSearcher(db, self.embedder)
        vector_results = vector_searcher.search(query, filters, limit)

        # Apply vector weight to scores
        for result in vector_results:
            result.score *= vector_weight

        all_results.extend(vector_results)

        # FTS search (leaf level)
        fts_searcher = FTSSearcher(db)
        fts_results = fts_searcher.search(query, filters, limit)

        # Apply FTS weight to scores
        for result in fts_results:
            result.score *= fts_weight

        all_results.extend(fts_results)

        # Deduplicate and rank
        return self._deduplicate_and_rank(all_results)[:limit]

    def _deduplicate_and_rank(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Deduplicate and rank search results.

        Args:
            results: List of search results

        Returns:
            Deduplicated and ranked results
        """
        # Deduplicate by unit_id, keeping highest scoring result
        seen_units = {}
        for result in results:
            if result.unit_id not in seen_units or result.score > seen_units[result.unit_id].score:
                seen_units[result.unit_id] = result

        # Sort by score (descending)
        deduplicated_results = sorted(seen_units.values(), key=lambda x: x.score, reverse=True)

        logger.debug(f"Deduplicated {len(results)} results to {len(deduplicated_results)}")
        return deduplicated_results

    def get_related_units(
        self,
        pasal_id: str,
        include_children: bool = True
    ) -> List[SearchResult]:
        """
        Get related units for a given pasal.

        Args:
            pasal_id: Pasal unit ID
            include_children: Whether to include child units (ayat, huruf, angka)

        Returns:
            List of related units
        """
        try:
            from ...db.session import get_db_session
            with get_db_session() as db:
                if include_children:
                    # Get all child units of the pasal
                    query = text("""
                        SELECT
                            lu.id,
                            lu.unit_id,
                            lu.unit_type,
                            lu.bm25_body as text,
                            lu.citation_string,
                            ld.doc_form,
                            ld.doc_year,
                            ld.doc_number
                        FROM legal_units lu
                        JOIN legal_documents ld ON lu.document_id = ld.id
                        WHERE lu.parent_pasal_id = :pasal_id
                           OR lu.unit_id = :pasal_id
                        ORDER BY lu.ordinal_int, lu.ordinal_suffix
                    """)
                else:
                    # Get only the pasal itself
                    query = text("""
                        SELECT
                            lu.id,
                            lu.unit_id,
                            lu.unit_type,
                            lu.bm25_body as text,
                            lu.citation_string,
                            ld.doc_form,
                            ld.doc_year,
                            ld.doc_number
                        FROM legal_units lu
                        JOIN legal_documents ld ON lu.document_id = ld.id
                        WHERE lu.unit_id = :pasal_id
                    """)

                results = db.execute(query, {"pasal_id": pasal_id}).fetchall()

                search_results = []
                for row in results:
                    search_results.append(SearchResult(
                        id=str(row.id),
                        text=row.text or "",
                        citation_string=row.citation_string or "",
                        score=1.0,
                        source_type="related",
                        unit_type=row.unit_type,
                        unit_id=row.unit_id,
                        doc_form=row.doc_form,
                        doc_year=row.doc_year,
                        doc_number=row.doc_number,
                    ))

                logger.debug(f"Found {len(search_results)} related units for {pasal_id}")
                return search_results

        except Exception as e:
            logger.error(f"Failed to get related units for {pasal_id}: {e}")
            return []
