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

from ...config.settings import settings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..embedding.embedder import JinaV4Embedder
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

    # Enhanced patterns for explicit legal references with smart routing
    EXPLICIT_PATTERNS = [
        # Comprehensive pasal/ayat combinations
        r'(?:pasal|pasal\s+)?(\d+[A-Z]?)\s*(?:ayat\s*(\d+))?(?:\s*huruf\s*([a-z]))?(?:\s*angka\s*(\d+))?',
        
        # Specific pasal references with document context
        r'(?:uu|undang-undang)\s+(?:no\.?\s*)?(\d+)(?:/|\s+tahun\s+)(\d{4})\s*(?:pasal\s+(\d+))?(?:\s+ayat\s+(\d+))?',
        
        # PP references with pasal
        r'(?:pp|peraturan\s+pemerintah)\s+(?:no\.?\s*)?(\d+)(?:/|\s+tahun\s+)(\d{4})\s*(?:pasal\s+(\d+))?(?:\s+ayat\s+(\d+))?',
        
        # Explicit pasal/ayat/huruf combinations
        r'pasal\s+(\d+(?:[A-Z])?)\s*(?:ayat\s*(\d+))?(?:\s*huruf\s*([a-z]))?',
        
        # Direct citation format: "UU 4/2009 Pasal 121 Ayat 1"
        r'(UU|PP|PERPU)\s+(\d+)/(\d{4})\s+pasal\s+(\d+(?:[A-Z])?)(?:\s+ayat\s+(\d+))?',
        
        # Flexible citation format
        r'(?:pasal|pasal\s+)?(\d+[A-Z]?)\s*(?:ayat\s*\(?\s*(\d+)\s*\)?)?(?:\s*huruf\s*([a-z]))?',
    ]

    def __init__(self):
        """Initialize query router with compiled patterns."""
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.EXPLICIT_PATTERNS]
        logger.debug(f"Initialized QueryRouter with {len(self.patterns)} explicit patterns")

    def is_explicit_query(self, query: str) -> bool:
        """
        Determine if query is explicit (specific legal reference).

        Args:
            query: Search query

        Returns:
            True if query contains explicit legal references
        """
        is_explicit = any(pattern.search(query) for pattern in self.patterns)
        if is_explicit:
            logger.debug(f"Query identified as explicit: '{query}'")
        return is_explicit

    def extract_explicit_references(self, query: str) -> Dict[str, Any]:
        """
        Extract explicit legal references from query with comprehensive parsing.

        Args:
            query: Search query

        Returns:
            Dictionary with extracted references
        """
        references = {
            'pasal': None,
            'ayat': None,
            'huruf': None,
            'angka': None,
            'doc_form': None,
            'doc_form_short': None,
            'doc_number': None,
            'doc_year': None
        }

        clean_query = query.lower().strip()

        # Comprehensive pattern matching for legal citations
        
        # Pattern 1: UU 4/2009 Pasal 121 Ayat 1
        pattern1 = r'(?:uu|undang-undang)\s+(?:no\.?\s*)?(\d+)(?:/|\s+tahun\s+)(\d{4})\s+pasal\s+(\d+(?:[A-Z])?)(?:\s+ayat\s+(\d+))?'
        match1 = re.search(pattern1, clean_query, re.IGNORECASE)
        if match1:
            references['doc_form'] = 'UU'
            references['doc_number'] = match1.group(1)
            references['doc_year'] = int(match1.group(2))
            references['pasal'] = match1.group(3)
            if match1.group(4):
                references['ayat'] = match1.group(4)
            return references

        # Pattern 2: Pasal 121 Ayat 1 UU 4/2009
        pattern2 = r'pasal\s+(\d+(?:[A-Z])?)\s*(?:ayat\s*(\d+))?\s*(?:uu|undang-undang)\s+(\d+)/(\d{4})'
        match2 = re.search(pattern2, clean_query, re.IGNORECASE)
        if match2:
            references['pasal'] = match2.group(1)
            if match2.group(2):
                references['ayat'] = match2.group(2)
            references['doc_form'] = 'UU'
            references['doc_number'] = match2.group(3)
            references['doc_year'] = int(match2.group(4))
            return references

        # Pattern 3: Flexible pasal/ayat combinations
        pattern3 = r'pasal\s+(\d+(?:[A-Z])?)(?:\s+ayat\s*(\d+))?(?:\s+huruf\s*([a-z]))?(?:\s+angka\s*(\d+))?'
        match3 = re.search(pattern3, clean_query, re.IGNORECASE)
        if match3:
            references['pasal'] = match3.group(1)
            if match3.group(2):
                references['ayat'] = match3.group(2)
            if match3.group(3):
                references['huruf'] = match3.group(3)
            if match3.group(4):
                references['angka'] = match3.group(4)

        # Pattern 4: Document references without pasal
        pattern4 = r'(uu|pp|perpu|perpres)\s+(?:no\.?\s*)?(\d+)(?:/|\s+tahun\s+)(\d{4})'
        match4 = re.search(pattern4, clean_query, re.IGNORECASE)
        if match4:
            doc_type = match4.group(1).upper()
            references['doc_form'] = doc_type
            references['doc_number'] = match4.group(2)
            references['doc_year'] = int(match4.group(3))

        # Pattern 5: UUD 1945 detection (Constitution)
        # Handle phrases like "uud 1945" and "u u d 1945".
        if (re.search(r'\buud\b', clean_query) or re.search(r'\bu\s*u\s*d\b', clean_query)) and re.search(r'1945', clean_query):
            references['doc_form_short'] = 'UUD'
            references['doc_year'] = references.get('doc_year') or 1945

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
                lu.bm25_body AS text,
                lu.citation_string,
                ld.doc_form,
                ld.doc_year,
                ld.doc_number,
                ts_rank(lu.content_vector, plainto_tsquery('indonesian', :query)) as score
            FROM legal_units lu
            JOIN legal_documents ld ON lu.document_id = ld.id
            WHERE LOWER(lu.unit_type::text) IN ('ayat', 'huruf', 'angka')
              AND lu.content_vector @@ plainto_tsquery('indonesian', :query)
        """

        filters_clause = ""
        query_params = {"query": query, "limit": limit}

        if filters:
            filter_conditions = []

            if filters.doc_forms:
                filter_conditions.append("ld.doc_form::text = ANY(:doc_forms::text[])")
                query_params["doc_forms"] = filters.doc_forms

            if filters.doc_years:
                filter_conditions.append("ld.doc_year = ANY(:doc_years::int[])")
                query_params["doc_years"] = filters.doc_years

            if filters.doc_numbers:
                filter_conditions.append("ld.doc_number::text = ANY(:doc_numbers::text[])")
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

    def __init__(self, db: Session, embedder: Optional["JinaV4Embedder"] = None):
        """Initialize vector searcher with database session and embedder."""
        self.db = db
        if embedder is None:
            from ..embedding.embedder import JinaV4Embedder as _JinaV4Embedder
            self.embedder = _JinaV4Embedder()
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
            # Get query embedding using retrieval.query task
            query_embedding = self.embedder.embed_query(query, dims=settings.embedding_dim)
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []

            # Build base query string (safe :qvec binding; no CAST on bind param)
            query_str = """
                SELECT
                    dv.id,
                    dv.unit_id,
                    lu.unit_type,
                    lu.bm25_body AS text,
                    lu.citation_string,
                    dv.doc_form,
                    dv.doc_year,
                    dv.doc_number,
                    dv.hierarchy_path,
                    1 - (dv.embedding <=> :qvec) AS score
                FROM document_vectors dv
                LEFT JOIN legal_units lu
                  ON lu.unit_id = dv.unit_id AND lu.unit_type = 'pasal'
                WHERE 1=1
            """

            filters_clause = ""
            query_params = {"qvec": query_embedding, "limit": limit}

            if filters:
                filter_conditions = []

                if filters.doc_forms:
                    filter_conditions.append("dv.doc_form::text = ANY(:doc_forms::text[])")
                    query_params["doc_forms"] = filters.doc_forms

                if filters.doc_years:
                    filter_conditions.append("dv.doc_year = ANY(:doc_years::int[])")
                    query_params["doc_years"] = filters.doc_years

                if filters.doc_numbers:
                    filter_conditions.append("dv.doc_number::text = ANY(:doc_numbers::text[])")
                    query_params["doc_numbers"] = filters.doc_numbers

                if filters.pasal_numbers:
                    filter_conditions.append("dv.pasal_number = ANY(:pasal_numbers::text[])")
                    query_params["pasal_numbers"] = filters.pasal_numbers

                if filter_conditions:
                    filters_clause = "AND " + " AND ".join(filter_conditions)

            if filters_clause:
                query_str += f" {filters_clause}"

            query_str += """
                ORDER BY dv.embedding <=> :qvec
                LIMIT :limit
            """

            vector_query = text(query_str)
            results = self.db.execute(vector_query, query_params).fetchall()

            # Convert to SearchResult objects
            search_results = []
            for row in results:
                # Citation fallback if missing
                fallback_citation = None
                try:
                    if row.doc_form and row.doc_number and row.doc_year:
                        # Example: UU 4/2009, Pasal 121
                        pasal_label = row.unit_id.split('/pasal-')[-1].split('/')[0] if row.unit_id else None
                        if pasal_label:
                            fallback_citation = f"{row.doc_form} {row.doc_number}/{row.doc_year}, Pasal {pasal_label}"
                except Exception:
                    fallback_citation = None

                search_results.append(SearchResult(
                    id=str(row.id),
                    text=(row.text or ""),
                    citation_string=(row.citation_string or fallback_citation or ""),
                    score=float(row.score or 0.0),
                    source_type="vector",
                    unit_type=(row.unit_type or "pasal"),
                    unit_id=row.unit_id,
                    doc_form=row.doc_form,
                    doc_year=row.doc_year,
                    doc_number=row.doc_number,
                    metadata={"hierarchy_path": getattr(row, 'hierarchy_path', None)}
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
        limit: int = 20,
    ) -> List[SearchResult]:
        """Deterministic explicit retrieval via parent-anchored chain.

        Chain: PASAL -> AYAT -> HURUF -> ANGKA using number_label and parent_* IDs.
        Returns the most specific unit found.
        """
        try:
            params: Dict[str, Any] = {}

            # Optional document constraint
            doc_clause = ""
            if references.get("doc_form") or references.get("doc_number") or references.get("doc_year"):
                dq = ["SELECT id FROM legal_documents WHERE 1=1"]
                has_num = bool(references.get("doc_number"))
                has_year = bool(references.get("doc_year"))
                # Prefer precise selection by number/year; only use form when num/year incomplete
                if references.get("doc_form") and not (has_num and has_year):
                    # Cast to text to avoid domain/enum lower() errors and compare case-insensitively
                    dq.append("AND (doc_form::text) ILIKE :doc_form")
                    params["doc_form"] = references["doc_form"]
                if has_num:
                    dq.append("AND doc_number = :doc_number")
                    params["doc_number"] = references["doc_number"]
                if has_year:
                    dq.append("AND doc_year = :doc_year")
                    params["doc_year"] = references["doc_year"]
                dq.append("ORDER BY doc_year DESC LIMIT 1")
                row = self.db.execute(text("\n".join(dq)), params).fetchone()
                if not row:
                    return []
                params["doc_id"] = row.id
                doc_clause = " AND document_id = :doc_id"

            # PASAL
            pasal_id: Optional[str] = None
            if references.get("pasal"):
                q = text(
                    f"""
                    SELECT unit_id FROM legal_units
                    WHERE LOWER(unit_type::text) = 'pasal' AND number_label = :pasal{doc_clause}
                    ORDER BY ordinal_int ASC
                    LIMIT 1
                    """
                )
                params["pasal"] = references["pasal"]
                r = self.db.execute(q, params).fetchone()
                if not r:
                    return []
                pasal_id = r.unit_id

            # AYAT
            ayat_id: Optional[str] = None
            if references.get("ayat"):
                if not pasal_id:
                    return []
                q = text(
                    """
                    SELECT unit_id FROM legal_units
                    WHERE LOWER(unit_type::text) = 'ayat' AND number_label = :ayat
                      AND parent_pasal_id = :pasal_id
                    ORDER BY ordinal_int ASC
                    LIMIT 1
                    """
                )
                r = self.db.execute(q, {"ayat": references["ayat"], "pasal_id": pasal_id}).fetchone()
                if not r:
                    return []
                ayat_id = r.unit_id

            # HURUF
            huruf_id: Optional[str] = None
            if references.get("huruf"):
                if not ayat_id:
                    return []
                q = text(
                    """
                    SELECT unit_id FROM legal_units
                    WHERE LOWER(unit_type::text) = 'huruf' AND number_label = :huruf
                      AND parent_ayat_id = :ayat_id
                    ORDER BY ordinal_int ASC
                    LIMIT 1
                    """
                )
                r = self.db.execute(q, {"huruf": references["huruf"], "ayat_id": ayat_id}).fetchone()
                if not r:
                    return []
                huruf_id = r.unit_id

            # ANGKA
            angka_id: Optional[str] = None
            if references.get("angka"):
                if not huruf_id:
                    return []
                q = text(
                    """
                    SELECT unit_id FROM legal_units
                    WHERE LOWER(unit_type::text) = 'angka' AND number_label = :angka
                      AND parent_huruf_id = :huruf_id
                    ORDER BY ordinal_int ASC
                    LIMIT 1
                    """
                )
                r = self.db.execute(q, {"angka": references["angka"], "huruf_id": huruf_id}).fetchone()
                if not r:
                    return []
                angka_id = r.unit_id

            final_unit_id = angka_id or huruf_id or ayat_id or pasal_id
            if not final_unit_id:
                return []

            # Fetch final unit
            q = text(
                """
                SELECT
                    lu.id,
                    lu.unit_id,
                    lu.unit_type,
                    COALESCE(lu.local_content, lu.content, lu.display_text, lu.bm25_body) AS text,
                    lu.citation_string,
                    ld.doc_form,
                    ld.doc_year,
                    ld.doc_number
                FROM legal_units lu
                JOIN legal_documents ld ON ld.id = lu.document_id
                WHERE lu.unit_id = :uid
                """
            )
            row = self.db.execute(q, {"uid": final_unit_id}).fetchone()
            if not row:
                return []

            return [SearchResult(
                id=str(row.id),
                text=row.text or "",
                citation_string=row.citation_string or "",
                score=1.0,
                source_type="explicit",
                unit_type=row.unit_type,
                unit_id=row.unit_id,
                doc_form=row.doc_form,
                doc_year=row.doc_year,
                doc_number=row.doc_number,
            )]
        except Exception as e:
            logger.error(f"Explicit search failed: {e}")
            return []
class HybridRetriever:
    """
    Main hybrid retriever orchestrating FTS, vector, and explicit search.
    Routes queries and combines results with fast-fail/fallback logic.
    """

    def __init__(self, embedder: Optional["JinaV4Embedder"] = None):
        if embedder is None:
            from ..embedding.embedder import JinaV4Embedder as _JinaV4Embedder
            self.embedder = _JinaV4Embedder()
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
        import time
        start = time.time()
        try:
            from ...db.session import get_db_session
            with get_db_session() as db:
                # Manual strategies
                if strategy == "explicit":
                    return self._explicit_search(db, query, filters, limit)
                if strategy == "fts":
                    return FTSSearcher(db).search(query, filters, limit)
                if strategy == "vector":
                    return VectorSearcher(db, self.embedder).search(query, filters, limit)

                # Auto routing
                if self.router.is_explicit_query(query):
                    results = self._explicit_search(db, query, filters, limit)
                    if results:
                        return results[:limit]
                    # FTS-only fallback
                    fts_results = FTSSearcher(db).search(query, filters, limit)
                    if fts_results:
                        return self._deduplicate_and_rank(fts_results)[:limit]
                    # Vector last resort
                    vector_results = VectorSearcher(db, self.embedder).search(query, filters, limit)
                    return self._deduplicate_and_rank(vector_results)[:limit]

                # Thematic/semantic hybrid
                return self._thematic_search(db, query, filters, limit, fts_weight, vector_weight)
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
        finally:
            duration_ms = (time.time() - start) * 1000
            logger.info("Hybrid search completed", extra=log_timing("hybrid_search", duration_ms, query_length=len(query)))

    def _explicit_search(
        self,
        db: Session,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
    ) -> List[SearchResult]:
        references = self.router.extract_explicit_references(query)
        logger.debug(f"Extracted references: {references}")
        return ExplicitSearcher(db).search(references, filters, limit)

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

        all_results: List[SearchResult] = []

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
        seen_units: Dict[str, SearchResult] = {}
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
                            COALESCE(lu.local_content, lu.content, lu.display_text, lu.bm25_body) as text,
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
                            COALESCE(lu.local_content, lu.content, lu.display_text, lu.bm25_body) as text,
                            lu.citation_string,
                            ld.doc_form,
                            ld.doc_year,
                            ld.doc_number
                        FROM legal_units lu
                        JOIN legal_documents ld ON lu.document_id = ld.id
                        WHERE lu.unit_id = :pasal_id
                    """)

                results = db.execute(query, {"pasal_id": pasal_id}).fetchall()

                search_results: List[SearchResult] = []
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
