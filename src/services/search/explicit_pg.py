"""Explicit PostgreSQL-backed retrieval service (ltree-first scaffold).

This service resolves explicit legal citations (UU/Pasal/Ayat/Huruf, BAB/Bagian/Paragraf)
using ltree-based paths. For Plan P0, we scaffold interfaces and return
unified schema; full SQL integration will be wired with Alembic migration.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from .citation_parser import parse_citation, build_unit_path
from ...config.settings import settings
from ...utils.logging import get_logger, log_error
from ...db.session import engine

logger = get_logger(__name__)


class ExplicitPGService:
    """Service for resolving explicit citations.

    Note: DB integration is intentionally abstracted. In tests we monkeypatch
    the DB call to keep unit tests hermetic.
    """

    def __init__(self) -> None:
        pass

    async def resolve_by_citation(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve explicit citation to unified results.

        Returns a unified dict with `results` and `metadata`.
        In this scaffold, we only parse and return empty results unless tests
        patch in DB resolution. This preserves contract and routing behavior.
        """
        citation = parse_citation(query)
        unit_path = build_unit_path(citation or {}) if citation else None

        logger.info("ExplicitPGService.resolve_by_citation", extra={
            "event": "explicit_pg_resolve",
            "query": query,
            "parsed": citation,
            "unit_path": unit_path,
        })

        results: List[Dict[str, Any]] = []
        metadata: Dict[str, Any] = {
            "query": query,
            "strategy": "explicit_pg",
            "search_type": "explicit",
            "total_results": 0,
            "feature_flags": {
                "NEW_PG_RETRIEVAL": settings.NEW_PG_RETRIEVAL,
                "USE_SQL_FUSION": settings.USE_SQL_FUSION,
                "USE_RERANKER": settings.USE_RERANKER,
            },
        }

        # If we cannot parse an explicit structure, return empty results with metadata
        if not citation:
            return {"results": results, "metadata": metadata}

        # Execute DB lookup in a thread to avoid blocking the event loop
        try:
            db_results = await asyncio.to_thread(self._query_db, unit_path, query, k, filters)
            results = db_results
            metadata["total_results"] = len(results)
            return {"results": results[:k], "metadata": metadata}
        except Exception as e:
            logger.error("ExplicitPGService DB lookup failed", extra=log_error(e, context={
                "query": query,
                "unit_path": unit_path,
            }))
            # Graceful degradation: return empty unified response
            return {"results": [], "metadata": metadata}

    def _query_db(self, unit_path: Optional[str], raw_query: str, k: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synchronous DB lookup using SQLAlchemy engine.

        - Prefer exact ltree match on unit_path when available
        - Fallback to FTS search on tsv_content
        Returns list of unified dicts
        """
        dialect = engine.url.get_backend_name()
        if dialect != "postgresql":
            return []

        rows: List[Dict[str, Any]] = []
        with engine.connect() as conn:
            conn = conn.execution_options(stream_results=False)

            # 1) Exact unit_path match (most precise)
            if unit_path:
                try:
                    exact_sql = text(
                        """
                        SELECT id, unit_id, unit_type::text AS unit_type, title, content, citation_string
                        FROM legal_units
                        WHERE unit_path = text2ltree(:path)
                        LIMIT :k
                        """
                    )
                    res = conn.execute(exact_sql, {"path": unit_path, "k": k})
                    for r in res.mappings():
                        rows.append({
                            "id": r.get("id"),
                            "unit_id": r.get("unit_id"),
                            "unit_type": r.get("unit_type"),
                            "title": r.get("title"),
                            "content": r.get("content"),
                            "citation_string": r.get("citation_string"),
                            "score": 1.0,
                        })
                except Exception as e:
                    logger.warning("Exact unit_path lookup failed; falling back to FTS", extra=log_error(e))

            if rows:
                return rows

            # 2) FTS fallback using indonesian config
            try:
                fts_sql = text(
                    """
                    SELECT id, unit_id, unit_type::text AS unit_type, title, content, citation_string,
                           ts_rank(tsv_content, plainto_tsquery('indonesian', :q)) AS score
                    FROM legal_units
                    WHERE tsv_content @@ plainto_tsquery('indonesian', :q)
                    ORDER BY score DESC
                    LIMIT :k
                    """
                )
                res = conn.execute(fts_sql, {"q": raw_query, "k": k})
                for r in res.mappings():
                    rows.append({
                        "id": r.get("id"),
                        "unit_id": r.get("unit_id"),
                        "unit_type": r.get("unit_type"),
                        "title": r.get("title"),
                        "content": r.get("content"),
                        "citation_string": r.get("citation_string"),
                        "score": float(r.get("score") or 0.0),
                    })
            except Exception as e:
                logger.warning("FTS fallback failed", extra=log_error(e))

        return rows
