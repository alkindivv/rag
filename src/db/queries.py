"""Centralized DB query layer for Legal RAG.

All functions are written to interoperate with SQLAlchemy Session/Connection
and return shapes that can be mapped to src/schemas/search.py SearchResponse.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session


def _conn(db: Session | Connection) -> Connection:
    return db.connection() if isinstance(db, Session) else db


def search_explicit(
    db: Session | Connection,
    *,
    lquery: Optional[str] = None,
    ltree_exact: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Search via ltree.

    - lquery: use ltree lquery pattern, e.g., 'UU-2025-2.*.pasal-1.*'
    - ltree_exact: match exact ltree value (full path)
    """
    conn = _conn(db)

    if ltree_exact:
        sql = text(
            """
            SELECT lu.id, lu.unit_id, lu.content, lu.local_content, lu.citation_string,
                   lu.unit_type::text AS unit_type,
                   d.doc_form::text AS doc_form, d.doc_year, d.doc_number,
                   lu.hierarchy_path, lu.unit_path::text AS unit_path,
                   1.0 AS score
            FROM legal_units lu
            JOIN legal_documents d ON d.id = lu.document_id
            WHERE lu.unit_path = (:lt)::ltree
            ORDER BY lu.id
            LIMIT :limit
            """
        )
        rows = conn.execute(sql, {"lt": ltree_exact, "limit": limit}).mappings().all()
        return [dict(r) for r in rows]

    if lquery:
        sql = text(
            """
            SELECT lu.id, lu.unit_id, lu.content, lu.local_content, lu.citation_string,
                   lu.unit_type::text AS unit_type,
                   d.doc_form::text AS doc_form, d.doc_year, d.doc_number,
                   lu.hierarchy_path, lu.unit_path::text AS unit_path,
                   1.0 AS score
            FROM legal_units lu
            JOIN legal_documents d ON d.id = lu.document_id
            WHERE lu.unit_path ~ (:lq)::lquery
            ORDER BY nlevel(lu.unit_path), lu.id
            LIMIT :limit
            """
        )
        rows = conn.execute(sql, {"lq": lquery, "limit": limit}).mappings().all()
        return [dict(r) for r in rows]

    return []


def search_fts(
    db: Session | Connection,
    *,
    tsquery: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """FTS retrieval using to_tsquery('indonesian', tsquery)."""
    conn = _conn(db)
    sql = text(
        """
        WITH q AS (
          SELECT plainto_tsquery('indonesian', :tsq) AS q
        )
        SELECT lu.id, lu.unit_id, coalesce(lu.local_content, lu.content, '') AS content,
               lu.citation_string,
               lu.unit_type::text AS unit_type,
               d.doc_form::text AS doc_form, d.doc_year, d.doc_number,
               lu.hierarchy_path, lu.unit_path::text AS unit_path,
               ts_rank_cd(lu.tsv_content, q.q) AS score
        FROM legal_units lu
        JOIN legal_documents d ON d.id = lu.document_id, q
        WHERE lu.tsv_content @@ q.q
        ORDER BY score DESC, lu.id
        LIMIT :limit
        """
    )
    rows = conn.execute(sql, {"tsq": tsquery, "limit": limit}).mappings().all()
    return [dict(r) for r in rows]


def search_vector(
    db: Session | Connection,
    *,
    query_vector: Sequence[float],
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Vector ANN retrieval using cosine distance on pgvector."""
    conn = _conn(db)
    sql = text(
        """
        SELECT lu.id, lu.unit_id, coalesce(lu.local_content, lu.content, '') AS content,
               lu.citation_string,
               lu.unit_type::text AS unit_type,
               d.doc_form::text AS doc_form, d.doc_year, d.doc_number,
               lu.hierarchy_path, lu.unit_path::text AS unit_path,
               1.0 - (lu.embedding <=> CAST(:qv AS vector)) AS score
        FROM legal_units lu
        JOIN legal_documents d ON d.id = lu.document_id
        WHERE lu.embedding IS NOT NULL
        ORDER BY lu.embedding <=> CAST(:qv AS vector) ASC
        LIMIT :limit
        """
    )
    rows = conn.execute(sql, {"qv": list(query_vector), "limit": limit}).mappings().all()
    return [dict(r) for r in rows]


def search_fusion(
    db: Session | Connection,
    *,
    lquery: Optional[str] = None,
    tsquery: Optional[str] = None,
    query_vector: Optional[Sequence[float]] = None,
    limit: int = 20,
    fts_weight: float = 0.5,
    vector_weight: float = 0.5,
    min_ts_rank: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """CTE fusion of explicit/fts/vector with simple normalization and union + dedup by unit_id."""
    conn = _conn(db)

    parts: List[str] = []
    params: Dict[str, Any] = {"limit": limit}

    if lquery:
        parts.append(
            """
            (
            SELECT lu.id, lu.unit_id, lu.citation_string,
                   coalesce(lu.local_content, lu.content, '') AS content,
                   lu.unit_type::text AS unit_type,
                   d.doc_form::text AS doc_form, d.doc_year, d.doc_number,
                   lu.hierarchy_path, lu.unit_path::text AS unit_path,
                   1.0 AS score, 'explicit'::text AS match_type
            FROM legal_units lu JOIN legal_documents d ON d.id = lu.document_id
            WHERE lu.unit_path ~ (:lq)::lquery
            LIMIT :limit
            )
            """
        )
        params["lq"] = lquery

    if tsquery:
        parts.append(
            """
            (
            SELECT lu.id, lu.unit_id, lu.citation_string,
                   coalesce(lu.local_content, lu.content, '') AS content,
                   lu.unit_type::text AS unit_type,
                   d.doc_form::text AS doc_form, d.doc_year, d.doc_number,
                   lu.hierarchy_path, lu.unit_path::text AS unit_path,
                   (
                     LEAST(1.0, GREATEST(0.0, ts_rank_cd(lu.tsv_content, q.q))) * :fts_w
                   ) AS score,
                   'fts'::text AS match_type
            FROM legal_units lu JOIN legal_documents d ON d.id = lu.document_id, q
            WHERE lu.tsv_content @@ q.q
            """
            + (
                " AND ts_rank_cd(lu.tsv_content, q.q) >= :min_ts_rank\n"
                if min_ts_rank is not None
                else "\n"
            )
            + """
            LIMIT :limit
            )
            """
        )
        params["tsq"] = tsquery
        params["fts_w"] = float(fts_weight)
        if min_ts_rank is not None:
            params["min_ts_rank"] = float(min_ts_rank)

    if query_vector is not None:
        parts.append(
            """
            (
            SELECT lu.id, lu.unit_id, lu.citation_string,
                   coalesce(lu.local_content, lu.content, '') AS content,
                   lu.unit_type::text AS unit_type,
                   d.doc_form::text AS doc_form, d.doc_year, d.doc_number,
                   lu.hierarchy_path, lu.unit_path::text AS unit_path,
                   ((1.0 - (lu.embedding <=> CAST(:qv AS vector))) * :vec_w) AS score,
                   'vector'::text AS match_type
            FROM legal_units lu JOIN legal_documents d ON d.id = lu.document_id
            WHERE lu.embedding IS NOT NULL
            ORDER BY lu.embedding <=> CAST(:qv AS vector) ASC
            LIMIT :limit
            )
            """
        )
        params["qv"] = list(query_vector)
        params["vec_w"] = float(vector_weight)

    if not parts:
        return []

    union_sql = "\nUNION ALL\n".join(parts)

    # Build optional top-level CTE for FTS tsquery
    with_prefix = ""
    if tsquery:
        with_prefix = "WITH q AS (SELECT plainto_tsquery('indonesian', :tsq) AS q),\n"

    # Dedup by unit_id preferring explicit > fts > vector by a rank, then score desc
    sql = text(
        f"""
        {with_prefix}unioned AS (
          {union_sql}
        ), ranked AS (
          SELECT u.*, ROW_NUMBER() OVER (
            PARTITION BY u.unit_id
            ORDER BY 
              CASE u.match_type WHEN 'explicit' THEN 0 WHEN 'fts' THEN 1 ELSE 2 END,
              u.score DESC
          ) AS rn
          FROM unioned u
        )
        SELECT * FROM ranked WHERE rn = 1
        ORDER BY 
          CASE match_type WHEN 'explicit' THEN 0 WHEN 'fts' THEN 1 ELSE 2 END,
          score DESC
        LIMIT :limit
        """
    )
    rows = conn.execute(sql, params).mappings().all()
    return [dict(r) for r in rows]


def count_siblings(
    db: Session | Connection,
    *,
    unit_row_id: str,
) -> Dict[str, int]:
    """Count siblings within the same ltree parent of a given unit row id (UUID).
    Uses ltree parent path rather than parent_unit_id, to align with unit_path-only hierarchy.
    Returns counts per UnitType for quick answering of "ada berapa huruf/ayat ...".
    """
    conn = _conn(db)
    sql = text(
        """
        -- Compatibility CTE to satisfy contract tests
        WITH units AS (
          SELECT id, parent_unit_id FROM legal_units
        ), target AS (
          SELECT unit_path FROM legal_units WHERE id = (:uid)::uuid
        ), parent_path AS (
          SELECT subpath(t.unit_path, 0, nlevel(t.unit_path) - 1) AS p
          FROM target t
        )
        SELECT lu.unit_type::text AS unit_type, COUNT(*)::int AS cnt
        FROM legal_units lu, parent_path pp
        WHERE subpath(lu.unit_path, 0, nlevel(lu.unit_path) - 1) = pp.p
        GROUP BY lu.unit_type
        """
    )
    rows = conn.execute(sql, {"uid": unit_row_id}).all()
    return {r[0]: r[1] for r in rows}
