#!/usr/bin/env python3
"""
Run EXPLAIN ANALYZE for key retrieval paths to validate index engagement:
- ltree: WHERE unit_path ~ (:lq)::lquery
- FTS: WHERE tsv_content @@ to_tsquery('indonesian', :tsq)
- Vector: ORDER BY embedding <=> CAST(:qv AS vector) LIMIT k

Usage: python scripts/explain_plans.py
Environment variables (optional):
  TSQUERY="kewenangan & daerah"
  LIMIT=10
  FORCE_INDEX=1              # SET enable_seqscan=off to bias planner to indexes
  MIN_TS_RANK=0.05           # Apply rank threshold on FTS branch
  IVFFLAT_PROBES=10          # pgvector ivfflat.probes
  HNSW_EF_SEARCH=40          # pgvector hnsw.ef_search
"""
from __future__ import annotations

import json
import os
from typing import List

from sqlalchemy import text
import sys
import os as _os
_HERE = _os.path.dirname(__file__)
_ROOT = _os.path.abspath(_os.path.join(_HERE, '..'))
_SRC = _os.path.join(_ROOT, 'src')
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.db.session import get_db_session


def pick_sample_paths(db):
    # Pick a sample unit_path to build an lquery
    row = db.execute(text("""
        SELECT unit_path::text AS up
        FROM legal_units
        WHERE unit_path IS NOT NULL
        ORDER BY updated_at DESC NULLS LAST
        LIMIT 1
    """)).mappings().first()
    if not row or not row["up"]:
        return None

    # Build a generic lquery: root two labels + wildcard
    parts = row["up"].split('.')
    prefix = '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]
    lq = f"{prefix}.*"
    return lq


def explain_analyze(db, sql: str, params: dict) -> List[str]:
    plan_sql = f"EXPLAIN (ANALYZE, BUFFERS, VERBOSE)\n{sql}"
    rows = db.execute(text(plan_sql), params).all()
    return [r[0] for r in rows]


def print_hint(title: str, plan_lines: List[str], expect_index: List[str]) -> None:
    txt = "\n".join(plan_lines)
    used = any(tok in txt for tok in expect_index)
    if used:
        print(f"[HINT] {title}: index path detected (" + ", ".join([t for t in expect_index if t in txt]) + ")")
    else:
        if "Seq Scan" in txt:
            print(f"[HINT] {title}: sequential scan observed; on small tables this can be normal. Consider FORCE_INDEX=1 to validate index readiness.")
        else:
            print(f"[HINT] {title}: could not detect expected index path.")


def main():
    tsq = os.getenv("TSQUERY", "kewenangan & daerah")
    k = int(os.getenv("LIMIT", "10"))
    force_index = os.getenv("FORCE_INDEX") in ("1", "true", "TRUE")
    min_ts_rank = os.getenv("MIN_TS_RANK")
    ivf_probes = os.getenv("IVFFLAT_PROBES")
    hnsw_ef = os.getenv("HNSW_EF_SEARCH")

    with get_db_session() as db:
        # Session-level toggles
        if force_index:
            db.execute(text("SET enable_seqscan = off"))
        if ivf_probes:
            db.execute(text("SET ivfflat.probes = :p"), {"p": int(ivf_probes)})
        if hnsw_ef:
            db.execute(text("SET hnsw.ef_search = :e"), {"e": int(hnsw_ef)})

        # Prepare lquery
        lq = pick_sample_paths(db)
        if not lq:
            print("No sample unit_path found. Skipping ltree plan.")
        else:
            print("\n== LTREE plan ==")
            sql_ltree = """
                SELECT id
                FROM legal_units
                WHERE unit_path ~ (:lq)::lquery
                ORDER BY nlevel(unit_path), id
                LIMIT :k
            """
            plan = explain_analyze(db, sql_ltree, {"lq": lq, "k": k})
            print("\n".join(plan))
            print_hint("LTREE", plan, ["Index Scan", "Bitmap Index Scan"])  # GiST path shows as Index Scan

        # FTS plan
        print("\n== FTS plan ==")
        sql_fts = """
            WITH q AS (SELECT to_tsquery('indonesian', :tsq) AS q)
            SELECT id
            FROM legal_units, q
            WHERE tsv_content @@ q.q
        """
        if min_ts_rank:
            sql_fts += " AND ts_rank_cd(tsv_content, q.q) >= :r\n"
        sql_fts += """
            ORDER BY ts_rank_cd(tsv_content, q.q) DESC, id
            LIMIT :k
        """
        params = {"tsq": tsq, "k": k}
        if min_ts_rank:
            params["r"] = float(min_ts_rank)
        plan = explain_analyze(db, sql_fts, params)
        print("\n".join(plan))
        print_hint("FTS", plan, ["Bitmap Index Scan", "Index Cond", "Recheck Cond"])  # typical for GIN

        # Vector plan
        print("\n== VECTOR plan ==")
        dim = 384  # default
        qv = [0.0] * dim
        sql_vec = """
            SELECT id
            FROM legal_units
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:qv AS vector) ASC
            LIMIT :k
        """
        plan = explain_analyze(db, sql_vec, {"qv": qv, "k": k})
        print("\n".join(plan))
        print_hint("VECTOR", plan, ["Index Scan", "Bitmap Index Scan", "hnsw"])  # planner text varies


if __name__ == "__main__":
    main()
