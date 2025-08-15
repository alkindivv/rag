import re
from typing import Any, Dict, List, Optional

import pytest

import src.db.queries as q


class _FakeResult:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    class _Maps:
        def __init__(self, rows):
            self._rows = rows
        def all(self):
            return self._rows

    def mappings(self):
        return _FakeResult._Maps(self._rows)

    def all(self):
        return self._rows


class _FakeConn:
    def __init__(self, expected_sql_patterns: List[str], return_rows: List[Dict[str, Any]], captured: Dict[str, Any]):
        self.expected_sql_patterns = expected_sql_patterns
        self.return_rows = return_rows
        self.captured = captured

    def execute(self, sql_obj, params: Optional[Dict[str, Any]] = None):
        sql_text = getattr(sql_obj, "text", str(sql_obj))
        self.captured["sql"] = sql_text
        self.captured["params"] = params or {}
        # Assert basic expectations
        for pat in self.expected_sql_patterns:
            assert pat in sql_text, f"Expected SQL to contain pattern: {pat}\nSQL:\n{sql_text}"
        return _FakeResult(self.return_rows)


@pytest.fixture()
def sample_row():
    return {
        "id": 1,
        "unit_id": 1,
        "content": "isi",
        "local_content": "isi lokal",
        "citation_string": "UU 1/2020 Pasal 1",
        "unit_type": "PASAL",
        "doc_form": "UU",
        "doc_year": 2020,
        "doc_number": "1",
        "hierarchy_path": "1.1",
        "unit_path": "uu_2020_1.pasal_1",
        "score": 1.0,
    }


def test_search_explicit_exact_sql(monkeypatch, sample_row):
    captured = {}
    fake = _FakeConn(
        expected_sql_patterns=[
            "FROM legal_units lu",
            "JOIN legal_documents d",
            "lu.unit_path = (:lt)::ltree",
        ],
        return_rows=[sample_row],
        captured=captured,
    )
    monkeypatch.setattr(q, "_conn", lambda db: fake)

    out = q.search_explicit(object(), ltree_exact="uu_2020_1.pasal_1", limit=5)
    assert out and out[0]["unit_path"] == sample_row["unit_path"]
    assert captured["params"]["lt"] == "uu_2020_1.pasal_1"


def test_search_explicit_lquery_sql(monkeypatch, sample_row):
    captured = {}
    fake = _FakeConn(
        expected_sql_patterns=[
            "lu.unit_path ~ (:lq)::lquery",
            "ORDER BY nlevel(lu.unit_path)",
        ],
        return_rows=[sample_row],
        captured=captured,
    )
    monkeypatch.setattr(q, "_conn", lambda db: fake)

    out = q.search_explicit(object(), lquery="uu_2020_1.pasal_*", limit=3)
    assert out and captured["params"]["lq"] == "uu_2020_1.pasal_*"


def test_search_fts_sql(monkeypatch, sample_row):
    captured = {}
    fake = _FakeConn(
        expected_sql_patterns=[
            "to_tsquery('indonesian', :tsq)",
            "ts_rank_cd(lu.tsv_content, q.q)",
            "lu.tsv_content @@ q.q",
        ],
        return_rows=[sample_row],
        captured=captured,
    )
    monkeypatch.setattr(q, "_conn", lambda db: fake)

    out = q.search_fts(object(), tsquery="dana & perimbangan", limit=10)
    assert out and captured["params"]["tsq"] == "dana & perimbangan"


def test_search_vector_sql(monkeypatch, sample_row):
    captured = {}
    fake = _FakeConn(
        expected_sql_patterns=[
            "CAST(:qv AS vector)",
            "ORDER BY lu.embedding <=> CAST(:qv AS vector) ASC",
        ],
        return_rows=[sample_row],
        captured=captured,
    )
    monkeypatch.setattr(q, "_conn", lambda db: fake)

    out = q.search_vector(object(), query_vector=[0.1, 0.2], limit=5)
    assert out and isinstance(captured["params"].get("qv"), list)


def test_search_fusion_sql_shapes(monkeypatch, sample_row):
    captured = {}
    fake = _FakeConn(
        expected_sql_patterns=[
            "UNION ALL",
            "ROW_NUMBER() OVER (",
            "PARTITION BY u.unit_id",
            "CASE u.match_type WHEN 'explicit' THEN 0",
        ],
        return_rows=[sample_row],
        captured=captured,
    )
    monkeypatch.setattr(q, "_conn", lambda db: fake)

    out = q.search_fusion(
        object(),
        lquery="uu_2020_1.pasal_*",
        tsquery="dana & perimbangan",
        query_vector=[0.1, 0.2],
        limit=10,
    )
    assert out and out[0]["id"] == sample_row["id"]
    # Ensure all params are present
    assert set(["lq", "tsq", "qv"]).issubset(captured["params"].keys())


def test_count_siblings_sql(monkeypatch):
    captured = {}
    rows = [("AYAT", 3), ("HURUF", 2)]
    fake = _FakeConn(
        expected_sql_patterns=[
            "SELECT id, parent_unit_id FROM legal_units",
            "GROUP BY lu.unit_type",
        ],
        return_rows=rows,
        captured=captured,
    )
    monkeypatch.setattr(q, "_conn", lambda db: fake)

    out = q.count_siblings(object(), unit_row_id="00000000-0000-0000-0000-000000000001")
    assert out == {"AYAT": 3, "HURUF": 2}
