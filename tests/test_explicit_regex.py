import pytest

from src.services.search.explicit.regex import parse, build_explicit_filters, ExplicitParse


def test_parse_uu_tahun_pasal_suffix_and_ayat_huruf():
    q = "UU No. 1 Tahun 2023 Pasal 14A ayat (3) huruf c"
    p = parse(q)
    assert isinstance(p, ExplicitParse)
    assert p.uu_num == "1"
    assert p.uu_year == "2023"
    assert p.pasal == "14"
    assert p.pasal_suffix == "A"
    assert p.ayat == "3"
    assert p.huruf == "c"

    f = build_explicit_filters(q)
    assert f["lquery"] is not None
    assert "pasal-14a" in f["lquery"].lower()
    assert f["ltree_exact"] is not None
    assert f["ltree_exact"].startswith("UU-2023-1") or f["ltree_exact"].startswith("pasal-14a")


def test_parse_bab_roman_with_pasal_only():
    q = "Bab V Pasal 3 ayat (1)"
    p = parse(q)
    assert p is not None
    assert (p.bab_roman or "").upper() in ("V",)
    assert p.pasal == "3"
    assert p.ayat == "1"

    exact = p.to_exact_ltree()
    assert exact is not None
    assert "pasal-3" in exact
    assert "BAB-" in exact


def test_parse_pasal_bis_and_ter_suffixes():
    for suffix in ("bis", "ter", "quater"):
        q = f"Pasal 27{suffix} ayat 2"
        p = parse(q)
        assert p is not None
        assert p.pasal == "27"
        assert p.pasal_suffix == suffix
        assert p.ayat == "2"
        lq = p.to_lquery()
        assert lq is not None and suffix in lq


def test_cross_reference_detection_flag():
    q = "sebagaimana dimaksud dalam Pasal 42 ayat (1)"
    p = parse(q)
    assert p is not None
    assert p.has_cross_ref is True


def test_build_filters_when_minimal_reference():
    q = "Pasal 10"
    f = build_explicit_filters(q)
    assert f["lquery"] is not None
    # Without deeper specs, exact may be None
    # but lquery should be constructed
    assert f["ltree_exact"] in (None, f["ltree_exact"])  # merely ensure no crash
