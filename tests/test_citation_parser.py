import pytest

from src.services.search.citation_parser import parse_citation, build_unit_path


def test_parse_basic_uu_pasal_ayat_huruf():
    q = "UU 4/2009 Pasal 149 ayat (2) huruf b"
    c = parse_citation(q)
    assert c is not None
    assert c.get("uu_number") == "4"
    assert c.get("uu_year") == "2009"
    assert c.get("pasal_num") == "149"
    assert c.get("ayat_num") == "2"
    assert c.get("huruf") == "b"

    path = build_unit_path(c)
    assert path == "uu.4.2009.pasal.149.ayat.2.huruf.b"


def test_parse_bab_roman_and_pasal_suffix():
    q = "Bab i Pasal 3A ayat (1)"
    c = parse_citation(q)
    assert c is not None
    # roman should be upper-normalized in build
    assert c.get("bab_roman") in ("I", "i")  # parsing stores lower, we normalize upper in build
    assert c.get("pasal_num") == "3"
    assert c.get("pasal_suffix") == "A"

    path = build_unit_path(c)
    # Note: build_unit_path uppercases suffix
    assert path.startswith("bab.I.pasal.3A")


def test_parse_variants_uu_tahun_format():
    q = "Undang-Undang No. 13 Tahun 2008 pasal 42"
    c = parse_citation(q)
    assert c is not None
    assert c.get("uu_number") == "13"
    assert c.get("uu_year") == "2008"
    assert c.get("pasal_num") == "42"

    path = build_unit_path(c)
    assert "uu.13.2008.pasal.42" in path
