import pytest

from src.services.pdf.pdf_orchestrator import PDFOrchestrator


def find_nodes(root, predicate):
    found = []
    def walk(n):
        if predicate(n):
            found.append(n)
        for c in n.get("children", []) or []:
            walk(c)
    walk(root)
    return found


def find_first(root, predicate):
    nodes = find_nodes(root, predicate)
    return nodes[0] if nodes else None


def test_deep_serialization_with_suffix_ayat_huruf_and_unit_path():
    orch = PDFOrchestrator()
    txt = (
        "BAB IV\n"
        "Pasal 14A\n"
        "(1) Ketentuan umum.\n"
        "(2) Ketentuan khusus:\n"
        "a. huruf a konten;\n"
        "b. huruf b konten.\n"
    )
    doc = {"doc_id": "uu_4_2009", "doc_title": "UU No. 4 Tahun 2009", "title": "UU No. 4 Tahun 2009"}

    out = orch.process_txt_content(doc, txt)
    root = out["document_tree"]

    # Find Pasal 14A
    pasal = find_first(root, lambda n: n.get("type") == "pasal" and n.get("number_label") == "14A")
    assert pasal, "Pasal 14A should exist"
    assert pasal.get("unit_path", "").startswith("uu_4_2009"), "unit_path should start with doc_id slug"
    assert ".pasal_14a" in pasal["unit_path"].lower(), "pasal token should be present in unit_path"

    # Find ayat (2)
    ayat2 = find_first(pasal, lambda n: n.get("type") == "ayat" and n.get("number_label") == "2")
    assert ayat2, "Ayat (2) should exist under Pasal 14A"
    assert ayat2["unit_path"].lower().endswith("pasal_14a.ayat_2"), ayat2["unit_path"]

    # Find huruf b
    huruf_b = find_first(ayat2, lambda n: n.get("type") == "huruf" and n.get("number_label").lower() == "b")
    assert huruf_b, "Huruf b should exist under ayat (2)"
    assert huruf_b["unit_path"].lower().endswith("pasal_14a.ayat_2.huruf_b"), huruf_b["unit_path"]

    # Citation string should include title and hierarchy parts
    assert "UU No. 4 Tahun 2009" in huruf_b.get("citation_string", "")
    assert "Pasal 14A" in huruf_b.get("citation_string", "")
    assert "ayat (2)" in huruf_b.get("citation_string", "")
    assert "huruf b" in huruf_b.get("citation_string", "")


def test_cross_reference_augmentation_absolute_and_relative():
    orch = PDFOrchestrator()
    txt = (
        "Pasal 6\n"
        "(1) Ini ayat satu.\n"
        "a. konten A.\n"
        "b. konten B.\n"
        "Pasal 7\n"
        "(1) Ketentuan sebagaimana dimaksud dalam Pasal 6 ayat (1) huruf b berlaku mutatis mutandis.\n"
        "(2) Lihat juga ayat (1) dan huruf b.\n"
    )
    doc = {"doc_id": "uu_xx_yyyy", "doc_title": "UU Dummy", "title": "UU Dummy"}

    out = orch.process_txt_content(doc, txt)
    root = out["document_tree"]

    pasal6 = find_first(root, lambda n: n.get("type") == "pasal" and n.get("number_label") == "6")
    assert pasal6
    ayat1_p6 = find_first(pasal6, lambda n: n.get("type") == "ayat" and n.get("number_label") == "1")
    assert ayat1_p6
    hurufb_p6 = find_first(ayat1_p6, lambda n: n.get("type") == "huruf" and n.get("number_label") == "b")
    assert hurufb_p6

    pasal7 = find_first(root, lambda n: n.get("type") == "pasal" and n.get("number_label") == "7")
    assert pasal7

    # Check references attached at pasal7 and its ayat
    refs_p7 = pasal7.get("reference_contents", [])
    # ayat (1) should reference Pasal 6 ayat (1) huruf b
    ayat1_p7 = find_first(pasal7, lambda n: n.get("type") == "ayat" and n.get("number_label") == "1")
    ayat2_p7 = find_first(pasal7, lambda n: n.get("type") == "ayat" and n.get("number_label") == "2")

    assert ayat1_p7 is not None
    r1 = ayat1_p7.get("reference_contents", [])
    assert any("Pasal 6" in r.get("ref", "") and "ayat (1)" in r.get("ref", "") and "huruf b" in r.get("ref", "") for r in r1), r1

    # Ensure resolved content is that of huruf b under Pasal 6 ayat (1)
    target_unit_id = hurufb_p6["unit_id"]
    assert any(r.get("unit_id") == target_unit_id and isinstance(r.get("content"), str) and len(r.get("content")) > 0 for r in r1)

    # ayat (2) relative refs: 'ayat (1)' and 'huruf b' should resolve within Pasal 7 context and Pasal 7 ayat(2) context
    assert ayat2_p7 is not None
    r2 = ayat2_p7.get("reference_contents", [])
    # relative ayat should point to Pasal 7 ayat (1)
    rel_ayat_target = find_first(pasal7, lambda n: n.get("type") == "ayat" and n.get("number_label") == "1")
    assert any(r.get("unit_id") == rel_ayat_target["unit_id"] for r in r2)

    # relative huruf b should not resolve since ayat (2) has no huruf; ensure it doesn't crash and list may be empty or exclude missing
    assert isinstance(r2, list)
