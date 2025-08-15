import pytest

from src.services.pdf.pdf_orchestrator import PDFOrchestrator


def find_first(node, t):
    if not node:
        return None
    if node.get("type") == t:
        return node
    for c in node.get("children", []) or []:
        res = find_first(c, t)
        if res:
            return res
    return None


def find_child(node, t):
    for c in node.get("children", []) or []:
        if c.get("type") == t:
            return c
    return None


def test_orchestrator_serializes_unit_path_and_parents():
    orch = PDFOrchestrator()

    # Minimal synthetic TXT content covering BAB roman, Pasal with suffix, ayat, huruf
    txt = (
        "BAB IV\n"
        "Pasal 14A\n"
        "(1) Ketentuan umum.\n"
        "(2) Ketentuan khusus:\n"
        "a. huruf a konten;\n"
        "b. huruf b konten.\n"
    )

    doc = {
        "doc_id": "uu_4_2009",
        "doc_title": "UU No. 4 Tahun 2009",
        "title": "UU No. 4 Tahun 2009",
    }

    out = orch.process_txt_content(doc, txt)

    assert out.get("document_tree"), "document_tree should be present"
    root = out["document_tree"]
    assert root["type"] == "dokumen"
    assert root["unit_id"] == doc["doc_id"]

    # Locate structural nodes
    bab = find_first(root, "bab")
    assert bab is not None, "BAB node should be detected"
    pasal = find_first(root, "pasal")
    assert pasal is not None, "Pasal node should be detected"
    ayat2 = None
    for c in pasal.get("children", []):
        if c.get("type") == "ayat" and c.get("number_label") == "2":
            ayat2 = c
            break
    assert ayat2 is not None, "Ayat (2) should be detected under pasal"
    huruf_b = None
    for c in ayat2.get("children", []):
        if c.get("type") == "huruf" and c.get("number_label") == "b":
            huruf_b = c
            break
    assert huruf_b is not None, "Huruf b should be detected under ayat (2)"

    # Verify unit_path tokens (ltree-friendly): uu_4_2009.bab_4.pasal_14a.ayat_2.huruf_b
    assert pasal.get("unit_path"), "pasal.unit_path must be emitted"
    assert pasal["unit_path"].startswith("uu_4_2009."), pasal["unit_path"]
    assert ".bab_4." in pasal["unit_path"], pasal["unit_path"]
    assert pasal["unit_path"].endswith("pasal_14a"), pasal["unit_path"]

    assert ayat2.get("unit_path"), "ayat.unit_path must be emitted"
    assert ayat2["unit_path"].endswith("ayat_2"), ayat2["unit_path"]
    assert huruf_b.get("unit_path"), "huruf.unit_path must be emitted"
    assert huruf_b["unit_path"].endswith("huruf_b"), huruf_b["unit_path"]

    # Verify parent_unit_id propagation
    assert bab.get("parent_unit_id") == root["unit_id"], "BAB parent should be doc root"
    assert pasal.get("parent_unit_id") == bab["unit_id"], "Pasal parent should be BAB"
    assert ayat2.get("parent_unit_id") == pasal["unit_id"], "Ayat parent should be Pasal"
    assert huruf_b.get("parent_unit_id") == ayat2["unit_id"], "Huruf parent should be Ayat"

    # Whitespace normalization for display_text/local_content on inline units
    assert ayat2.get("display_text", "").startswith("(2) "), ayat2.get("display_text")
    assert "  " not in ayat2.get("display_text", ""), "display_text should be normalized"
    assert huruf_b.get("display_text", "").lower().startswith("b. "), huruf_b.get("display_text")
    assert huruf_b.get("local_content", ""), "local_content must be present for huruf"
    assert "\n" not in huruf_b.get("local_content", ""), "local_content should be single-line"
