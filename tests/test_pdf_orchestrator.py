import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite://")

sys.path.append(str(Path(__file__).resolve().parents[1]))
import src.services.pdf.pdf_orchestrator as po  # noqa: E402
from src.services.pdf.pdf_orchestrator import LegalNode  # noqa: E402
from src.schemas.document_contract import validate_document_json


def test_serialize_tree_generates_unique_unit_ids_and_metadata():
    class DummyExtractor:
        def extract_text(self, *_):
            return None

    po.UnifiedPDFExtractor = DummyExtractor
    orchestrator = po.PDFOrchestrator()

    root = LegalNode(type="document", number="root", title="Doc")
    pasal1 = LegalNode(type="pasal", number="1")
    ayat1 = LegalNode(type="ayat", number="1")
    pasal1.children = [ayat1]
    pasal1_dup = LegalNode(type="pasal", number="1")
    root.children = [pasal1, pasal1_dup]

    tree = orchestrator._serialize_tree(root, "DOC-1", "Doc Title")

    unit_ids = [child["unit_id"] for child in tree["children"]]
    assert len(unit_ids) == 2
    assert unit_ids[0] != unit_ids[1]

    pasal = tree["children"][0]
    ayat = pasal["children"][0]

    # parent pasal id and seq sort key
    assert ayat["parent_pasal_id"] == pasal["unit_id"]
    assert ayat["seq_sort_key"].startswith("0001")
    assert "Pasal 1" in pasal["citation_string"]
    assert "ayat (1)" in ayat["citation_string"]


def test_validate_document_json_success():
    doc = {
        "doc_source": "BPK",
        "doc_id": "UU-2024-1",
        "doc_type": "Peraturan",
        "doc_title": "Test Doc",
        "doc_teu": "Gov",
        "doc_number": "1",
        "doc_form": "UU",
        "doc_form_short": "UU",
        "doc_year": 2024,
        "doc_place_enacted": "Jakarta",
        "doc_date_enacted": "2024-01-01",
        "doc_date_promulgated": "2024-01-01",
        "doc_date_effective": "2024-01-01",
        "doc_subject": ["TEST"],
        "doc_status": "Berlaku",
        "doc_language": "Bahasa Indonesia",
        "doc_location": "Pusat",
        "doc_field": "HUKUM",
        "relationships": {
            "mengubah": [],
            "diubah_dengan": [],
            "mencabut": [],
            "dicabut_dengan": [],
            "menetapkan": []
        },
        "detail_url": "http://detail",
        "source_url": "http://source",
        "pdf_url": "http://pdf",
        "uji_materi_pdf_url": None,
        "uji_materi": [],
        "pdf_path": "a.pdf",
        "text_path": "a.txt",
        "doc_content": "content",
        "doc_processing_status": "pdf_processed",
        "last_updated": "2024-01-01T00:00:00",
        "document_tree": {
            "type": "dokumen",
            "unit_id": "UU-2024-1",
            "number_label": None,
            "ordinal_int": 0,
            "ordinal_suffix": "",
            "label_display": "Test Doc",
            "seq_sort_key": "0000|",
            "citation_string": "Test Doc",
            "path": [{"type": "dokumen", "label": "Test Doc", "unit_id": "UU-2024-1"}],
            "title": "Test Doc",
            "content": None,
            "children": []
        }
    }

    result = validate_document_json(doc)
    assert result.doc_id == "UU-2024-1"

