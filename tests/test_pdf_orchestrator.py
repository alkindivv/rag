import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite://")

sys.path.append(str(Path(__file__).resolve().parents[1]))
import src.services.pdf.pdf_orchestrator as po  # noqa: E402
from src.services.pdf.pdf_orchestrator import LegalNode  # noqa: E402


def test_serialize_tree_generates_unique_unit_ids():
    class DummyExtractor:
        def extract_text(self, *_):
            return None

    po.UnifiedPDFExtractor = DummyExtractor
    orchestrator = po.PDFOrchestrator()

    root = LegalNode(type="document", number="root", title="Doc")
    pasal1 = LegalNode(type="pasal", number="1")
    pasal1_dup = LegalNode(type="pasal", number="1")
    root.children = [pasal1, pasal1_dup]

    tree = orchestrator._serialize_tree(root, "DOC-1", "Doc Title")
    unit_ids = [child["unit_id"] for child in tree["children"]]

    assert len(unit_ids) == 2
    assert unit_ids[0] != unit_ids[1]
    assert unit_ids[1].startswith(unit_ids[0])

