from src.service.pdf.orchestrator import build_tree


def test_build_tree_simple():
    pages = ["Pasal 5\n(1) Setiap orang\n a. pertama\n b. kedua"]
    tree = build_tree(pages)
    pasal = tree["children"][0]
    assert pasal["number"] == "5"
    ayat = pasal["children"][0]
    assert ayat["number"] == "1"
    assert ayat["children"][0]["number"] == "a"
