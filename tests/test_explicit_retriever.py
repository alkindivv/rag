from types import SimpleNamespace

from src.retriever.hybrid_retriever import hybrid_search


class DummyUnit:
    unit_id = "u1"
    citation = "Pasal 5 ayat (1)"
    bm25_body = "isi"


class FakeQuery:
    def __init__(self, obj):
        self.obj = obj

    def join(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.obj


class FakeSession:
    def __init__(self, obj):
        self.obj = obj

    def query(self, model):
        return FakeQuery(self.obj)


def test_explicit():
    session = FakeSession(DummyUnit())
    res = hybrid_search(session, "UU 8 Tahun 1981 Pasal 5 ayat (1)")
    assert res[0]["citation"] == "Pasal 5 ayat (1)"
