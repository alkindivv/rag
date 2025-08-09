Retrieval.md

Hybrid Retrieval default:
Explisit 
Contoh pertanyaan: 
1. apa isi pasal 5 ayat (1) huruf b UU 8/1981?
2. Ayat 1 huruf b undang undang no 11 tahun 2020 isinya apa?
Criteria
- Query DB legal_units by keys (status (should priortize berlaku/active doc), form, number, year, pasal, ayat, huruf/angka).
- Return leaf + konteks pasal (parent_pasal_id).

Tematik:
Contoh pertanyaan:
1. Perbuatan melawan hukum diatur dalam pasal berapa?
2. Sanksi terkait tindak pidana pencurian diatur dalam pasal dan undang-undang apa? Dan ancaman denda berapa lalu hukum pidananya berapa lama?
3. Jila seseorang malakulan wanprestasi, apa akibat hukumnya?
4. Ketentuan atau definisi perseroan diatur dalam undang undang dan pasal berapa dan apa saja?
5. 
Criteria:
- Embed query (Jina v4 dim=768).
- Vector search top‑10 pasal (HNSW, cosine) dari document_vectors.
- FTS top‑10 leaf dari legal_units (tsvector indonesian pada bm25_body.
- Kombinasikan hasil jawaban menggunakan jina reranker dengan min skor 0.7

Semua ini otomatis di gabungkan dalam Gunakan src/service/retriever/retrieval.py

BELOW IS JUST A TEMPLATE AND YOU SHOULD IMPROVE IT BASED ON MY CODE BASE STRUCTURE

<!-- import httpx
from typing import List
from .base import Reranker, RerankItem
from src.config.settings import settings

from abc import ABC, abstractmethod
from typing import List, Dict

class RerankItem:  # kandidat yang akan direrank
    def __init__(self, id: str, text: str, meta: Dict):
        self.id, self.text, self.meta = id, text, meta

class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, items: List[RerankItem], top_k: int = 10) -> List[RerankItem]: ...
    
class Reranker(Reranker):
    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=30)

    async def rerank(self, query: str, items: List[RerankItem], top_k: int = 10) -> List[RerankItem]:
        headers = {"Authorization": f"Bearer {settings.jina_api_key}"}
        payload = {
            "model": settings.jina_rerank_model,
            "query": query,
            "documents": [i.text for i in items],
            "top_n": top_k
        }
        r = await self.client.post(settings.jina_rerank_base, json=payload, headers=headers)
        r.raise_for_status()
        scores = r.json().get("results", [])
        # Map skor ke items
        ranked = []
        for res in scores:
            idx = res["index"]
            ranked.append(RerankItem(items[idx].id, items[idx].text, {**items[idx].meta, "score": res.get("score", 0)}))
        return ranked -->