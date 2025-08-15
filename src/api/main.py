"""
FastAPI API for Legal RAG System
Provides REST endpoints for legal document search and LLM integration
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from ..services.search.vector_search import SearchFilters
from ..services.search.hybrid_search import HybridSearchService
from ..services.llm.legal_llm import LegalLLMService
from ..services.search.query_optimizer import get_query_optimizer, get_optimization_stats
from ..services.embedding.cache import get_cache_performance_report
from ..config.settings import settings
from ..utils.logging import get_logger
from ..schemas.search import SearchResponse as UnifiedSearchResponse, SearchHit, SearchMetadata

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Legal RAG API",
    description="API for legal document search and LLM integration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
search_service = HybridSearchService()
llm_service = LegalLLMService()
query_optimizer = get_query_optimizer()

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    limit: int = 15
    use_reranking: bool = False
    filters: Optional[dict] = None
    strategy: str = "auto"  # auto, hybrid, vector_only, bm25_only

class SearchResponse(BaseModel):
    results: List[dict]
    metadata: dict

class LLMRequest(BaseModel):
    query: str
    context_limit: int = 5
    temperature: float = 0.3
    max_tokens: int = 1000

class LLMResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float
    duration_ms: float

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "legal-rag-api"}

@app.post("/search", response_model=UnifiedSearchResponse)
async def search_documents(request: SearchRequest):
    """Search legal documents using hybrid search (vector + BM25 + RRF)"""
    try:
        # Convert filters if provided
        filters = None
        if request.filters:
            filters = SearchFilters(**request.filters)

        raw = await search_service.search_async(
            query=request.query,
            k=request.limit,
            filters=filters,
            strategy=request.strategy
        )

        # Normalize to unified schema and return directly for strict validation
        unified = _normalize_to_unified_response(
            raw, query=request.query, limit=request.limit, strategy=request.strategy
        )
        return unified
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/cache")
def get_cache_stats():
    """Get embedding cache performance statistics"""
    try:
        return get_cache_performance_report()
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/optimization")
def get_optimization_stats_endpoint():
    """Get query optimization performance statistics"""
    try:
        return get_optimization_stats()
    except Exception as e:
        logger.error(f"Error getting optimization stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/metrics")
def get_performance_metrics():
    """Get comprehensive performance metrics"""
    try:
        cache_stats = get_cache_performance_report()
        optimization_stats = get_optimization_stats()

        return {
            "cache": cache_stats,
            "optimization": optimization_stats,
            "system": {
                "embedding_dim": settings.embedding_dim,
                "vector_search_k": settings.vector_search_k,
                "citation_confidence": settings.citation_confidence_threshold
            }
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=LLMResponse)
async def ask_legal_question(request: LLMRequest):
    try:
        # First search for relevant documents using async search
        search_results = await search_service.search_async(
            query=request.query,
            k=request.context_limit or 5,
            # Force hybrid retrieval when feature is enabled; else keep auto
            strategy=("hybrid" if settings.NEW_PG_RETRIEVAL else "auto"),
            # Respect global reranker flag for pinning explicit-first downstream
            use_reranking=bool(settings.USE_RERANKER)
        )

        # Generate answer with LLM (LLM service now accepts SearchResult objects directly)
        answer = await llm_service.generate_answer(
            query=request.query,
            context=search_results["results"],
            temperature=request.temperature or 0.3,
            max_tokens=request.max_tokens or 1000
        )

        return LLMResponse(
            answer=answer["answer"],
            sources=answer["sources"],
            confidence=answer["confidence"],
            duration_ms=answer["duration_ms"]
        )
    except Exception as e:
        logger.error(f"Error generating async answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search", response_model=UnifiedSearchResponse)
async def search_get(
    query: str = Query(..., description="Search query"),
    limit: int = Query(15, description="Result limit"),
    use_reranking: bool = Query(False, description="Use reranking")
):
    """GET endpoint for async search (for browser testing)"""
    try:
        raw = await search_service.search_async(
            query=query,
            k=limit,
            use_reranking=use_reranking
        )

        unified = _normalize_to_unified_response(
            raw, query=query, limit=limit, strategy="auto"
        )
        return unified
    except Exception as e:
        logger.error(f"Async search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


def _normalize_to_unified_response(
    raw_results,
    *,
    query: Optional[str],
    limit: Optional[int],
    strategy: Optional[str]
) -> UnifiedSearchResponse:
    """
    Normalize various service return shapes into UnifiedSearchResponse.

    Supported inputs:
    - List[SearchResult] (objects with .to_dict())
    - List[dict]
    - {"results": List[...], "metadata": {...}}
    """
    try:
        # Extract list case
        if isinstance(raw_results, list):
            hits: List[SearchHit] = []
            for item in raw_results:
                if hasattr(item, "to_dict"):
                    data = item.to_dict()
                elif isinstance(item, dict):
                    data = item
                else:
                    # Fallback: attempt to serialize known attrs
                    data = {
                        "id": getattr(item, "id", None),
                        "unit_id": getattr(item, "unit_id", None),
                        "content": getattr(item, "content", ""),
                        "citation_string": getattr(item, "citation_string", None),
                        "score": getattr(item, "score", None),
                        "unit_type": getattr(item, "unit_type", None),
                        "doc_form": getattr(item, "doc_form", None),
                        "doc_year": getattr(item, "doc_year", None),
                        "doc_number": getattr(item, "doc_number", None),
                        "hierarchy_path": getattr(item, "hierarchy_path", None),
                        "metadata": getattr(item, "metadata", {}) or {},
                    }
                # Coerce unit_id to string if provided as int
                try:
                    if isinstance(data, dict) and data.get("unit_id") is not None and not isinstance(data.get("unit_id"), str):
                        data["unit_id"] = str(data["unit_id"])
                except Exception:
                    pass
                hits.append(SearchHit(**data))

            meta = SearchMetadata(
                query=query,
                strategy=strategy,
                total_results=len(hits),
                limit=limit,
                feature_flags={
                    "NEW_PG_RETRIEVAL": settings.NEW_PG_RETRIEVAL,
                    "USE_SQL_FUSION": settings.USE_SQL_FUSION,
                    "USE_RERANKER": settings.USE_RERANKER,
                },
            )
            return UnifiedSearchResponse(results=hits, metadata=meta)

        # Dict case with results key
        if isinstance(raw_results, dict) and "results" in raw_results:
            raw_list = raw_results.get("results", [])
            hits: List[SearchHit] = []
            for item in raw_list:
                data = item.to_dict() if hasattr(item, "to_dict") else item
                if isinstance(data, dict) and data.get("unit_id") is not None and not isinstance(data.get("unit_id"), str):
                    try:
                        data["unit_id"] = str(data["unit_id"])
                    except Exception:
                        pass
                hits.append(SearchHit(**data))

            raw_meta = raw_results.get("metadata", {}) or {}
            meta = SearchMetadata(**{
                "query": raw_meta.get("query", query),
                "search_type": raw_meta.get("search_type"),
                "strategy": raw_meta.get("strategy", strategy),
                "total_results": raw_meta.get("total_results", len(hits)),
                "limit": raw_meta.get("limit", limit),
                "duration_ms": raw_meta.get("duration_ms"),
                "fts_query": raw_meta.get("fts_query"),
                "used_reranker": raw_meta.get("used_reranker"),
                "feature_flags": {
                    "NEW_PG_RETRIEVAL": settings.NEW_PG_RETRIEVAL,
                    "USE_SQL_FUSION": settings.USE_SQL_FUSION,
                    "USE_RERANKER": settings.USE_RERANKER,
                },
            })
            return UnifiedSearchResponse(results=hits, metadata=meta)

        # Unknown shape -> empty safe default
        return UnifiedSearchResponse(
            results=[],
            metadata=SearchMetadata(
                query=query,
                strategy=strategy,
                total_results=0,
                limit=limit,
                feature_flags={
                    "NEW_PG_RETRIEVAL": settings.NEW_PG_RETRIEVAL,
                    "USE_SQL_FUSION": settings.USE_SQL_FUSION,
                    "USE_RERANKER": settings.USE_RERANKER,
                },
            ),
        )
    except Exception as e:
        logger.error(f"Failed to normalize search response: {e}")
        return UnifiedSearchResponse(
            results=[],
            metadata=SearchMetadata(
                query=query,
                strategy=strategy,
                total_results=0,
                limit=limit,
                feature_flags={
                    "NEW_PG_RETRIEVAL": settings.NEW_PG_RETRIEVAL,
                    "USE_SQL_FUSION": settings.USE_SQL_FUSION,
                    "USE_RERANKER": settings.USE_RERANKER,
                },
            ),
        )
