"""
FastAPI API for Legal RAG System
Provides REST endpoints for legal document search and LLM integration
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from ..services.search.vector_search import VectorSearchService, SearchFilters
from ..services.llm.legal_llm import LegalLLMService
from ..services.search.query_optimizer import get_query_optimizer, get_optimization_stats
from ..services.embedding.cache import get_cache_performance_report
from ..config.settings import settings
from ..utils.logging import get_logger

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
search_service = VectorSearchService()
llm_service = LegalLLMService()
query_optimizer = get_query_optimizer()

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    limit: int = 15
    use_reranking: bool = False
    filters: Optional[dict] = None

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

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search legal documents using async vector search"""
    try:
        # Convert filters if provided
        filters = None
        if request.filters:
            filters = SearchFilters(**request.filters)

        results = await search_service.search_async(
            query=request.query,
            k=request.limit,
            filters=filters,
            use_reranking=request.use_reranking
        )

        # Convert SearchResult objects to dicts for API response
        api_results = {
            "results": [result.to_dict() for result in results["results"]],
            "metadata": results["metadata"]
        }
        return SearchResponse(**api_results)
    except Exception as e:
        logger.error(f"Async search failed: {e}")
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
            use_reranking=True
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

@app.get("/search")
async def search_get(
    query: str = Query(..., description="Search query"),
    limit: int = Query(15, description="Result limit"),
    use_reranking: bool = Query(False, description="Use reranking")
):
    """GET endpoint for async search (for browser testing)"""
    try:
        results = await search_service.search_async(
            query=query,
            k=limit,
            use_reranking=use_reranking
        )

        # Convert SearchResult objects to dicts for API response
        api_results = {
            "results": [result.to_dict() for result in results["results"]],
            "metadata": results["metadata"]
        }
        return api_results
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
