"""
FastAPI API for Legal RAG System
Provides REST endpoints for legal document search and LLM integration
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from ..services.search.hybrid_search import HybridSearchService
from ..services.llm.legal_llm import LegalLLMService
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
search_service = HybridSearchService()
llm_service = LegalLLMService()

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    strategy: str = "auto"
    use_reranking: bool = True

class SearchResponse(BaseModel):
    results: List[dict]
    total: int
    query: str
    strategy: str
    reranked: bool
    duration_ms: float

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
    """Search legal documents"""
    try:
        results = search_service.search(
            query=request.query,
            limit=request.limit,
            strategy=request.strategy,
            use_reranking=request.use_reranking
        )
        return SearchResponse(**results)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=LLMResponse)
async def ask_legal_question(request: LLMRequest):
    try:
        # First search for relevant documents
        search_results = search_service.search(
            query=request.query,
            limit=request.context_limit or 5,
            strategy="auto",
            use_reranking=True
        )
        
        # Generate answer with LLM
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
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_get(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Result limit"),
    strategy: str = Query("auto", description="Search strategy"),
    use_reranking: bool = Query(True, description="Use reranking")
):
    """GET endpoint for search (for browser testing)"""
    try:
        results = search_service.search(
            query=query,
            limit=limit,
            strategy=strategy,
            use_reranking=use_reranking
        )
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
