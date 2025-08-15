from __future__ import annotations
"""
Unified search response schemas for Legal RAG system.

P0 goal: provide a single, stable response shape that all search services
(hybrid/vector/fts) can return, and the API can expose.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    """A single search hit from any retrieval strategy."""
    id: Optional[int] = Field(default=None, description="Primary unit id")
    unit_id: Optional[str] = Field(default=None, description="Alias of unit id when present")
    content: str = Field(default="", description="Unit content/snippet")
    citation_string: Optional[str] = Field(default=None, description="Human-readable citation")
    score: Optional[float] = Field(default=None, description="Retriever score (normalized where possible)")

    unit_type: Optional[str] = Field(default=None, description="Unit type: PASAL/AYAT/HURUF/ANGKA")
    doc_form: Optional[str] = Field(default=None, description="UU/PP/Perpres/etc.")
    doc_year: Optional[int] = Field(default=None)
    doc_number: Optional[str] = Field(default=None)

    hierarchy_path: Optional[str] = Field(default=None, description="Legacy path; will be replaced by unit_path (ltree)")
    unit_path: Optional[str] = Field(default=None, description="ltree path when available")

    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchMetadata(BaseModel):
    """Meta info describing the search execution and result set."""
    query: Optional[str] = None
    search_type: Optional[str] = Field(default=None, description="vector|bm25_fts|hybrid|explicit|...")
    strategy: Optional[str] = Field(default=None, description="auto|hybrid|vector_only|bm25_only|explicit_first|...")
    total_results: int = 0
    limit: Optional[int] = None
    duration_ms: Optional[float] = None

    # Debug and flags
    fts_query: Optional[str] = None
    used_reranker: Optional[bool] = None
    feature_flags: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Unified search response returned by services and API."""
    results: List[SearchHit]
    metadata: SearchMetadata
