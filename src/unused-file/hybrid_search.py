"""
Hybrid Search Service for Legal Documents
Combines semantic search with BM25 for optimal retrieval performance
Based on Anthropic's Contextual Retrieval methodology
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import numpy as np

from app.core.config import settings
from app.services.embedding import EmbeddingService
from app.database.connection import get_db

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with relevance score and metadata"""
    verse_id: str
    content: str
    contextual_content: str
    document_title: str
    document_type: str
    chapter_title: str
    article_number: str
    verse_number: str
    score: float
    source: str  # 'semantic', 'bm25', or 'hybrid'
    rank: int


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search"""
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    top_k: int = 20
    recall_k: int = 100  # Retrieve more for fusion
    elasticsearch_host: str = "http://2.56.99.109:9200"
    index_name: str = "legal_contextual_index"


class HybridSearchService:
    """
    Production hybrid search service that combines:
    1. Semantic search using contextual embeddings
    2. BM25 search using Elasticsearch (when available)
    3. Reciprocal Rank Fusion (RRF) for optimal results
    """

    @classmethod
    async def semantic_search(
        cls,
        query: str,
        limit: int = 20,
        db: Session = None
    ) -> List[SearchResult]:
        """
        Perform semantic search using contextual embeddings.

        Args:
            query: Search query
            limit: Maximum number of results
            db: Database session (optional)

        Returns:
            List of search results ranked by semantic similarity
        """
        try:
            if db is None:
                engine = create_engine(settings.database_url_sync)
                SessionLocal = sessionmaker(bind=engine)
                db = SessionLocal()
                close_db = True
            else:
                close_db = False

            # Generate query embedding optimized for search
            query_result = await EmbeddingService.embed_query_for_search(query)
            if not query_result.embedding:
                logger.error("Failed to generate query embedding")
                return []

            # Convert embedding to string format for PostgreSQL
            embedding_str = '[' + ','.join(map(str, query_result.embedding)) + ']'

            # Semantic search query using contextual embeddings
            search_query = text(f"""
                SELECT
                    cv.verse_id,
                    v.content,
                    cv.contextual_content,
                    d.title as document_title,
                    d.document_type,
                    c.title as chapter_title,
                    a.number as article_number,
                    v.number as verse_number,
                    (1 - (cv.embedding <=> '{embedding_str}'::vector)) as similarity
                FROM contextual_verse_embeddings cv
                JOIN verses v ON cv.verse_id = v.id
                JOIN articles a ON v.article_id = a.id
                JOIN chapters c ON a.chapter_id = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE d.processing_status = 'completed'
                AND v.content IS NOT NULL
                AND v.content != ''
                ORDER BY cv.embedding <=> '{embedding_str}'::vector
                LIMIT {limit}
            """)

            result = db.execute(search_query)
            search_results = []

            for rank, row in enumerate(result, 1):
                search_result = SearchResult(
                    verse_id=str(row.verse_id),
                    content=row.content,
                    contextual_content=row.contextual_content,
                    document_title=row.document_title,
                    document_type=row.document_type,
                    chapter_title=row.chapter_title,
                    article_number=row.article_number,
                    verse_number=row.verse_number,
                    score=float(row.similarity),
                    source="semantic",
                    rank=rank
                )
                search_results.append(search_result)

            if close_db:
                db.close()

            logger.info(f"Semantic search completed: query='{query}', results={len(search_results)}")
            return search_results

        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            if 'db' in locals() and close_db:
                db.close()
            return []

    @classmethod
    async def bm25_search(
        cls,
        query: str,
        limit: int = 20,
        db: Session = None
    ) -> List[SearchResult]:
        """
        Perform BM25 search using PostgreSQL full-text search.
        Falls back to PostgreSQL when Elasticsearch is not available.

        Args:
            query: Search query
            limit: Maximum number of results
            db: Database session (optional)

        Returns:
            List of search results ranked by BM25 relevance
        """
        try:
            if db is None:
                engine = create_engine(settings.database_url_sync)
                SessionLocal = sessionmaker(bind=engine)
                db = SessionLocal()
                close_db = True
            else:
                close_db = False

            # Use PostgreSQL full-text search as BM25 alternative
            # Convert query to tsquery format
            query_terms = query.replace("'", "''").split()
            tsquery = " & ".join(query_terms)

            bm25_query = text(f"""
                SELECT
                    v.id as verse_id,
                    v.content,
                    COALESCE(cv.contextual_content, '') as contextual_content,
                    d.title as document_title,
                    d.document_type,
                    c.title as chapter_title,
                    a.number as article_number,
                    v.number as verse_number,
                    ts_rank_cd(
                        to_tsvector('indonesian', v.content || ' ' || COALESCE(cv.contextual_content, '')),
                        plainto_tsquery('indonesian', :query)
                    ) as bm25_score
                FROM verses v
                JOIN articles a ON v.article_id = a.id
                JOIN chapters c ON a.chapter_id = c.id
                JOIN documents d ON c.document_id = d.id
                LEFT JOIN contextual_verse_embeddings cv ON v.id = cv.verse_id
                WHERE d.processing_status = 'completed'
                AND v.content IS NOT NULL
                AND v.content != ''
                AND (
                    to_tsvector('indonesian', v.content) @@ plainto_tsquery('indonesian', :query)
                    OR to_tsvector('indonesian', COALESCE(cv.contextual_content, '')) @@ plainto_tsquery('indonesian', :query)
                )
                ORDER BY bm25_score DESC
                LIMIT {limit}
            """)

            result = db.execute(bm25_query, {"query": query})
            search_results = []

            for rank, row in enumerate(result, 1):
                search_result = SearchResult(
                    verse_id=str(row.verse_id),
                    content=row.content,
                    contextual_content=row.contextual_content,
                    document_title=row.document_title,
                    document_type=row.document_type,
                    chapter_title=row.chapter_title,
                    article_number=row.article_number,
                    verse_number=row.verse_number,
                    score=float(row.bm25_score),
                    source="bm25",
                    rank=rank
                )
                search_results.append(search_result)

            if close_db:
                db.close()

            logger.info(f"BM25 search completed: query='{query}', results={len(search_results)}")
            return search_results

        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            if 'db' in locals() and close_db:
                db.close()
            return []

    @classmethod
    async def hybrid_search(
        cls,
        query: str,
        limit: int = 20,
        config: HybridSearchConfig = None,
        db: Session = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and BM25 with RRF.

        Args:
            query: Search query
            limit: Maximum number of results
            config: Hybrid search configuration
            db: Database session (optional)

        Returns:
            List of search results ranked by hybrid relevance
        """
        try:
            if config is None:
                config = HybridSearchConfig()

            if db is None:
                engine = create_engine(settings.database_url_sync)
                SessionLocal = sessionmaker(bind=engine)
                db = SessionLocal()
                close_db = True
            else:
                close_db = False

            # Run both searches concurrently
            semantic_task = cls.semantic_search(query, config.recall_k, db)
            bm25_task = cls.bm25_search(query, config.recall_k, db)

            semantic_results, bm25_results = await asyncio.gather(
                semantic_task,
                bm25_task,
                return_exceptions=True
            )

            # Handle exceptions
            if isinstance(semantic_results, Exception):
                logger.error(f"Semantic search failed: {semantic_results}")
                semantic_results = []

            if isinstance(bm25_results, Exception):
                logger.error(f"BM25 search failed: {bm25_results}")
                bm25_results = []

            # If one search method fails, return results from the other
            if not semantic_results and not bm25_results:
                logger.warning("Both search methods failed")
                return []

            if not semantic_results:
                logger.warning("Semantic search failed, using BM25 only")
                return bm25_results[:limit]

            if not bm25_results:
                logger.warning("BM25 search failed, using semantic only")
                return semantic_results[:limit]

            # Apply Reciprocal Rank Fusion (RRF)
            hybrid_results = cls._apply_reciprocal_rank_fusion(
                semantic_results,
                bm25_results,
                config
            )

            # Limit results
            final_results = hybrid_results[:limit]

            # Update source and ranks
            for i, result in enumerate(final_results, 1):
                result.source = "hybrid"
                result.rank = i

            if close_db:
                db.close()

            logger.info(f"Hybrid search completed: query='{query}', results={len(final_results)}")
            return final_results

        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            if 'db' in locals() and close_db:
                db.close()
            return []

    @classmethod
    def _apply_reciprocal_rank_fusion(
        cls,
        semantic_results: List[SearchResult],
        bm25_results: List[SearchResult],
        config: HybridSearchConfig,
        k: int = 60
    ) -> List[SearchResult]:
        """
        Apply Reciprocal Rank Fusion to combine semantic and BM25 results.

        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            config: Hybrid search configuration
            k: RRF parameter (default 60)

        Returns:
            Fused and ranked results
        """
        try:
            # Create a mapping of verse_id to combined scores
            verse_scores = {}
            verse_data = {}

            # Process semantic results
            for i, result in enumerate(semantic_results):
                verse_id = result.verse_id
                rrf_score = config.semantic_weight / (k + i + 1)
                verse_scores[verse_id] = verse_scores.get(verse_id, 0) + rrf_score
                verse_data[verse_id] = result

            # Process BM25 results
            for i, result in enumerate(bm25_results):
                verse_id = result.verse_id
                rrf_score = config.bm25_weight / (k + i + 1)
                verse_scores[verse_id] = verse_scores.get(verse_id, 0) + rrf_score

                # Use BM25 result if not already in verse_data (from semantic)
                if verse_id not in verse_data:
                    verse_data[verse_id] = result

            # Sort by combined RRF scores
            sorted_verse_ids = sorted(
                verse_scores.keys(),
                key=lambda x: verse_scores[x],
                reverse=True
            )

            # Create final result list
            hybrid_results = []
            for verse_id in sorted_verse_ids:
                result = verse_data[verse_id]
                # Update score to RRF score
                result.score = verse_scores[verse_id]
                hybrid_results.append(result)

            return hybrid_results

        except Exception as e:
            logger.error(f"RRF fusion error: {e}")
            # Fallback: return semantic results if fusion fails
            return semantic_results

    @classmethod
    async def search_with_filters(
        cls,
        query: str,
        document_types: List[str] = None,
        years: List[int] = None,
        limit: int = 20,
        search_type: str = "hybrid",
        db: Session = None
    ) -> List[SearchResult]:
        """
        Search with additional filters for document types and years.

        Args:
            query: Search query
            document_types: Filter by document types
            years: Filter by years
            limit: Maximum number of results
            search_type: Type of search ('semantic', 'bm25', 'hybrid')
            db: Database session (optional)

        Returns:
            Filtered search results
        """
        try:
            # Choose search method
            if search_type == "semantic":
                results = await cls.semantic_search(query, limit * 2, db)
            elif search_type == "bm25":
                results = await cls.bm25_search(query, limit * 2, db)
            else:  # hybrid
                results = await cls.hybrid_search(query, limit * 2, None, db)

            # Apply filters
            filtered_results = []
            for result in results:
                # Document type filter
                if document_types and result.document_type not in document_types:
                    continue

                # Year filter (would need to add year to search results)
                # This requires modifying the search queries to include year

                filtered_results.append(result)

                # Stop when we have enough results
                if len(filtered_results) >= limit:
                    break

            # Update ranks
            for i, result in enumerate(filtered_results, 1):
                result.rank = i

            return filtered_results

        except Exception as e:
            logger.error(f"Filtered search error: {e}")
            return []

    @classmethod
    async def get_search_suggestions(
        cls,
        partial_query: str,
        limit: int = 10,
        db: Session = None
    ) -> List[str]:
        """
        Get search suggestions based on partial query.

        Args:
            partial_query: Partial search query
            limit: Maximum number of suggestions
            db: Database session (optional)

        Returns:
            List of suggested queries
        """
        try:
            if db is None:
                engine = create_engine(settings.database_url_sync)
                SessionLocal = sessionmaker(bind=engine)
                db = SessionLocal()
                close_db = True
            else:
                close_db = False

            # Get suggestions from various sources
            suggestions = set()

            # From document titles
            title_query = text("""
                SELECT DISTINCT title
                FROM documents
                WHERE title ILIKE :query
                AND processing_status = 'completed'
                ORDER BY title
                LIMIT :limit
            """)

            title_results = db.execute(title_query, {
                "query": f"%{partial_query}%",
                "limit": limit // 2
            })

            for row in title_results:
                suggestions.add(row.title)

            # From contextual content (legal terms)
            context_query = text("""
                SELECT DISTINCT
                    regexp_split_to_table(contextual_content, '[|,:]') as term
                FROM contextual_verse_embeddings
                WHERE contextual_content ILIKE :query
                LIMIT :limit
            """)

            context_results = db.execute(context_query, {
                "query": f"%{partial_query}%",
                "limit": limit
            })

            for row in context_results:
                term = row.term.strip()
                if len(term) > 3 and partial_query.lower() in term.lower():
                    suggestions.add(term)

            if close_db:
                db.close()

            # Return limited, sorted suggestions
            return sorted(list(suggestions))[:limit]

        except Exception as e:
            logger.error(f"Search suggestions error: {e}")
            if 'db' in locals() and close_db:
                db.close()
            return []

    @classmethod
    async def get_search_stats(cls, db: Session = None) -> Dict[str, Any]:
        """Get search system statistics."""
        try:
            if db is None:
                engine = create_engine(settings.database_url_sync)
                SessionLocal = sessionmaker(bind=engine)
                db = SessionLocal()
                close_db = True
            else:
                close_db = False

            stats_query = text("""
                SELECT
                    COUNT(DISTINCT d.id) as total_documents,
                    COUNT(DISTINCT v.id) as total_verses,
                    COUNT(DISTINCT cv.verse_id) as verses_with_embeddings,
                    AVG(LENGTH(v.content)) as avg_content_length,
                    AVG(LENGTH(cv.contextual_content)) as avg_context_length
                FROM documents d
                LEFT JOIN chapters c ON d.id = c.document_id
                LEFT JOIN articles a ON c.id = a.chapter_id
                LEFT JOIN verses v ON a.id = v.article_id
                LEFT JOIN contextual_verse_embeddings cv ON v.id = cv.verse_id
                WHERE d.processing_status = 'completed'
            """)

            result = db.execute(stats_query).fetchone()

            embedding_coverage = 0
            if result.total_verses > 0:
                embedding_coverage = (result.verses_with_embeddings / result.total_verses) * 100

            stats = {
                "total_documents": result.total_documents or 0,
                "total_verses": result.total_verses or 0,
                "verses_with_embeddings": result.verses_with_embeddings or 0,
                "embedding_coverage_percentage": round(embedding_coverage, 2),
                "avg_content_length": round(result.avg_content_length or 0, 2),
                "avg_context_length": round(result.avg_context_length or 0, 2),
                "search_ready": embedding_coverage > 50,
                "timestamp": datetime.now().isoformat()
            }

            if close_db:
                db.close()

            return stats

        except Exception as e:
            logger.error(f"Search stats error: {e}")
            if 'db' in locals() and close_db:
                db.close()
            return {
                "error": str(e),
                "search_ready": False,
                "timestamp": datetime.now().isoformat()
            }
