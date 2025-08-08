"""
Simple Embedding Service
ONE FILE to replace ALL the overengineered embedding complexity.

This is the ACTUAL implementation of KISS principle for embeddings:
- Direct integration with Google Gemini
- Context7-compatible content preparation
- Simple chunk embedding without overengineering
- Legal document optimization in 20 lines

Author: KISS Principle Implementation
Purpose: Simple but powerful embedding generation for legal documents
"""

import os
import re
import time
import logging
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Simple embedding result with just what we need."""
    embedding: List[float]
    text: str
    token_count: int
    processing_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


class SimpleEmbedding:
    """
    KISS Embedding Service - Simple but Powerful

    Does ONE thing well: Creates good embeddings for legal document chunks.
    No overengineering, no complex abstractions, just practical embedding.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "models/text-embedding-004"):
        """Initialize with minimal config."""
        self.model = model

        # Get API key from parameter, environment variable, or .env file
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            logger.warning("No GEMINI_API_KEY found. Embedding functionality will be limited.")
            return

        # Configure Gemini
        try:
            genai.configure(api_key=self.api_key)
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {str(e)}")
            self.api_key = None

        # Simple legal term boosting patterns
        self.legal_terms = [
            'pasal', 'ayat', 'huruf', 'bab', 'undang-undang', 'peraturan',
            'pemerintah', 'negara', 'republik', 'indonesia', 'presiden',
            'menteri', 'hukum', 'hak', 'kewajiban', 'sanksi', 'pidana'
        ]

    def embed_chunk(self, chunk_content: str, chunk_metadata: Dict[str, Any] = None) -> EmbeddingResult:
        """
        Embed a single chunk - KISS approach.

        Args:
            chunk_content: The text content to embed
            chunk_metadata: Optional metadata for context enhancement

        Returns:
            EmbeddingResult with embedding and metadata
        """
        start_time = time.time()

        # Check if API key is available
        if not self.api_key:
            return EmbeddingResult(
                embedding=[],
                text=chunk_content,
                token_count=len(chunk_content.split()),
                processing_time=time.time() - start_time,
                success=False,
                error="No API key available"
            )

        try:
            # Prepare content for embedding (Context7 approach)
            embedding_text = self._prepare_for_embedding(chunk_content, chunk_metadata or {})

            # Get embedding from Gemini
            result = genai.embed_content(
                model=self.model,
                content=embedding_text,
                task_type="retrieval_document"
            )

            processing_time = time.time() - start_time

            return EmbeddingResult(
                embedding=result['embedding'],
                text=embedding_text,
                token_count=len(embedding_text.split()),
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Embedding failed: {str(e)}")

            return EmbeddingResult(
                embedding=[],
                text=chunk_content,
                token_count=len(chunk_content.split()),
                processing_time=processing_time,
                success=False,
                error=str(e)
            )

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[EmbeddingResult]:
        """
        Embed multiple chunks efficiently.

        Args:
            chunks: List of chunks with 'content' and optional metadata

        Returns:
            List of EmbeddingResults
        """
        results = []

        for chunk in chunks:
            content = chunk.get('content', '')
            metadata = {k: v for k, v in chunk.items() if k != 'content'}

            result = self.embed_chunk(content, metadata)
            results.append(result)

            # Simple rate limiting
            time.sleep(0.1)

        return results

    def _prepare_for_embedding(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Prepare content for embedding - Context7 inspired approach.

        This is where we optimize for legal document search:
        1. Add legal context
        2. Boost important legal terms
        3. Include citation information
        """
        context_parts = []

        # Add document context if available
        if 'citation' in metadata:
            context_parts.append(f"Rujukan: {metadata['citation']}")

        if 'keywords' in metadata:
            keywords = metadata['keywords'][:5]  # Top 5 keywords
            context_parts.append(f"Kata Kunci: {', '.join(keywords)}")

        # Legal domain context (simple detection)
        legal_context = self._detect_legal_domain(content)
        if legal_context:
            context_parts.append(f"Domain: {legal_context}")

        # Build final embedding text
        if context_parts:
            context_prefix = " | ".join(context_parts)
            embedding_text = f"{context_prefix}\n\n{content}"
        else:
            embedding_text = content

        # Simple legal term boosting (repeat important terms)
        embedding_text = self._boost_legal_terms(embedding_text)

        return embedding_text

    def _detect_legal_domain(self, text: str) -> Optional[str]:
        """Simple legal domain detection."""
        text_lower = text.lower()

        if any(term in text_lower for term in ['pidana', 'sanksi', 'hukuman', 'pelanggaran']):
            return "Hukum Pidana"
        elif any(term in text_lower for term in ['perdata', 'kontrak', 'perjanjian', 'ganti rugi']):
            return "Hukum Perdata"
        elif any(term in text_lower for term in ['administrasi', 'perizinan', 'pelayanan']):
            return "Hukum Administrasi"
        elif any(term in text_lower for term in ['konstitusi', 'pemerintahan', 'kekuasaan']):
            return "Hukum Tata Negara"

        return None

    def _boost_legal_terms(self, text: str) -> str:
        """Simple legal term boosting by repetition."""
        text_lower = text.lower()
        found_terms = []

        for term in self.legal_terms:
            if term in text_lower:
                found_terms.append(term)

        if found_terms:
            # Add legal terms at the end for emphasis
            boost_text = f" [{' '.join(found_terms)}]"
            return text + boost_text

        return text

    def embed_query(self, query: str) -> EmbeddingResult:
        """
        Embed a search query with legal optimization.

        Args:
            query: Search query text

        Returns:
            EmbeddingResult for the query
        """
        start_time = time.time()

        # Check if API key is available
        if not self.api_key:
            return EmbeddingResult(
                embedding=[],
                text=query,
                token_count=len(query.split()),
                processing_time=time.time() - start_time,
                success=False,
                error="No API key available"
            )

        try:
            # Optimize query for legal search
            optimized_query = self._optimize_query(query)

            result = genai.embed_content(
                model=self.model,
                content=optimized_query,
                task_type="retrieval_query"
            )

            processing_time = time.time() - start_time

            return EmbeddingResult(
                embedding=result['embedding'],
                text=optimized_query,
                token_count=len(optimized_query.split()),
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Query embedding failed: {str(e)}")

            return EmbeddingResult(
                embedding=[],
                text=query,
                token_count=len(query.split()),
                processing_time=processing_time,
                success=False,
                error=str(e)
            )

    def _optimize_query(self, query: str) -> str:
        """Simple query optimization for legal search."""
        # Add legal context to queries
        query_lower = query.lower()

        # Expand common abbreviations
        expansions = {
            'uu': 'undang-undang',
            'pp': 'peraturan pemerintah',
            'perpres': 'peraturan presiden',
            'permen': 'peraturan menteri'
        }

        expanded_query = query
        for abbrev, full in expansions.items():
            if abbrev in query_lower:
                expanded_query += f" {full}"

        # Add legal terms if detected
        boost_terms = []
        for term in self.legal_terms:
            if term in query_lower:
                boost_terms.append(term)

        if boost_terms:
            expanded_query += f" [{' '.join(boost_terms)}]"

        return expanded_query


# Integration helper functions
def create_embeddings_for_chunks(chunks: List[Dict[str, Any]], api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Simple integration function for existing pipeline.

    Args:
        chunks: List of chunks from simple chunker
        api_key: Optional Gemini API key

    Returns:
        List of chunks with embeddings added
    """
    embedder = SimpleEmbedding(api_key=api_key)

    # Extract content and metadata from chunks
    embedding_input = []
    for chunk in chunks:
        embedding_input.append({
            'content': chunk.get('content', ''),
            'citation': chunk.get('citation_path', ''),
            'keywords': chunk.get('semantic_keywords', [])
        })

    # Get embeddings
    embedding_results = embedder.embed_chunks(embedding_input)

    # Add embeddings back to chunks
    for i, (chunk, result) in enumerate(zip(chunks, embedding_results)):
        if result.success:
            chunk['embedding'] = result.embedding
            chunk['embedding_text'] = result.text
            chunk['embedding_tokens'] = result.token_count
            chunk['embedding_time'] = result.processing_time
        else:
            chunk['embedding'] = None
            chunk['embedding_error'] = result.error
            logger.warning(f"Embedding failed for chunk {i}: {result.error}")

    return chunks


def search_embeddings(query: str, document_embeddings: List[Dict[str, Any]], top_k: int = 5, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Simple semantic search using embeddings.

    Args:
        query: Search query
        document_embeddings: List of documents with embeddings
        top_k: Number of top results to return
        api_key: Optional Gemini API key

    Returns:
        List of top matching documents with scores
    """
    embedder = SimpleEmbedding(api_key=api_key)

    # Get query embedding
    query_result = embedder.embed_query(query)
    if not query_result.success:
        logger.error(f"Query embedding failed: {query_result.error}")
        return []

    query_embedding = query_result.embedding

    # Calculate similarities
    similarities = []
    for i, doc in enumerate(document_embeddings):
        if doc.get('embedding'):
            similarity = _cosine_similarity(query_embedding, doc['embedding'])
            similarities.append((similarity, i, doc))

    # Sort by similarity and return top results
    similarities.sort(key=lambda x: x[0], reverse=True)

    results = []
    for similarity, idx, doc in similarities[:top_k]:
        result = doc.copy()
        result['similarity_score'] = similarity
        result['rank'] = len(results) + 1
        results.append(result)

    return results


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import numpy as np

    a = np.array(vec1)
    b = np.array(vec2)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def get_simple_embedding_info() -> Dict[str, Any]:
    """Get info about this simple embedding service."""
    return {
        'name': 'SimpleEmbedding',
        'approach': 'KISS - Keep It Simple, Stupid',
        'lines_of_code': '~300 lines total',
        'replaces': [
            'adaptive_embedding_service.py (500+ lines)',
            'embedding_config.py (200+ lines)',
            'All embedding/* complex files (1000+ lines)'
        ],
        'features': [
            'Direct Gemini integration',
            'Context7-compatible content preparation',
            'Legal term boosting',
            'Simple query optimization',
            'Semantic search capability',
            'Rate limiting built-in'
        ],
        'complexity': 'LOW - Easy to understand and maintain',
        'performance': 'HIGH - Direct API calls, minimal overhead'
    }


# Example usage
if __name__ == "__main__":
    # Test the simple embedding service
    import os

    # Load .env file and get API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY in .env file or environment variable")
        print("Example: GEMINI_API_KEY=your_api_key_here")
        exit(1)

    # Test embedding
    embedder = SimpleEmbedding(api_key=api_key)

    sample_chunk = {
        'content': 'Pasal 1 mengatur tentang ketentuan umum dalam undang-undang ini.',
        'citation': 'Pasal 1 UU No. 1 Tahun 2024',
        'keywords': ['pasal', 'ketentuan', 'undang-undang']
    }

    print("=== SIMPLE EMBEDDING TEST ===")
    result = embedder.embed_chunk(sample_chunk['content'], sample_chunk)

    if result.success:
        print(f"✅ Embedding successful!")
        print(f"Text: {result.text[:100]}...")
        print(f"Embedding dimension: {len(result.embedding)}")
        print(f"Token count: {result.token_count}")
        print(f"Processing time: {result.processing_time:.3f}s")
    else:
        print(f"❌ Embedding failed: {result.error}")
