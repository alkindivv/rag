"""
Integration tests for embedding services in Legal RAG System.

These tests validate the complete embedding workflow including:
- JinaV4Embedder integration with real API calls (when configured)
- Embedding dimension validation and error handling
- Batch processing of multiple texts
- Integration with indexer and search components

Test Coverage:
- Embedding service instantiation and configuration
- Text embedding generation and validation
- Multi-vector embedding support
- Error handling for API failures
- Integration with document processing pipeline
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from src.services.embedding.embedder import JinaV4Embedder
from src.config.settings import settings


class TestEmbeddingServiceIntegration:
    """Test embedding service integration with the Legal RAG system."""

    def test_jina_v4_embedder_instantiation(self):
        """Test JinaV4Embedder can be instantiated with default settings."""
        # Test with default settings
        embedder = JinaV4Embedder()
        assert embedder is not None
        assert embedder.model == settings.EMBEDDING_MODEL
        assert embedder.dims == settings.EMBEDDING_DIM
        
    def test_jina_v4_embedder_with_custom_config(self):
        """Test JinaV4Embedder can be instantiated with custom configuration."""
        # Test with custom configuration
        embedder = JinaV4Embedder(
            api_key="test-key",
            model="jina-embeddings-test",
            dims=512,
            default_task="retrieval.passage"
        )
        assert embedder.api_key == "test-key"
        assert embedder.model == "jina-embeddings-test"
        assert embedder.dims == 512
        assert embedder.default_task == "retrieval.passage"
        
    def test_embed_single_text(self, mock_jina_embedder):
        """Test embedding a single text passage."""
        test_text = "Pasal 1 UU No. 1 Tahun 2025 tentang Test Document"
        
        # Mock the HTTP client response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 1024}]
        }
        mock_jina_embedder.client.post.return_value = mock_response
        
        # Test embedding generation
        embedding = mock_jina_embedder.embed_single(test_text)
        assert len(embedding) == 1024
        assert all(isinstance(x, float) for x in embedding)
        
        # Verify API call was made
        mock_jina_embedder.client.post.assert_called_once()
        
    def test_embed_multiple_texts(self, mock_jina_embedder):
        """Test embedding multiple text passages with batching."""
        test_texts = [
            "Pasal 1 UU No. 1 Tahun 2025 tentang Test Document",
            "Pasal 2 UU No. 1 Tahun 2025 tentang Test Document",
            "Pasal 3 UU No. 1 Tahun 2025 tentang Test Document"
        ]
        
        # Mock the HTTP client response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 1024} for _ in test_texts]
        }
        mock_jina_embedder.client.post.return_value = mock_response
        
        # Test batch embedding generation
        embeddings = mock_jina_embedder.embed_texts(test_texts)
        assert len(embeddings) == len(test_texts)
        for embedding in embeddings:
            assert len(embedding) == 1024
            assert all(isinstance(x, float) for x in embedding)
            
        # Verify API call was made
        mock_jina_embedder.client.post.assert_called_once()
        
    def test_embedding_dimension_validation(self, mock_jina_embedder):
        """Test that embedding dimension validation works correctly."""
        test_text = "Test passage for dimension validation"
        
        # Mock response with wrong dimensions
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 512}]  # Wrong dimension
        }
        mock_jina_embedder.client.post.return_value = mock_response
        
        # Should raise ValueError for dimension mismatch
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            mock_jina_embedder.embed_single(test_text)
            
    def test_embedding_service_error_handling(self, mock_jina_embedder):
        """Test error handling when embedding service fails."""
        test_text = "Test passage for error handling"
        
        # Mock API error response
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_jina_embedder.client.post.return_value = mock_response
        
        # Should handle API errors gracefully
        with pytest.raises(Exception, match="API Error"):
            mock_jina_embedder.embed_single(test_text)
            
    def test_embedding_with_multivector_support(self):
        """Test embedding service with multivector support enabled."""
        with patch.dict(os.environ, {"JINA_API_KEY": "test-key"}):
            embedder = JinaV4Embedder(return_multivector=True)
            assert embedder.return_multivector is True
            
            # Mock response with multivector format
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "data": [
                    {
                        "embedding": {
                            "values": [0.1] * 1024,
                            "indices": list(range(1024))
                        }
                    }
                ]
            }
            
            with patch.object(embedder.client, 'post', return_value=mock_response):
                embedding = embedder.embed_single("Test multivector passage")
                assert len(embedding) == 1024
                assert all(isinstance(x, float) for x in embedding)
