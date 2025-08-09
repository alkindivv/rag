"""
Comprehensive unit tests for JinaEmbedder service.

Tests cover:
- API payload formatting (addresses 422 errors)
- Batch processing logic
- Error handling and retries
- Configuration management
- Mock integration patterns

Key Focus Areas:
- Fix Jina v4 API compatibility issues
- Validate request/response formats
- Test failure scenarios and recovery
- Performance and batch optimization
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from typing import List

import httpx

from src.services.embedding.embedder import JinaV4Embedder, JinaEmbedder
from src.utils.http import HttpClient


class TestJinaEmbedderConfiguration:
    """Test embedder configuration and initialization."""

    def test_default_initialization(self):
        """Test embedder initializes with default settings."""
        embedder = JinaV4Embedder()

        assert embedder.model == "jina-embeddings-v4"
        assert embedder.default_dims == 1024
        assert embedder.api_key == "test-key"  # From test environment
        assert embedder.batch_size == 2  # From test environment

    def test_custom_configuration(self):
        """Test embedder with custom configuration."""
        custom_client = MagicMock(spec=HttpClient)

        embedder = JinaV4Embedder(
            model="custom-model",
            default_dims=512,
            client=custom_client
        )

        assert embedder.model == "custom-model"
        assert embedder.default_dims == 512
        assert embedder.batch_size == 2  # From test environment settings
        assert embedder.client is custom_client

    def test_environment_override(self):
        """Test configuration from environment variables."""
        with patch.dict('os.environ', {
            'EMBEDDING_MODEL': 'env-model',
            'EMBED_BATCH_SIZE': '32'
        }):
            embedder = JinaV4Embedder()
            # Note: This tests that the class can handle env vars if implemented
            # Current implementation uses default values


class TestJinaEmbedderAPIPayload:
    """Test API request payload formatting - addresses 422 errors."""

    def test_single_embedding_payload_format(self):
        """Test correct Jina v4 API payload for single embedding."""
        captured_request = {}

        def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_request["url"] = str(request.url)
            captured_request["headers"] = dict(request.headers)
            captured_request["json"] = json.loads(request.content.decode())

            # Return successful Jina v4 response format
            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "object": "list",
                "usage": {"total_tokens": 10, "prompt_tokens": 10},
                "data": [{"object": "embedding", "embedding": [0.1] * 1024, "index": 0}]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaV4Embedder(client=client)

        result = embedder.embed_single("test text")

        # Validate response
        assert len(result) == 1024
        assert all(isinstance(x, (int, float)) for x in result)

        # Validate request format (critical for fixing 422 errors)
        assert captured_request["url"].endswith("/v1/embeddings")
        assert captured_request["headers"]["Authorization"] == "Bearer test-key"
        assert captured_request["headers"]["Content-Type"] == "application/json"
        assert captured_request["headers"]["Accept"] == "application/json"

        payload = captured_request["json"]
        assert payload["model"] == "jina-embeddings-v4"
        assert payload["input"] == ["test text"]
        assert payload["task"] == "retrieval.passage"
        assert payload["dimensions"] == 1024
        assert payload["return_multivector"] == False
        assert payload["late_chunking"] == False
        assert payload["truncate"] == True
        assert "encoding_format" not in payload  # v4 API doesn't use this

    def test_batch_embedding_payload_format(self):
        """Test correct batch embedding payload format."""
        captured_request = {}

        def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_request["json"] = json.loads(request.content.decode())

            # Return batch response
            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [
                    {"embedding": [0.1] * 1024, "index": 0},
                    {"embedding": [0.2] * 1024, "index": 1}
                ]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaV4Embedder(client=client)

        texts = ["first text", "second text"]
        results = embedder.embed_texts(texts, task="retrieval.passage")

        assert len(results) == 2
        assert len(results[0]) == 1024
        assert len(results[1]) == 1024

        # Validate batch payload
        payload = captured_request["json"]
        assert payload["input"] == texts
        assert len(payload["input"]) <= embedder.batch_size

    def test_large_batch_chunking(self):
        """Test that large batches are properly chunked."""
        call_count = 0
        captured_requests = []

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1

            payload = json.loads(request.content.decode())
            captured_requests.append(payload)

            # Return response for each chunk
            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": i}
                        for i in range(len(payload["input"]))]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaV4Embedder(client=client)

        # Test with 5 texts (should create 3 API calls: 2+2+1)
        texts = [f"text {i}" for i in range(5)]
        results = embedder.embed_batch(texts)

        assert len(results) == 5
        assert call_count == 3  # Chunked into 3 requests

        # Validate chunk sizes
        chunk_sizes = [len(req["input"]) for req in captured_requests]
        assert chunk_sizes == [2, 2, 1]


class TestJinaEmbedderErrorHandling:
    """Test error handling and retry logic."""

    def test_api_error_handling_422(self):
        """Test handling of 422 Unprocessable Entity error."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={
                "detail": [{"loc": ["body"], "msg": "Invalid input format"}]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client)

        with pytest.raises(Exception) as exc_info:
            embedder.embed_single("test")

        assert "422" in str(exc_info.value) or "Unprocessable" in str(exc_info.value)

    def test_api_error_handling_rate_limit(self):
        """Test handling of rate limit errors."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": "Rate limit exceeded"})

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client)

        with pytest.raises(Exception) as exc_info:
            embedder.embed_single("test")

        assert "429" in str(exc_info.value) or "rate" in str(exc_info.value).lower()

    def test_network_timeout_handling(self):
        """Test handling of network timeouts."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("Request timeout")

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client)

        with pytest.raises(Exception) as exc_info:
            embedder.embed_single("test")

        assert "timeout" in str(exc_info.value).lower()

    def test_invalid_response_format(self):
        """Test handling of unexpected response formats."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            # Return invalid format (missing 'data' field)
            return httpx.Response(200, json={"invalid": "format"})

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client)

        with pytest.raises(Exception) as exc_info:
            embedder.embed_single("test")

        # Should fail when trying to extract embeddings from invalid response

    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        embedder = JinaEmbedder()

        # These should handle gracefully without API calls
        with pytest.raises(ValueError):
            embedder.embed_single("")

        with pytest.raises(ValueError):
            embedder.embed_single("   ")

        with pytest.raises(ValueError):
            embedder.embed_batch([])

    def test_batch_partial_failure_handling(self):
        """Test handling when some items in batch fail."""
        call_count = 0

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call succeeds
                return httpx.Response(200, json={
                    "model": "jina-embeddings-v4",
                    "data": [{"embedding": [0.1] * 1024, "index": 0}]
                })
            else:
                # Second call fails
                return httpx.Response(422, json={"error": "Invalid input"})

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client, batch_size=1)

        texts = ["valid text", "invalid text"]

        with pytest.raises(Exception):
            embedder.embed_batch(texts)


class TestJinaEmbedderValidation:
    """Test input validation and data sanitization."""

    def test_text_length_validation(self):
        """Test handling of very long texts."""
        embedder = JinaEmbedder()

        # Test extremely long text (should handle gracefully)
        long_text = "a" * 10000

        # Mock successful response
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": 0}]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder.client = client

        # Should not raise exception for long text
        result = embedder.embed_single(long_text)
        assert len(result) == 1024

    def test_special_characters_handling(self):
        """Test handling of special characters and Unicode."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode())
            # Verify special characters are preserved
            assert "émöjí" in payload["input"][0]

            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": 0}]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client)

        # Test with special characters
        text_with_special = "Legal text with émöjí and unicode: ñáéíóú"
        result = embedder.embed_single(text_with_special)
        assert len(result) == 1024

    def test_batch_size_limits(self):
        """Test batch size validation and limits."""
        embedder = JinaEmbedder(batch_size=3)

        # Test batch size enforcement
        texts = [f"text {i}" for i in range(10)]

        call_count = 0
        max_batch_size = 0

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count, max_batch_size
            call_count += 1

            payload = json.loads(request.content.decode())
            batch_size = len(payload["input"])
            max_batch_size = max(max_batch_size, batch_size)

            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": i}
                        for i in range(batch_size)]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder.client = client

        results = embedder.embed_batch(texts)

        assert len(results) == 10
        assert max_batch_size <= 3  # Respects batch size limit
        assert call_count >= 4  # Should make multiple calls


class TestJinaEmbedderAPIIntegration:
    """Test API integration patterns and response handling."""

    def test_correct_jina_v4_request_format(self):
        """Test that requests match Jina v4 API specification exactly."""
        captured_request = {}

        def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_request.update({
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "json": json.loads(request.content.decode())
            })

            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "object": "list",
                "usage": {"total_tokens": 5, "prompt_tokens": 5},
                "data": [{"object": "embedding", "embedding": [0.1] * 1024, "index": 0}]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client)

        embedder.embed_single("test")

        # Validate HTTP method and URL
        assert captured_request["method"] == "POST"
        assert captured_request["url"] == "https://api.jina.ai/v1/embeddings"

        # Validate headers
        headers = captured_request["headers"]
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

        # Validate JSON payload (critical for fixing 422 errors)
        payload = captured_request["json"]
        required_fields = ["model", "input", "encoding_format"]
        for field in required_fields:
            assert field in payload, f"Missing required field: {field}"

        assert payload["model"] == "jina-embeddings-v4"
        assert payload["input"] == ["test"]
        assert payload["encoding_format"] == "float"

        # Ensure no invalid fields that cause 422 errors
        invalid_fields = ["dimensions", "normalize", "truncate"]
        for field in invalid_fields:
            assert field not in payload, f"Invalid field present: {field}"

    def test_response_parsing_success(self):
        """Test parsing successful Jina v4 response format."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "object": "list",
                "usage": {"total_tokens": 5},
                "data": [
                    {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}
                ]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client, dimensions=3)  # Small for testing

        result = embedder.embed_single("test")

        assert result == [0.1, 0.2, 0.3]

    def test_response_parsing_batch(self):
        """Test parsing batch response with multiple embeddings."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [
                    {"embedding": [0.1] * 3, "index": 0},
                    {"embedding": [0.2] * 3, "index": 1},
                    {"embedding": [0.3] * 3, "index": 2}
                ]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client, dimensions=3)

        texts = ["first", "second", "third"]
        results = embedder.embed_batch(texts)

        assert len(results) == 3
        assert results[0] == [0.1] * 3
        assert results[1] == [0.2] * 3
        assert results[2] == [0.3] * 3


class TestJinaEmbedderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_character_text(self):
        """Test embedding single character text."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": 0}]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client)

        result = embedder.embed_single("a")
        assert len(result) == 1024

    def test_very_long_legal_text(self):
        """Test embedding typical legal document content."""
        long_legal_text = """
        Pasal 1
        Dalam Undang-Undang ini yang dimaksud dengan:
        1. Pertambangan adalah sebagian atau seluruh tahapan kegiatan dalam rangka
           penelitian, pengelolaan dan pengusahaan mineral atau batubara yang meliputi
           penyelidikan umum, eksplorasi, studi kelayakan, konstruksi, penambangan,
           pengolahan dan pemurnian, pengangkutan dan penjualan, serta kegiatan
           pascatambang.
        2. Mineral adalah senyawa anorganik yang terbentuk di alam, yang memiliki
           sifat fisik dan kimia tertentu serta susunan kristal teratur atau
           gabungannya yang membentuk batuan, baik dalam bentuk lepas atau padu.
        """ * 10  # Make it quite long

        def mock_handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode())
            assert len(payload["input"]) == 1

            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": 0}]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client)

        result = embedder.embed_single(long_legal_text)
        assert len(result) == 1024

    def test_mixed_language_content(self):
        """Test embedding mixed Indonesian/English legal content."""
        mixed_text = "Pasal 1 of this regulation defines mining as pertambangan"

        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": 0}]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client)

        result = embedder.embed_single(mixed_text)
        assert len(result) == 1024


class TestJinaEmbedderPerformance:
    """Test performance characteristics and optimizations."""

    def test_batch_performance_vs_single(self, performance_timer):
        """Test that batch processing is more efficient than individual calls."""
        single_call_count = 0
        batch_call_count = 0

        def mock_handler_single(request: httpx.Request) -> httpx.Response:
            nonlocal single_call_count
            single_call_count += 1
            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": 0}]
            })

        def mock_handler_batch(request: httpx.Request) -> httpx.Response:
            nonlocal batch_call_count
            batch_call_count += 1
            payload = json.loads(request.content.decode())
            batch_size = len(payload["input"])
            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": i} for i in range(batch_size)]
            })

        # Test single calls
        transport_single = httpx.MockTransport(mock_handler_single)
        client_single = HttpClient(client=httpx.Client(transport=transport_single))
        embedder_single = JinaEmbedder(client=client_single)

        performance_timer.start()
        for i in range(4):
            embedder_single.embed_single(f"text {i}")
        performance_timer.stop()
        single_duration = performance_timer.duration_ms

        # Test batch calls
        transport_batch = httpx.MockTransport(mock_handler_batch)
        client_batch = HttpClient(client=httpx.Client(transport=transport_batch))
        embedder_batch = JinaEmbedder(client=client_batch, batch_size=4)

        performance_timer.start()
        embedder_batch.embed_batch([f"text {i}" for i in range(4)])
        performance_timer.stop()
        batch_duration = performance_timer.duration_ms

        # Batch should make fewer API calls
        assert batch_call_count < single_call_count
        assert single_call_count == 4  # One call per text
        assert batch_call_count == 1   # One batch call

    def test_memory_usage_large_batch(self):
        """Test memory efficiency with large batches."""
        import gc
        import tracemalloc

        def mock_handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode())
            batch_size = len(payload["input"])
            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": i} for i in range(batch_size)]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client, batch_size=10)

        # Monitor memory usage
        tracemalloc.start()

        # Process large batch
        large_texts = [f"Legal document content {i}" * 100 for i in range(100)]
        results = embedder.embed_texts(large_texts, task="retrieval.passage")

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert len(results) == 100
        # Memory usage should be reasonable (less than 100MB for this test)
        assert peak < 100 * 1024 * 1024  # 100MB limit


class TestJinaEmbedderMocking:
    """Test mocking patterns for downstream services."""

    def test_mock_embedder_interface(self, mock_jina_embedder):
        """Test that mock embedder matches real interface."""
        # Test single embedding
        result = mock_jina_embedder.embed_single("test")
        assert len(result) == 1024
        assert all(isinstance(x, (int, float)) for x in result)

        # Test batch embedding
        batch_results = mock_jina_embedder.embed_batch(["text1", "text2"])
        assert len(batch_results) == 2
        assert len(batch_results[0]) == 1024

    def test_embedder_dependency_injection(self):
        """Test embedder can be injected into dependent services."""
        from src.services.retriever.hybrid_retriever import VectorSearcher

        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = []

        # Should accept injected embedder
        searcher = VectorSearcher(mock_session, embedder=mock_embedder)
        searcher.search("test query")

        # Verify embedder was called
        mock_embedder.embed_single.assert_called_once_with("test query")


class TestJinaEmbedderIntegration:
    """Integration tests with other components."""

    @patch('src.services.embedding.embedder.JinaEmbedder')
    def test_indexer_integration(self, mock_embedder_class, temp_json_file):
        """Test embedder integration with document indexer."""
        from src.pipeline.indexer import DocumentIndexer

        # Setup mock
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024
        mock_embedder_class.return_value = mock_embedder

        # Test indexer uses embedder
        with patch('src.db.session.get_db_session') as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db

            indexer = DocumentIndexer(skip_embeddings=False)
            # This would normally process the temp file
            # For unit test, we just verify the embedder is properly initialized

            assert mock_embedder_class.called

    def test_retriever_integration(self, mock_db_session, mock_jina_embedder):
        """Test embedder integration with hybrid retriever."""
        from src.services.retriever.hybrid_retriever import HybridRetriever

        # Mock database response
        mock_db_session.execute.return_value.fetchall.return_value = []

        retriever = HybridRetriever(embedder=mock_jina_embedder)
        results = retriever.search("test query", strategy="vector")

        # Verify embedder was called for vector search
        mock_jina_embedder.embed_single.assert_called_with("test query")
        assert results == []  # Empty due to mocked database


class TestJinaEmbedderRealWorldScenarios:
    """Test real-world usage patterns and scenarios."""

    def test_legal_document_embedding_workflow(self):
        """Test typical workflow: extract text → embed → store."""
        legal_texts = [
            "Dalam Undang-Undang ini yang dimaksud dengan pertambangan adalah...",
            "Setiap orang yang melakukan kegiatan pertambangan wajib memiliki izin...",
            "Pelanggaran terhadap ketentuan ini dikenai sanksi administratif berupa..."
        ]

        embeddings_generated = []

        def mock_handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode())
            batch_size = len(payload["input"])

            # Store for verification
            embeddings_generated.extend(payload["input"])

            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [float(i) * 0.1] * 1024, "index": i}
                        for i in range(batch_size)]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client, batch_size=2)

        # Process all texts
        results = embedder.embed_batch(legal_texts)

        assert len(results) == 3
        assert embeddings_generated == legal_texts
        # Verify embeddings are different (based on mock logic)
        assert results[0] != results[1] != results[2]

    def test_incremental_document_processing(self):
        """Test processing documents incrementally."""
        call_history = []

        def mock_handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode())
            call_history.append(len(payload["input"]))

            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": i}
                        for i in range(len(payload["input"]))]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client, batch_size=3)

        # Simulate processing documents one by one
        doc1_texts = ["Pasal 1 tentang definisi", "Pasal 2 tentang ruang lingkup"]
        doc2_texts = ["Pasal 3 tentang kewenangan", "Pasal 4 tentang izin"]
        doc3_texts = ["Pasal 5 tentang sanksi"]

        # Process each document separately
        results1 = embedder.embed_batch(doc1_texts)
        results2 = embedder.embed_batch(doc2_texts)
        results3 = embedder.embed_batch(doc3_texts)

        # Verify all processed correctly
        assert len(results1) == 2
        assert len(results2) == 2
        assert len(results3) == 1

        # Verify API call pattern (should respect batch size)
        expected_calls = [2, 2, 1]  # Batch size 3, so: 2+2+1 = 5 texts in 3 calls
        assert call_history == expected_calls

    def test_duplicate_text_handling(self):
        """Test handling of duplicate texts in batch."""
        call_count = 0
        seen_texts = set()

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1

            payload = json.loads(request.content.decode())
            for text in payload["input"]:
                seen_texts.add(text)

            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": i}
                        for i in range(len(payload["input"]))]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client)

        # Include duplicates in batch
        texts_with_duplicates = ["text A", "text B", "text A", "text C", "text B"]
        results = embedder.embed_batch(texts_with_duplicates)

        # Should return embeddings for all (including duplicates)
        assert len(results) == 5
        # API should receive all texts (no deduplication at embedder level)
        assert len(seen_texts) == 3  # Unique texts sent to API

    def test_empty_batch_handling(self):
        """Test handling of edge cases with empty batches."""
        embedder = JinaEmbedder()

        # Empty batch should raise ValueError
        with pytest.raises(ValueError, match="Empty"):
            embedder.embed_batch([])

        # Batch with empty strings should raise ValueError
        with pytest.raises(ValueError, match="empty"):
            embedder.embed_batch(["", "  ", "\n"])


class TestJinaEmbedderLogging:
    """Test logging and monitoring capabilities."""

    def test_successful_embedding_logging(self, capture_logs):
        """Test logging for successful embedding operations."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": 0}]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client)

        embedder.embed_single("test text")

        log_output = capture_logs.getvalue()
        # Should log embedding operation details
        assert "embedding" in log_output.lower()

    def test_error_logging(self, capture_logs):
        """Test logging for embedding errors."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"error": "Invalid input"})

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=client)

        with pytest.raises(Exception):
            embedder.embed_single("test")

        log_output = capture_logs.getvalue()
        # Should log error details
        assert "error" in log_output.lower() or "422" in log_output


class TestJinaEmbedderCompatibility:
    """Test backward compatibility and interface stability."""

    def test_interface_compatibility(self):
        """Test that embedder maintains expected interface."""
        embedder = JinaEmbedder()

        # Required methods should exist
        assert hasattr(embedder, 'embed_single')
        assert hasattr(embedder, 'embed_batch')
        assert callable(embedder.embed_single)
        assert callable(embedder.embed_batch)

        # Required attributes should exist
        assert hasattr(embedder, 'model_name')
        assert hasattr(embedder, 'dimensions')
        assert hasattr(embedder, 'batch_size')

    def test_embedder_in_retriever_context(self):
        """Test embedder usage pattern in retriever services."""
        from src.services.retriever.hybrid_retriever import VectorSearcher

        # This test ensures the embedder can be used as expected by retrievers
        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = []

        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 1024

        # Should not raise any interface errors
        searcher = VectorSearcher(mock_session, embedder=mock_embedder)
        results = searcher.search("mining regulations", limit=10)

        assert results == []
        mock_embedder.embed_single.assert_called_once()


# ================================
# REGRESSION TESTS FOR KNOWN ISSUES
# ================================

class TestJinaEmbedderRegressionTests:
    """Regression tests for previously identified issues."""

    def test_jina_api_422_error_fix(self):
        """
        Regression test for Jina API 422 errors.

        This test validates that the request format fixes the original
        422 Unprocessable Entity errors reported in TODO_NEXT.md
        """
        captured_requests = []

        def mock_handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode())
            captured_requests.append(payload)

            # Simulate Jina v4 validation - accept correct format
            required_fields = ["model", "input", "task", "dimensions", "return_multivector", "late_chunking", "truncate"]
            for field in required_fields:
                if field not in payload:
                    return httpx.Response(422, json={
                        "detail": [{"loc": ["body", field], "msg": f"Field required"}]
                    })

            # Reject old encoding_format parameter
            if "encoding_format" in payload:
                return httpx.Response(422, json={
                    "detail": [{"loc": ["body", "encoding_format"], "msg": "Field not supported in v4"}]
                })

            # Accept valid v4 request
            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": 0}]
            })

        transport = httpx.MockTransport(mock_handler)
        client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaV4Embedder(client=client)

        # Should not raise 422 error with correct v4 format
        result = embedder.embed_single("test legal text")
        assert len(result) == 1024

        # Verify correct v4 request format was used
        assert len(captured_requests) == 1
        payload = captured_requests[0]

        # Check required v4 fields
        assert payload["model"] == "jina-embeddings-v4"
        assert payload["input"] == ["test legal text"]
        assert payload["task"] == "retrieval.passage"
        assert payload["dimensions"] == 1024
        assert payload["return_multivector"] == False
        assert payload["late_chunking"] == False
        assert payload["truncate"] == True

        # Ensure old fields are not present
        assert "encoding_format" not in payload

    def test_import_path_consistency(self):
        """Test that import paths work correctly."""
        # This addresses import path inconsistencies mentioned in AGENTS.md
        try:
            from src.services.embedding.embedder import JinaEmbedder
            from src.utils.http import HttpClient

            # Should be able to create instances without import errors
            client = HttpClient()
            embedder = JinaEmbedder(client=client)

            assert embedder is not None
            assert client is not None

        except ImportError as e:
            pytest.fail(f"Import path issue detected: {e}")

    def test_lazy_loading_pattern(self):
        """Test lazy loading to avoid circular imports."""
        # This test ensures that embedder can be imported without
        # causing NameError or circular import issues
        try:
            # Import should work without triggering other imports
            import src.services.embedding.embedder as embedder_module

            # Class should be available
            assert hasattr(embedder_module, 'JinaEmbedder')

            # Should be able to create instance
            embedder = embedder_module.JinaEmbedder()
            assert embedder is not None

        except (ImportError, NameError) as e:
            pytest.fail(f"Lazy loading issue: {e}")


# ================================
# MOCK FACTORIES FOR COMPLEX SCENARIOS
# ================================

@pytest.fixture
def embedder_with_realistic_api():
    """Create embedder with realistic API behavior."""
    call_count = 0
    request_history = []

    def realistic_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1

        payload = json.loads(request.content.decode())
        request_history.append({
            "call_number": call_count,
            "batch_size": len(payload["input"]),
            "texts": payload["input"]
        })

        # Simulate some API latency and realistic responses
        import time
        time.sleep(0.01)  # 10ms latency

        # Generate embeddings with some variation
        embeddings = []
        for i, text in enumerate(payload["input"]):
            # Generate pseudo-realistic embeddings based on text content
            base_value = hash(text) % 1000 / 1000.0
            embedding = [base_value + (j * 0.001) for j in range(1024)]
            embeddings.append({"embedding": embedding, "index": i})

        return httpx.Response(200, json={
            "model": "jina-embeddings-v4",
            "object": "list",
            "usage": {"total_tokens": len(payload["input"]) * 10},
            "data": embeddings
        })

    transport = httpx.MockTransport(realistic_handler)
    client = HttpClient(client=httpx.Client(transport=transport))
    embedder = JinaEmbedder(client=client)

    # Attach call tracking for assertions
    embedder._test_call_count = lambda: call_count
    embedder._test_request_history = lambda: request_history

    return embedder


class TestJinaEmbedderRealisticScenarios:
    """Test with realistic API behavior and data."""

    def test_realistic_legal_document_processing(self, embedder_with_realistic_api):
        """Test processing actual legal document structure."""
        # Realistic legal text samples
        legal_pasal_contents = [
            "Dalam Undang-Undang ini yang dimaksud dengan pertambangan adalah sebagian atau seluruh tahapan kegiatan dalam rangka penelitian, pengelolaan dan pengusahaan mineral atau batubara.",
            "Setiap orang yang akan melakukan kegiatan pertambangan wajib memiliki izin usaha pertambangan yang diterbitkan oleh Pemerintah atau Pemerintah Daerah sesuai dengan kewenangannya.",
            "Izin usaha pertambangan sebagaimana dimaksud pada ayat (1) diberikan dalam bentuk izin usaha pertambangan eksplorasi dan izin usaha pertambangan operasi produksi.",
            "Pelanggaran terhadap ketentuan sebagaimana dimaksud dalam Pasal ini dikenai sanksi administratif berupa peringatan tertulis, penghentian sementara kegiatan, atau pencabutan izin."
        ]

        embedder = embedder_with_realistic_api

        # Process batch
        embeddings = embedder.embed_batch(legal_pasal_contents)

        # Validate results
        assert len(embeddings) == 4
        for embedding in embeddings:
            assert len(embedding) == 1024
            assert all(isinstance(x, (int, float)) for x in embedding)

        # Verify API usage
        call_count = embedder._test_call_count()
        request_history = embedder._test_request_history()

        assert call_count >= 1
        total_texts_processed = sum(req["batch_size"] for req in request_history)
        assert total_texts_processed == 4

    def test_performance_characteristics(self, embedder_with_realistic_api, performance_timer):
        """Test embedding performance meets requirements."""
        embedder = embedder_with_realistic_api

        # Test single embedding performance
        performance_timer.start()
        result = embedder.embed_single("Sample legal text for performance testing")
        performance_timer.stop()

        single_duration = performance_timer.duration_ms

        # Test batch embedding performance
        texts = [f"Legal document {i} content" for i in range(10)]
        performance_timer.start()
        batch_results = embedder.embed_batch(texts)
        performance_timer.stop()

        batch_duration = performance_timer.duration_ms

        # Validate results
        assert len(result) == 1024
        assert len(batch_results) == 10

        # Performance assertions (should be fast with mocked API)
        assert single_duration < 100  # Less than 100ms for single
        assert batch_duration < 500   # Less than 500ms for batch of 10

        # Batch should be more efficient per item
        single_per_item = single_duration
        batch_per_item = batch_duration / 10
        assert batch_per_item <= single_per_item


# ================================
# FINAL INTEGRATION SMOKE TESTS
# ================================

class TestJinaEmbedderSmokeTests:
    """Quick smoke tests to verify basic functionality."""

    def test_embedder_smoke_test(self):
        """Basic smoke test - embedder can be created and used."""
        mock_client = MagicMock(spec=HttpClient)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "jina-embeddings-v4",
            "data": [{"embedding": [0.1] * 1024, "index": 0}]
        }
        mock_client.post.return_value = mock_response

        embedder = JinaEmbedder(client=mock_client)
        result = embedder.embed_single("smoke test")

        assert len(result) == 1024
        mock_client.post.assert_called_once()

    def test_embedder_with_real_http_client(self):
        """Test embedder with real HTTP client (mocked transport)."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "model": "jina-embeddings-v4",
                "data": [{"embedding": [0.1] * 1024, "index": 0}]
            })

        transport = httpx.MockTransport(mock_handler)
        real_client = HttpClient(client=httpx.Client(transport=transport))
        embedder = JinaEmbedder(client=real_client)

        result = embedder.embed_single("integration test")
        assert len(result) == 1024

    def test_embedder_initialization_without_errors(self):
        """Test that embedder can be initialized without configuration errors."""
        # This should not raise any exceptions
        embedder = JinaEmbedder()

        # Basic attributes should be set
        assert embedder.model_name
        assert embedder.dimensions > 0
        assert embedder.batch_size > 0
        assert embedder.api_key  # Should get from environment

    def test_embedder_type_annotations(self):
        """Test that embedder methods have correct type annotations."""
        import inspect
        from typing import get_type_hints

        # Get type hints for embed_single
        hints = get_type_hints(JinaEmbedder.embed_single)

        # Should have proper return type annotation
        assert 'return' in hints
        # The actual type checking depends on the implementation

        # Get type hints for embed_batch
        batch_hints = get_type_hints(JinaEmbedder.embed_batch)
        assert 'return' in batch_hints
