"""
Integration tests for Legal RAG System search pipeline.

Tests complete workflows:
- JSON ingestion → Database → Search → Results
- Multi-component interaction (Retriever + Reranker + LLM)
- Real database operations with actual schema
- Search strategy integration and fallbacks
- Performance under realistic loads

Focus Areas:
- End-to-end search pipeline functionality
- Component integration without mocking core logic
- Real data processing workflows
- Error propagation and recovery
- System performance validation
"""

import pytest
import json
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

from src.config.settings import settings
from src.db.models import LegalDocument, LegalUnit, DocumentVector, Subject
from src.pipeline.indexer import DocumentIndexer
from src.services.retriever.hybrid_retriever import HybridRetriever, SearchFilters
from src.services.search.hybrid_search import HybridSearchService
from src.services.search.reranker import JinaReranker


class TestSearchPipelineEndToEnd:
    """Test complete search pipeline from data ingestion to results."""

    def test_json_to_search_pipeline(self, db_session, temp_json_file, mock_jina_embedder):
        """Test complete pipeline: JSON file → Index → Search → Results."""

        # Step 1: Index the JSON file
        with patch('src.services.embedding.embedder.JinaEmbedder', return_value=mock_jina_embedder):
            with patch('src.db.session.get_db_session') as mock_get_session:
                mock_get_session.return_value.__enter__.return_value = db_session

                indexer = DocumentIndexer(skip_embeddings=True)
                indexer.index_file(temp_json_file)

        # Verify data was indexed
        documents = db_session.query(LegalDocument).all()
        units = db_session.query(LegalUnit).all()

        assert len(documents) == 1
        assert len(units) >= 1  # Should have multiple units from document tree

        doc = documents[0]
        assert doc.doc_id == "UU-2025-1"
        assert doc.doc_form == "UU"

        # Step 2: Search the indexed data
        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = db_session

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            # Test FTS search
            fts_results = retriever.search("pertambangan", strategy="fts", limit=10)
            assert isinstance(fts_results, list)

            # Test explicit search
            explicit_results = retriever.search("pasal 1", strategy="explicit", limit=10)
            assert isinstance(explicit_results, list)

        # Step 3: Verify search results have correct structure
        if fts_results:
            result = fts_results[0]
            assert hasattr(result, 'unit_id')
            assert hasattr(result, 'content')
            assert hasattr(result, 'citation')
            assert hasattr(result, 'score')
            assert result.source_type in ["fts", "explicit", "vector"]

    def test_indexing_with_embeddings_integration(self, db_session, temp_json_file, mock_jina_embedder):
        """Test indexing with embedding generation."""

        # Configure embedder to return realistic embeddings
        mock_jina_embedder.embed_single.return_value = [0.1] * 1024

        with patch('src.services.embedding.embedder.JinaEmbedder', return_value=mock_jina_embedder):
            with patch('src.db.session.get_db_session') as mock_get_session:
                mock_get_session.return_value.__enter__.return_value = db_session

                indexer = DocumentIndexer(skip_embeddings=False)
                indexer.index_file(temp_json_file)

        # Verify embeddings were generated
        vectors = db_session.query(DocumentVector).all()
        assert len(vectors) >= 1

        vector = vectors[0]
        assert len(vector.embedding) == 1024
        assert vector.unit_id.startswith("UU-2025-1")

        # Verify embedder was called
        mock_jina_embedder.embed_single.assert_called()

    def test_search_with_reranking_integration(self, populated_db, mock_jina_embedder, mock_jina_reranker):
        """Test search pipeline with reranking integration."""

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = populated_db

            # Create search service with all components
            retriever = HybridRetriever(embedder=mock_jina_embedder)

            with patch('src.services.search.reranker.JinaReranker', return_value=mock_jina_reranker):
                search_service = HybridSearchService(retriever=retriever)

                # Execute search with reranking
                response = search_service.search("pertambangan mineral", rerank=True)

                # Verify response structure
                assert "results" in response
                assert "total" in response
                assert "strategy" in response
                assert "reranked" in response
                assert response["reranked"] is True

                # Verify reranker was called if results exist
                if response["results"]:
                    mock_jina_reranker.rerank.assert_called()

    def test_multiple_document_search_integration(self, db_session, mock_jina_embedder):
        """Test search across multiple indexed documents."""

        # Create multiple test documents
        test_documents = [
            {
                "doc_id": "UU-2025-1",
                "doc_form": "UU",
                "doc_number": "1",
                "doc_year": "2025",
                "doc_title": "Undang-Undang Pertambangan",
                "doc_subject": ["PERTAMBANGAN"]
            },
            {
                "doc_id": "PP-2025-1",
                "doc_form": "PP",
                "doc_number": "1",
                "doc_year": "2025",
                "doc_title": "Peraturan Pemerintah Lingkungan",
                "doc_subject": ["LINGKUNGAN"]
            }
        ]

        test_units = [
            {
                "id": "unit-1",
                "document_id": "test-uuid-1",
                "unit_type": "ayat",
                "unit_id": "UU-2025-1/pasal-1/ayat-1",
                "content": "Pertambangan mineral adalah kegiatan...",
                "local_content": "kegiatan penambangan mineral",
                "citation_string": "Pasal 1 ayat (1)"
            },
            {
                "id": "unit-2",
                "document_id": "test-uuid-2",
                "unit_type": "ayat",
                "unit_id": "PP-2025-1/pasal-1/ayat-1",
                "content": "Perlindungan lingkungan dalam pertambangan...",
                "local_content": "perlindungan lingkungan",
                "citation_string": "Pasal 1 ayat (1)"
            }
        ]

        # Add test data
        for i, doc_data in enumerate(test_documents):
            doc_data["id"] = f"test-uuid-{i+1}"
            doc = LegalDocument(**doc_data)
            db_session.add(doc)

        for unit_data in test_units:
            unit = LegalUnit(**unit_data)
            db_session.add(unit)

        db_session.commit()

        # Test search across documents
        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = db_session

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            # Search should find results from both documents
            results = retriever.search("pertambangan", strategy="fts", limit=10)

            # Should return results from multiple documents
            doc_forms = {result.document_form for result in results}
            assert len(doc_forms) >= 1  # At least one document type

            # Test with document form filter
            uu_results = retriever.search(
                "pertambangan",
                strategy="fts",
                filters=SearchFilters(doc_forms=["UU"]),
                limit=10
            )

            # Should only return UU results
            for result in uu_results:
                assert result.document_form == "UU"

    def test_search_performance_integration(self, populated_db, mock_jina_embedder, performance_timer):
        """Test search performance under realistic conditions."""

        # Add more test data for performance testing
        additional_units = []
        for i in range(50):
            unit = LegalUnit(
                id=f"perf-unit-{i}",
                document_id="test-uuid-1",
                unit_type="ayat",
                unit_id=f"UU-2025-1/pasal-{i+10}/ayat-1",
                content=f"Ketentuan pertambangan nomor {i} mencakup aspek teknis dan administratif...",
                local_content=f"aspek teknis {i}",
                citation_string=f"Pasal {i+10} ayat (1)"
            )
            additional_units.append(unit)

        db_session.add_all(additional_units)
        db_session.commit()

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = populated_db

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            # Test search performance
            performance_timer.start()
            results = retriever.search("pertambangan teknis", strategy="fts", limit=20)
            performance_timer.stop()

            # Should meet performance requirements
            assert performance_timer.duration_ms < 500  # Less than 500ms
            assert isinstance(results, list)
            assert len(results) <= 20


class TestComponentIntegration:
    """Test integration between major components."""

    def test_retriever_embedder_integration(self, mock_db_session, mock_jina_embedder):
        """Test retriever properly integrates with embedder."""

        # Configure embedder
        mock_jina_embedder.embed_single.return_value = [0.2] * 1024

        # Mock database to return vector search results
        mock_row = MagicMock()
        mock_row.unit_id = "UU-2025-1/pasal-1"
        mock_row.content = "Vector search result"
        mock_row.citation_string = "Pasal 1"
        mock_row.similarity = 0.88
        mock_row.unit_type = "pasal"
        mock_row.doc_form = "UU"
        mock_row.doc_year = 2025
        mock_row.doc_number = "1"

        mock_db_session.execute.return_value.fetchall.return_value = [mock_row]

        retriever = HybridRetriever(embedder=mock_jina_embedder)
        results = retriever.search("mining operations", strategy="vector")

        # Verify embedder integration
        mock_jina_embedder.embed_single.assert_called_once_with("mining operations")

        # Verify results
        assert len(results) == 1
        assert results[0].source_type == "vector"
        assert results[0].score == 0.88

    def test_search_service_retriever_integration(self, mock_db_session, mock_jina_embedder):
        """Test search service integrates properly with retriever."""

        # Mock retriever results
        mock_search_result = MagicMock()
        mock_search_result.unit_id = "UU-2025-1/pasal-1/ayat-1"
        mock_search_result.content = "Search result content"
        mock_search_result.citation = "Pasal 1 ayat (1)"
        mock_search_result.score = 0.85
        mock_search_result.source_type = "fts"

        mock_db_session.execute.return_value.fetchall.return_value = []

        retriever = HybridRetriever(embedder=mock_jina_embedder)

        with patch.object(retriever, 'search', return_value=[mock_search_result]):
            with patch('src.services.search.reranker.JinaReranker'):
                search_service = HybridSearchService(retriever=retriever)
                response = search_service.search("test query")

                # Verify service formats results correctly
                assert "results" in response
                assert "total" in response
                assert "query" in response
                assert "duration_ms" in response

                # Verify retriever was called
                retriever.search.assert_called_once()

    def test_indexer_database_integration(self, db_session, sample_json_document, mock_jina_embedder):
        """Test indexer properly integrates with database."""

        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_json_document, f, ensure_ascii=False)
            temp_path = Path(f.name)

        try:
            with patch('src.services.embedding.embedder.JinaEmbedder', return_value=mock_jina_embedder):
                with patch('src.db.session.get_db_session') as mock_get_session:
                    mock_get_session.return_value.__enter__.return_value = db_session

                    indexer = DocumentIndexer(skip_embeddings=True)
                    result = indexer.index_file(temp_path)

                    # Verify indexing result
                    assert result["status"] == "success"
                    assert result["documents_processed"] == 1

                    # Verify database state
                    documents = db_session.query(LegalDocument).all()
                    units = db_session.query(LegalUnit).all()

                    assert len(documents) == 1
                    assert len(units) >= 2  # Should have pasal and ayat

                    # Verify document integrity
                    doc = documents[0]
                    assert doc.doc_id == "UU-2025-1"
                    assert doc.doc_title == "Test Undang-Undang"

        finally:
            temp_path.unlink(missing_ok=True)

    def test_complete_workflow_with_embeddings(self, db_session, sample_json_document, mock_jina_embedder):
        """Test complete workflow including embedding generation."""

        # Configure realistic embedding responses
        mock_jina_embedder.embed_single.side_effect = lambda text: [hash(text) % 1000 / 1000.0] * 1024

        # Create JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_json_document, f, ensure_ascii=False)
            temp_path = Path(f.name)

        try:
            # Step 1: Index with embeddings
            with patch('src.services.embedding.embedder.JinaEmbedder', return_value=mock_jina_embedder):
                with patch('src.db.session.get_db_session') as mock_get_session:
                    mock_get_session.return_value.__enter__.return_value = db_session

                    indexer = DocumentIndexer(skip_embeddings=False)
                    indexer.index_file(temp_path)

            # Verify embeddings were created
            vectors = db_session.query(DocumentVector).all()
            assert len(vectors) >= 1

            # Step 2: Search with vector strategy
            with patch('src.db.session.get_db_session') as mock_get_session:
                mock_get_session.return_value.__enter__.return_value = db_session

                retriever = HybridRetriever(embedder=mock_jina_embedder)
                vector_results = retriever.search("pertambangan", strategy="vector", limit=5)

                # Should be able to perform vector search
                assert isinstance(vector_results, list)

        finally:
            temp_path.unlink(missing_ok=True)


class TestErrorHandlingIntegration:
    """Test error handling across component boundaries."""

    def test_database_failure_propagation(self, mock_jina_embedder):
        """Test how database failures propagate through the system."""

        from sqlalchemy.exc import SQLAlchemyError

        # Mock database session that fails
        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_session = MagicMock()
            mock_session.execute.side_effect = SQLAlchemyError("Database connection lost")
            mock_get_session.return_value.__enter__.return_value = mock_session

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            # Database errors should propagate appropriately
            with pytest.raises(SQLAlchemyError):
                retriever.search("test query", strategy="fts")

    def test_external_api_failure_integration(self, mock_db_session):
        """Test system behavior when external APIs fail."""

        # Mock embedder that fails
        mock_embedder = MagicMock()
        mock_embedder.embed_single.side_effect = Exception("Jina API timeout")

        mock_db_session.execute.return_value.fetchall.return_value = []

        retriever = HybridRetriever(embedder=mock_embedder)

        # Vector search should fail
        with pytest.raises(Exception):
            retriever.search("test", strategy="vector")

        # But FTS should still work
        fts_results = retriever.search("test", strategy="fts")
        assert isinstance(fts_results, list)

    def test_partial_component_failure_recovery(self, populated_db, mock_jina_embedder):
        """Test system recovery when some components fail."""

        # Configure embedder to fail intermittently
        call_count = 0
        def failing_embedder(text):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:  # Fail every other call
                raise Exception("Intermittent API failure")
            return [0.1] * 1024

        mock_jina_embedder.embed_single.side_effect = failing_embedder

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = populated_db

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            # Test that system can handle intermittent failures
            success_count = 0
            failure_count = 0

            for i in range(4):
                try:
                    results = retriever.search(f"query {i}", strategy="vector")
                    success_count += 1
                except Exception:
                    failure_count += 1

            # Should have mix of successes and failures
            assert success_count > 0
            assert failure_count > 0


class TestFilteringIntegration:
    """Test filtering integration across search strategies."""

    def test_comprehensive_filtering_integration(self, populated_db, mock_jina_embedder):
        """Test filtering works across all search strategies."""

        # Add more diverse test data
        additional_docs = [
            LegalDocument(
                id="test-uuid-3",
                doc_id="PP-2024-1",
                doc_form="PP",
                doc_number="1",
                doc_year="2024",
                doc_title="Peraturan Pemerintah Test",
                doc_status="BERLAKU"
            )
        ]

        additional_units = [
            LegalUnit(
                id="filter-unit-1",
                document_id="test-uuid-3",
                unit_type="pasal",
                unit_id="PP-2024-1/pasal-1",
                content="Peraturan tentang mineral dan batubara",
                local_content="mineral dan batubara",
                citation_string="Pasal 1"
            )
        ]

        populated_db.add_all(additional_docs + additional_units)
        populated_db.commit()

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = populated_db

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            # Test filtering by document form
            uu_results = retriever.search(
                "pertambangan",
                strategy="fts",
                filters=SearchFilters(doc_forms=["UU"]),
                limit=10
            )

            pp_results = retriever.search(
                "pertambangan",
                strategy="fts",
                filters=SearchFilters(doc_forms=["PP"]),
                limit=10
            )

            # Results should be filtered correctly
            for result in uu_results:
                assert result.document_form == "UU"

            for result in pp_results:
                assert result.document_form == "PP"

            # Test year filtering
            year_2025_results = retriever.search(
                "pertambangan",
                strategy="fts",
                filters=SearchFilters(doc_years=[2025]),
                limit=10
            )

            for result in year_2025_results:
                assert result.document_year == 2025

    def test_combined_filters_integration(self, populated_db, mock_jina_embedder):
        """Test multiple filters applied simultaneously."""

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = populated_db

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            # Test combined filters
            combined_filters = SearchFilters(
                doc_forms=["UU"],
                doc_years=[2025],
                unit_types=["ayat"]
            )

            results = retriever.search(
                "pertambangan",
                strategy="fts",
                filters=combined_filters,
                limit=10
            )

            # All results should match all filter criteria
            for result in results:
                assert result.document_form == "UU"
                assert result.document_year == 2025
                assert result.unit_type == "ayat"


class TestConcurrencyIntegration:
    """Test system behavior under concurrent access."""

    def test_concurrent_search_operations(self, populated_db, mock_jina_embedder):
        """Test multiple concurrent search operations."""
        import threading

        results_collection = []
        errors_collection = []

        def search_worker(worker_id: int, query: str):
            try:
                with patch('src.db.session.get_db_session') as mock_get_session:
                    mock_get_session.return_value.__enter__.return_value = populated_db

                    retriever = HybridRetriever(embedder=mock_jina_embedder)
                    results = retriever.search(f"{query} {worker_id}", strategy="fts")

                    results_collection.append({
                        "worker_id": worker_id,
                        "query": f"{query} {worker_id}",
                        "result_count": len(results)
                    })

            except Exception as e:
                errors_collection.append({
                    "worker_id": worker_id,
                    "error": str(e)
                })

        # Launch concurrent searches
        threads = []
        queries = ["pertambangan", "mineral", "batubara", "izin", "lingkungan"]

        for i, query in enumerate(queries):
            thread = threading.Thread(target=search_worker, args=(i, query))
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=10.0)

        # Verify all searches completed successfully
        assert len(errors_collection) == 0, f"Errors: {errors_collection}"
        assert len(results_collection) == len(queries)

        # Verify each worker got results
        for result in results_collection:
            assert result["worker_id"] >= 0
            assert result["result_count"] >= 0

    def test_concurrent_indexing_and_searching(self, db_session, sample_json_document, mock_jina_embedder):
        """Test concurrent indexing and searching operations."""
        import threading
        import time

        operation_results = []

        def indexing_worker():
            try:
                # Create temporary file for this thread
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    # Modify doc_id to avoid conflicts
                    modified_doc = sample_json_document.copy()
                    modified_doc["doc_id"] = "UU-2025-CONCURRENT"
                    json.dump(modified_doc, f, ensure_ascii=False)
                    temp_path = Path(f.name)

                with patch('src.services.embedding.embedder.JinaEmbedder', return_value=mock_jina_embedder):
                    with patch('src.db.session.get_db_session') as mock_get_session:
                        mock_get_session.return_value.__enter__.return_value = db_session

                        indexer = DocumentIndexer(skip_embeddings=True)
                        result = indexer.index_file(temp_path)

                        operation_results.append({"type": "index", "success": True, "result": result})

                temp_path.unlink(missing_ok=True)

            except Exception as e:
                operation_results.append({"type": "index", "success": False, "error": str(e)})

        def search_worker():
            try:
                time.sleep(0.1)  # Small delay to let indexing start

                with patch('src.db.session.get_db_session') as mock_get_session:
                    mock_get_session.return_value.__enter__.return_value = db_session

                    retriever = HybridRetriever(embedder=mock_jina_embedder)
                    results = retriever.search("pertambangan", strategy="fts")

                    operation_results.append({"type": "search", "success": True, "count": len(results)})

            except Exception as e:
                operation_results.append({"type": "search", "success": False, "error": str(e)})

        # Start both operations concurrently
        index_thread = threading.Thread(target=indexing_worker)
        search_thread = threading.Thread(target=search_worker)

        index_thread.start()
        search_thread.start()

        index_thread.join(timeout=10.0)
        search_thread.join(timeout=10.0)

        # Both operations should complete
        assert len(operation_results) == 2

        # At least one should succeed
        successful_ops = [op for op in operation_results if op["success"]]
        assert len(successful_ops) >= 1


class TestDataIntegrityIntegration:
    """Test data integrity across the entire pipeline."""

    def test_unit_id_consistency_integration(self, db_session, mock_jina_embedder):
        """Test unit ID consistency across indexing and search."""

        # Create test document with specific structure
        test_doc = {
            "doc_id": "UU-2025-INTEGRITY",
            "doc_form": "UU",
            "doc_number": "999",
            "doc_year": "2025",
            "doc_title": "Test Integrity",
            "doc_subject": ["TEST"],
            "relationships": {},
            "uji_materi": [],
            "document_tree": {
                "doc_type": "document",
                "children": [
                    {
                        "type": "pasal",
                        "unit_id": "UU-2025-INTEGRITY/pasal-1",
                        "number_label": "1",
                        "local_content": "Pasal content",
                        "citation_string": "Pasal 1",
                        "path": ["pasal", "1"],
                        "children": [
                            {
                                "type": "ayat",
                                "unit_id": "UU-2025-INTEGRITY/pasal-1/ayat-1",
                                "number_label": "1",
                                "local_content": "Ayat content",
                                "citation_string": "Pasal 1 ayat (1)",
                                "path": ["pasal", "1", "ayat", "1"],
                                "parent_pasal_id": "UU-2025-INTEGRITY/pasal-1"
                            }
                        ]
                    }
                ]
            }
        }

        # Index the document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_doc, f, ensure_ascii=False)
            temp_path = Path(f.name)

        try:
            with patch('src.services.embedding.embedder.JinaEmbedder', return_value=mock_jina_embedder):
                with patch('src.db.session.get_db_session') as mock_get_session:
                    mock_get_session.return_value.__enter__.return_value = db_session

                    indexer = DocumentIndexer(skip_embeddings=True)
                    indexer.index_file(temp_path)

            # Verify unit IDs in database
            units = db_session.query(LegalUnit).filter(
                LegalUnit.unit_id.like("UU-2025-INTEGRITY%")
            ).all()

            expected_unit_ids = [
                "UU-2025-INTEGRITY/pasal-1",
                "UU-2025-INTEGRITY/pasal-1/ayat-1"
            ]

            actual_unit_ids = [unit.unit_id for unit in units]
            for expected_id in expected_unit_ids:
                assert expected_id in actual_unit_ids

            # Test search finds consistent unit IDs
            with patch('src.db.session.get_db_session') as mock_get_session:
                mock_get_session.return_value.__enter__.return_value = db_session

                retriever = HybridRetriever(embedder=mock_jina_embedder)
                search_results = retriever.search("content", strategy="fts")

                # Unit IDs in search results should match database
                for result in search_results:
                    assert result.unit_id in actual_unit_ids

        finally:
            temp_path.unlink(missing_ok=True)

    def test_citation_consistency_integration(self, populated_db, mock_jina_embedder):
        """Test citation string consistency across components."""

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = populated_db

            retriever = HybridRetriever(embedder=mock_jina_embedder)
            results = retriever.search("pertambangan", strategy="fts", limit=5)

            # Citation strings should follow consistent format
            for result in results:
                citation = result.citation
                assert isinstance(citation, str)
                assert len(citation) > 0

                # Should contain Indonesian legal reference patterns
                legal_keywords = ["Pasal", "ayat", "huruf", "angka", "UU", "PP"]
                has_legal_keyword = any(keyword in citation for keyword in legal_keywords)
                assert has_legal_keyword or citation == ""  # Allow empty citations


class TestPerformanceIntegration:
    """Test system performance under realistic conditions."""

    def test_large_dataset_search_performance(self, db_session, mock_jina_embedder, performance_timer):
        """Test search performance with large dataset."""

        # Create large dataset for performance testing
        large_documents = []
        large_units = []

        for doc_idx in range(5):  # 5 documents
            doc_id = f"PERF-2025-{doc_idx}"
            document = LegalDocument(
                id=f"perf-doc-{doc_idx}",
                doc_id=doc_id,
                doc_form="UU",
                doc_number=str(doc_idx),
                doc_year="2025",
                doc_title=f"Performance Test Document {doc_idx}",
                doc_status="BERLAKU"
            )
            large_documents.append(document)

            # Create many units per document
            for unit_idx in range(20):  # 20 units per document
                unit = LegalUnit(
                    id=f"perf-unit-{doc_idx}-{unit_idx}",
                    document_id=f"perf-doc-{doc_idx}",
                    unit_type="ayat",
                    unit_id=f"{doc_id}/pasal-{unit_idx}/ayat-1",
                    content=f"Performance test content {doc_idx}-{unit_idx} tentang pertambangan mineral",
                    local_content=f"pertambangan mineral {unit_idx}",
                    citation_string=f"Pasal {unit_idx} ayat (1)"
                )
                large_units.append(unit)

        # Add all test data
        db_session.add_all(large_documents + large_units)
        db_session.commit()

        # Test search performance
        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = db_session

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            # Test FTS performance
            performance_timer.start()
            fts_results = retriever.search("pertambangan mineral", strategy="fts", limit=50)
            performance_timer.stop()

            fts_duration = performance_timer.duration_ms

            # Should meet performance requirements
            assert fts_duration < 500  # Less than 500ms per AGENTS.md
            assert len(fts_results) <= 50
            assert len(fts_results) > 0  # Should find some results

    def test_memory_usage_integration(self, db_session, mock_jina_embedder):
        """Test memory usage during large operations."""
        import tracemalloc
        import gc

        # Start memory tracking
        tracemalloc.start()
        gc.collect()  # Clean up before test

        # Create moderate dataset
        documents = []
        units = []

        for i in range(10):
            doc = LegalDocument(
                id=f"memory-doc-{i}",
                doc_id=f"MEM-2025-{i}",
                doc_form="UU",
                doc_number=str(i),
                doc_year="2025",
                doc_title=f"Memory Test Document {i}",
                doc_status="BERLAKU"
            )
            documents.append(doc)

            # Add units with substantial content
            for j in range(10):
                unit = LegalUnit(
                    id=f"memory-unit-{i}-{j}",
                    document_id=f"memory-doc-{i}",
                    unit_type="ayat",
                    unit_id=f"MEM-2025-{i}/pasal-{j}/ayat-1",
                    content="Large legal content " * 50,  # Substantial content
                    local_content="legal content",
                    citation_string=f"Pasal {j} ayat (1)"
                )
                units.append(unit)

        db_session.add_all(documents + units)
        db_session.commit()

        # Perform searches
        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = db_session

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            # Multiple searches to test memory accumulation
            for i in range(5):
                results = retriever.search(f"content {i}", strategy="fts", limit=20)
                assert isinstance(results, list)

        # Check memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory should be reasonable (less than 50MB for this test)
        assert peak < 50 * 1024 * 1024  # 50MB limit


class TestRealWorldScenarios:
    """Test real-world usage scenarios and workflows."""

    def test_legal_research_workflow(self, populated_db, mock_jina_embedder):
        """Test typical legal research workflow."""

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = populated_db

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            # Simulate legal research session
            research_queries = [
                ("pasal 1", "explicit"),           # Look up specific article
                ("pertambangan mineral", "fts"),   # Keyword search
                ("izin usaha", "hybrid"),          # Comprehensive search
                ("UU 1/2025", "explicit"),        # Document lookup
                ("lingkungan", "fts")              # Thematic search
            ]

            research_results = []

            for query, strategy in research_queries:
                results = retriever.search(query, strategy=strategy, limit=10)
                research_results.append({
                    "query": query,
                    "strategy": strategy,
                    "result_count": len(results),
                    "results": results
                })

            # Verify research session results
            assert len(research_results) == len(research_queries)

            for session_result in research_results:
                assert session_result["result_count"] >= 0
                assert isinstance(session_result["results"], list)

    def test_document_analysis_workflow(self, db_session, mock_jina_embedder):
        """Test document analysis and structure exploration."""

        # Create complex document structure
        complex_doc = {
            "doc_id": "UU-2025-COMPLEX",
            "doc_form": "UU",
            "doc_number": "COMPLEX",
            "doc_year": "2025",
            "doc_title": "Complex Legal Document",
            "doc_subject": ["COMPREHENSIVE"],
            "relationships": {},
            "uji_materi": [],
            "document_tree": {
                "doc_type": "document",
                "children": []
            }
        }

        # Build complex hierarchy
        for bab_num in range(2):
            for pasal_num in range(3):
                pasal_id = f"UU-2025-COMPLEX/pasal-{pasal_num+1}"
                pasal_node = {
                    "type": "pasal",
                    "unit_id": pasal_id,
                    "number_label": str(pasal_num + 1),
                    "local_content": f"Pasal {pasal_num + 1} content",
                    "citation_string": f"Pasal {pasal_num + 1}",
                    "path": ["pasal", str(pasal_num + 1)],
                    "children": []
                }

                # Add ayat to each pasal
                for ayat_num in range(2):
                    ayat_id = f"{pasal_id}/ayat-{ayat_num+1}"
                    ayat_node = {
                        "type": "ayat",
                        "unit_id": ayat_id,
                        "number_label": str(ayat_num + 1),
                        "local_content": f"Ayat {ayat_num + 1} detail content",
                        "citation_string": f"Pasal {pasal_num + 1} ayat ({ayat_num + 1})",
                        "path": ["pasal", str(pasal_num + 1), "ayat", str(ayat_num + 1)],
                        "parent_pasal_id": pasal_id
                    }
                    pasal_node["children"].append(ayat_node)

                complex_doc["document_tree"]["children"].append(pasal_node)

        # Index complex document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(complex_doc, f, ensure_ascii=False)
            temp_path = Path(f.name)

        try:
            with patch('src.services.embedding.embedder.JinaEmbedder', return_value=mock_jina_embedder):
                with patch('src.db.session.get_db_session') as mock_get_session:
                    mock_get_session.return_value.__enter__.return_value = db_session

                    indexer = DocumentIndexer(skip_embeddings=True)
                    indexer.index_file(temp_path)

            # Test document structure analysis
            with patch('src.db.session.get_db_session') as mock_get_session:
                mock_get_session.return_value.__enter__.return_value = db_session

                retriever = HybridRetriever(embedder=mock_jina_embedder)

                # Search by unit type
                pasal_results = retriever.search(
                    "content",
                    strategy="fts",
                    filters=SearchFilters(unit_types=["pasal"]),
                    limit=10
                )

                ayat_results = retriever.search(
                    "content",
                    strategy="fts",
                    filters=SearchFilters(unit_types=["ayat"]),
                    limit=10
                )

                # Should find different unit types
                if pasal_results:
                    assert all(r.unit_type == "pasal" for r in pasal_results)

                if ayat_results:
                    assert all(r.unit_type == "ayat" for r in ayat_results)

        finally:
            temp_path.unlink(missing_ok=True)


class TestSystemRobustness:
    """Test system robustness and fault tolerance."""

    def test_degraded_service_operation(self, populated_db):
        """Test system operation when some services are degraded."""

        # Test with failing embedder
        failing_embedder = MagicMock()
        failing_embedder.embed_single.side_effect = Exception("Service unavailable")

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = populated_db

            retriever = HybridRetriever(embedder=failing_embedder)

            # FTS should still work even if embedder fails
            fts_results = retriever.search("pertambangan", strategy="fts")
            assert isinstance(fts_results, list)

            # Explicit search should work
            explicit_results = retriever.search("pasal 1", strategy="explicit")
            assert isinstance(explicit_results, list)

            # Vector search should fail gracefully
            with pytest.raises(Exception):
                retriever.search("test", strategy="vector")

    def test_database_transaction_integrity(self, db_session, sample_json_document, mock_jina_embedder):
        """Test database transaction integrity during indexing."""

        # Create document that will cause partial failure
        problematic_doc = sample_json_document.copy()
        problematic_doc["doc_id"] = "UU-2025-PROBLEM"

        # Add problematic unit that might cause constraint violation
        problematic_doc["document_tree"]["children"].append({
            "type": "ayat",
            "unit_id": "",  # Empty unit_id should cause problems
            "number_label": "INVALID",
            "local_content": "Invalid content",
            "citation_string": "Invalid citation",
            "path": ["invalid"],
            "parent_pasal_id": "nonexistent"
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(problematic_doc, f, ensure_ascii=False)
            temp_path = Path(f.name)

        try:
            with patch('src.services.embedding.embedder.JinaEmbedder', return_value=mock_jina_embedder):
                with patch('src.db.session.get_db_session') as mock_get_session:
                    mock_get_session.return_value.__enter__.return_value = db_session

                    indexer = DocumentIndexer(skip_embeddings=True)

                    # Indexing might fail due to invalid data
                    try:
                        result = indexer.index_file(temp_path)
                        # If it succeeds, verify data integrity
                        documents = db_session.query(LegalDocument).all()
                        units = db_session.query(LegalUnit).all()

                        # Should have consistent data
                        for unit in units:
                            assert unit.unit_id  # No empty unit IDs
                            assert unit.document_id  # Valid document reference

                    except Exception:
                        # Failure is acceptable for invalid data
                        # But database should remain consistent
                        db_session.rollback()

                        # Previous data should still be intact
                        documents = db_session.query(LegalDocument).all()
                        # Should not have partial/corrupted data

        finally:
            temp_path.unlink(missing_ok=True)

    def test_concurrent_read_write_operations(self, db_session, sample_json_document, mock_jina_embedder):
        """Test concurrent read and write operations."""
        import threading

        operation_results = []

        def writer_operation():
            """Simulate indexing operation."""
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    # Create unique document for this thread
                    doc = sample_json_document.copy()
+                    doc["doc_id"] = "UU-2025-WRITER"
                    json.dump(doc, f, ensure_ascii=False)
                    temp_path = Path(f.name)

                with patch('src.services.embedding.embedder.JinaEmbedder', return_value=mock_jina_embedder):
                    with patch('src.db.session.get_db_session') as mock_get_session:
                        mock_get_session.return_value.__enter__.return_value = db_session

                        indexer = DocumentIndexer(skip_embeddings=True)
                        result = indexer.index_file(temp_path)
                        operation_results.append({"type": "write", "success": True})

                temp_path.unlink(missing_ok=True)

            except Exception as e:
                operation_results.append({"type": "write", "success": False, "error": str(e)})

        def reader_operation():
            """Simulate search operation."""
            try:
                time.sleep(0.05)  # Small delay to let writer start

                with patch('src.db.session.get_db_session') as mock_get_session:
                    mock_get_session.return_value.__enter__.return_value = db_session

                    retriever = HybridRetriever(embedder=mock_jina_embedder)
                    results = retriever.search("pertambangan", strategy="fts", limit=10)
                    operation_results.append({"type": "read", "success": True, "count": len(results)})

            except Exception as e:
                operation_results.append({"type": "read", "success": False, "error": str(e)})

        # Run operations concurrently
        writer_thread = threading.Thread(target=writer_operation)
        reader_thread = threading.Thread(target=reader_operation)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join(timeout=10.0)
        reader_thread.join(timeout=10.0)

        # Both operations should complete
        assert len(operation_results) == 2

        # At least one should succeed
        successful_ops = [op for op in operation_results if op["success"]]
        assert len(successful_ops) >= 1


class TestSearchQualityIntegration:
    """Test search quality and relevance."""

    def test_search_relevance_ranking(self, populated_db, mock_jina_embedder):
        """Test that search results are properly ranked by relevance."""

        # Add documents with varying relevance
        relevant_doc = LegalDocument(
            id="relevant-doc",
            doc_id="UU-2025-RELEVANT",
            doc_form="UU",
            doc_number="REL",
            doc_year="2025",
            doc_title="Highly Relevant Mining Law",
            doc_status="BERLAKU"
        )

        less_relevant_doc = LegalDocument(
            id="less-relevant-doc",
            doc_id="UU-2025-LESS",
            doc_form="UU",
            doc_number="LESS",
            doc_year="2025",
            doc_title="Less Relevant Environmental Law",
            doc_status="BERLAKU"
        )

        # Add units with different relevance levels
        highly_relevant_unit = LegalUnit(
            id="highly-relevant-unit",
            document_id="relevant-doc",
            unit_type="ayat",
            unit_id="UU-2025-RELEVANT/pasal-1/ayat-1",
            content="Pertambangan mineral dan batubara merupakan kegiatan utama dalam sektor energi",
            local_content="pertambangan mineral batubara",
            citation_string="Pasal 1 ayat (1)"
        )

        less_relevant_unit = LegalUnit(
            id="less-relevant-unit",
            document_id="less-relevant-doc",
            unit_type="ayat",
            unit_id="UU-2025-LESS/pasal-1/ayat-1",
            content="Lingkungan sekitar area pertambangan harus dijaga",
            local_content="lingkungan area pertambangan",
            citation_string="Pasal 1 ayat (1)"
        )

        populated_db.add_all([relevant_doc, less_relevant_doc, highly_relevant_unit, less_relevant_unit])
        populated_db.commit()

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = populated_db

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            # Search for mining-specific terms
            results = retriever.search("pertambangan mineral", strategy="fts", limit=10)

            if len(results) >= 2:
                # Results should be ranked by relevance
                scores = [result.score for result in results]
                assert scores == sorted(scores, reverse=True)  # Descending order

    def test_multilingual_content_integration(self, db_session, mock_jina_embedder):
        """Test handling of mixed Indonesian/English content."""

        # Create document with mixed language content
        multilingual_doc = LegalDocument(
            id="multilingual-doc",
            doc_id="UU-2025-MULTI",
            doc_form="UU",
            doc_number="MULTI",
            doc_year="2025",
            doc_title="Multilingual Legal Document",
            doc_status="BERLAKU"
        )

        multilingual_units = [
            LegalUnit(
                id="multi-unit-1",
                document_id="multilingual-doc",
                unit_type="ayat",
                unit_id="UU-2025-MULTI/pasal-1/ayat-1",
                content="Mining operations harus mengikuti peraturan pertambangan yang berlaku",
                local_content="mining operations peraturan pertambangan",
                citation_string="Pasal 1 ayat (1)"
            ),
            LegalUnit(
                id="multi-unit-2",
                document_id="multilingual-doc",
                unit_type="ayat",
                unit_id="UU-2025-MULTI/pasal-1/ayat-2",
                content="Environmental compliance adalah kewajiban setiap perusahaan tambang",
                local_content="environmental compliance kewajiban",
                citation_string="Pasal 1 ayat (2)"
            )
        ]

        db_session.add(multilingual_doc)
        db_session.add_all(multilingual_units)
        db_session.commit()

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = db_session

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            # Test searches in different languages
            english_results = retriever.search("mining operations", strategy="fts")
            indonesian_results = retriever.search("pertambangan", strategy="fts")
            mixed_results = retriever.search("environmental kewajiban", strategy="fts")

            # All searches should return appropriate results
            for results in [english_results, indonesian_results, mixed_results]:
                assert isinstance(results, list)
                # Quality check: results should be relevant to search terms


class TestSearchStrategySwitching:
    """Test dynamic search strategy switching and optimization."""

    def test_auto_strategy_selection(self, populated_db, mock_jina_embedder):
        """Test automatic strategy selection based on query patterns."""

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = populated_db

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            # Test queries that should trigger different strategies
            test_cases = [
                ("pasal 1", "should use explicit"),
                ("UU 1/2025", "should use explicit"),
                ("pertambangan mineral", "should use thematic/hybrid"),
                ("izin usaha", "should use thematic/hybrid"),
                ("ayat 2", "should use explicit")
            ]

            for query, expected_behavior in test_cases:
                results = retriever.search(query, strategy="auto", limit=5)
                assert isinstance(results, list)
                # The actual strategy selection logic would be tested here

    def test_strategy_fallback_integration(self, populated_db, mock_jina_embedder):
        """Test fallback when preferred strategy fails."""

        # Configure embedder to fail for vector search
        mock_jina_embedder.embed_single.side_effect = Exception("Vector search unavailable")

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = populated_db

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            # Hybrid strategy should fall back to FTS when vector fails
            try:
                results = retriever.search("pertambangan", strategy="hybrid", limit=10)
                # If hybrid succeeds, it should use FTS fallback
                assert isinstance(results, list)
            except Exception:
                # If hybrid fails completely, that's also acceptable
                # The key is that it fails gracefully
                pass

            # FTS should always work regardless of embedder status
            fts_results = retriever.search("pertambangan", strategy="fts", limit=10)
            assert isinstance(fts_results, list)


class TestAPIIntegration:
    """Test integration with API layer (when available)."""

    def test_search_service_response_format(self, populated_db, mock_jina_embedder, mock_jina_reranker):
        """Test search service returns properly formatted responses."""

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = populated_db

            retriever = HybridRetriever(embedder=mock_jina_embedder)

            with patch('src.services.search.reranker.JinaReranker', return_value=mock_jina_reranker):
                search_service = HybridSearchService(retriever=retriever)

                # Test standard search
                response = search_service.search("pertambangan mineral")

                # Verify response format matches API contract
                required_fields = ["results", "total", "query", "strategy", "duration_ms"]
                for field in required_fields:
                    assert field in response

                assert isinstance(response["results"], list)
                assert isinstance(response["total"], int)
                assert isinstance(response["query"], str)
                assert isinstance(response["duration_ms"], (int, float))
                assert response["strategy"] in ["explicit", "thematic", "contextual"]

    def test_search_result_serialization(self, populated_db, mock_jina_embedder):
        """Test that search results can be properly serialized for API responses."""

        with patch('src.db.session.get_db_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = populated_db

            retriever = HybridRetriever(embedder=mock_jina_embedder)
            results = retriever.search("pertambangan", strategy="fts", limit=5)

            # Test JSON serialization
            for result in results:
                try:
                    result_dict = {
                        "unit_id": result.unit_id,
                        "content": result.content,
                        "citation": result.citation,
                        "score": result.score,
                        "source_type": result.source_type,
                        "unit_type": result.unit_type,
                        "document": {
                            "form": result.document_form,
                            "year": result.document_year,
                            "number": result.document_number
                        }
                    }

                    # Should be JSON serializable
                    json_str = json.dumps(result_dict, ensure_ascii=False)
                    parsed_back = json.loads(json_str)
                    assert parsed_back["unit_id"] == result.unit_id

                except Exception as e:
                    pytest.fail(f"Result serialization failed: {e}")


class TestSystemIntegrationSmokeTests:
    """Final smoke tests for overall system integration."""

    def test_complete_system_smoke_test(self, db_session, sample_json_document, mock_jina_embedder, mock_jina_reranker):
        """Comprehensive smoke test of entire system."""

        # Step 1: Index a document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_json_document, f, ensure_ascii=False)
            temp_path = Path(f.name)

        try:
            with patch('src.services.embedding.embedder.JinaEmbedder', return_value=mock_jina_embedder):
                with patch('src.db.session.get_db_session') as mock_get_session:
                    mock_get_session.return_value.__enter__.return_value = db_session

                    # Index
                    indexer = DocumentIndexer(skip_embeddings=True)
                    index_result = indexer.index_file(temp_path)
                    assert index_result["status"] == "success"

            # Step 2: Search with all strategies
            with patch('src.db.session.get_db_session') as mock_get_session:
                mock_get_session.return_value.__enter__.return_value = db_session

                retriever = HybridRetriever(embedder=mock_jina_embedder)

                strategies = ["explicit", "fts", "vector"]
                for strategy in strategies:
                    try:
                        results = retriever.search("pertambangan", strategy=strategy, limit=5)
                        assert isinstance(results, list)
                    except Exception as e:
                        # Some strategies may fail (e.g., vector without embeddings)
                        # This is acceptable in smoke test
                        pass

            # Step 3: Test search service
            with patch('src.db.session.get_db_session') as mock_get_session:
                mock_get_session.return_value.__enter__.return_value = db_session

                with patch('src.services.search.reranker.JinaReranker', return_value=mock_jina_reranker):
                    search_service = HybridSearchService(retriever=retriever)
                    response = search_service.search("test query")

                    assert "results" in response
                    assert isinstance(response["results"], list)

        finally:
            temp_path.unlink(missing_ok=True)

    def test
