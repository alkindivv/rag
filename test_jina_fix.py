#!/usr/bin/env python3
"""
Test Jina API Emergency Fix
===========================

This script tests whether the emergency timeout fixes for Jina API are working.
Run this after applying the emergency fix to verify functionality.

Usage:
    python test_jina_fix.py
"""

import os
import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def log(message: str, level: str = "INFO"):
    """Log with timestamp and emoji."""
    timestamp = time.strftime("%H:%M:%S")
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "TEST": "🧪"}
    print(f"[{timestamp}] {icons.get(level, '📝')} {message}")

def test_environment():
    """Test environment configuration."""
    log("Testing environment configuration...", "TEST")

    try:
        from src.config.settings import settings

        # Check required settings
        checks = {
            "JINA_API_KEY": settings.jina_api_key,
            "Embedding Model": settings.embedding_model,
            "Embedding Dimensions": settings.embedding_dim,
            "Jina Embed Base": settings.jina_embed_base
        }

        for name, value in checks.items():
            if value:
                if name == "JINA_API_KEY":
                    display_value = f"{str(value)[:8]}..." if len(str(value)) > 8 else "***"
                else:
                    display_value = value
                log(f"  ✅ {name}: {display_value}", "SUCCESS")
            else:
                log(f"  ❌ {name}: NOT SET", "ERROR")
                return False

        return True

    except Exception as e:
        log(f"Environment test failed: {e}", "ERROR")
        return False

def test_timeout_settings():
    """Test timeout configuration."""
    log("Checking timeout settings...", "TEST")

    timeout_vars = [
        "EMBEDDING_REQUEST_TIMEOUT",
        "EMBEDDING_MAX_RETRIES",
        "EMBEDDING_BATCH_SIZE"
    ]

    for var in timeout_vars:
        value = os.getenv(var, "not set")
        log(f"  {var}: {value}", "INFO")

    # Check if emergency timeouts are applied
    timeout = int(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "30"))
    if timeout >= 90:
        log("  ✅ Emergency timeout settings detected", "SUCCESS")
        return True
    else:
        log(f"  ⚠️ Timeout is {timeout}s, emergency fix may not be applied", "WARNING")
        return False

def test_basic_embedding():
    """Test basic embedding functionality."""
    log("Testing basic embedding functionality...", "TEST")

    try:
        from src.services.embedding.embedder import JinaV4Embedder

        # Initialize embedder
        log("  Initializing JinaV4Embedder...", "INFO")
        embedder = JinaV4Embedder()

        # Test simple embedding
        test_text = "test embedding"
        log(f"  Embedding text: '{test_text}'", "INFO")

        start_time = time.time()
        embedding = embedder.embed_single(test_text, task="retrieval.query")
        duration = (time.time() - start_time) * 1000

        if embedding and len(embedding) == 384:
            log(f"  ✅ Embedding successful: 384 dimensions, {duration:.1f}ms", "SUCCESS")
            return True
        else:
            log(f"  ❌ Embedding failed: got {len(embedding) if embedding else 0} dimensions", "ERROR")
            return False

    except Exception as e:
        log(f"  ❌ Embedding test failed: {e}", "ERROR")
        return False

def test_legal_queries():
    """Test embedding with legal query examples."""
    log("Testing legal query embeddings...", "TEST")

    try:
        from src.services.embedding.embedder import JinaV4Embedder
        embedder = JinaV4Embedder()

        test_queries = [
            "definisi badan hukum",
            "sanksi pidana korupsi",
            "UU 8/2019 Pasal 6"
        ]

        for i, query in enumerate(test_queries, 1):
            log(f"  Query {i}/3: '{query}'", "INFO")

            start_time = time.time()
            embedding = embedder.embed_single(query, task="retrieval.query")
            duration = (time.time() - start_time) * 1000

            if embedding and len(embedding) == 384:
                log(f"    ✅ Success: {duration:.1f}ms", "SUCCESS")
            else:
                log(f"    ❌ Failed: {duration:.1f}ms", "ERROR")
                return False

        log("  ✅ All legal queries embedded successfully", "SUCCESS")
        return True

    except Exception as e:
        log(f"  ❌ Legal query test failed: {e}", "ERROR")
        return False

def test_batch_embedding():
    """Test batch embedding functionality."""
    log("Testing batch embedding...", "TEST")

    try:
        from src.services.embedding.embedder import JinaV4Embedder
        embedder = JinaV4Embedder()

        batch_texts = [
            "definisi badan hukum",
            "sanksi pidana",
            "tanggung jawab sosial"
        ]

        log(f"  Batch size: {len(batch_texts)} texts", "INFO")

        start_time = time.time()
        embeddings = embedder.embed_texts(batch_texts, task="retrieval.query")
        duration = (time.time() - start_time) * 1000

        if embeddings and len(embeddings) == len(batch_texts):
            all_correct = all(len(emb) == 384 for emb in embeddings)
            if all_correct:
                avg_time = duration / len(batch_texts)
                log(f"  ✅ Batch embedding successful: {duration:.1f}ms total, {avg_time:.1f}ms per text", "SUCCESS")
                return True
            else:
                log(f"  ❌ Wrong embedding dimensions in batch", "ERROR")
                return False
        else:
            log(f"  ❌ Batch embedding failed: expected {len(batch_texts)}, got {len(embeddings) if embeddings else 0}", "ERROR")
            return False

    except Exception as e:
        log(f"  ❌ Batch embedding test failed: {e}", "ERROR")
        return False

def test_search_service():
    """Test vector search service integration."""
    log("Testing vector search service integration...", "TEST")

    try:
        from src.services.search.vector_search import VectorSearchService

        # Initialize search service
        search_service = VectorSearchService()

        # Test query embedding (internal method)
        test_query = "definisi badan hukum"
        log(f"  Testing query embedding: '{test_query}'", "INFO")

        start_time = time.time()
        try:
            embedding = search_service._embed_query(test_query)
            duration = (time.time() - start_time) * 1000

            if embedding and len(embedding) == 384:
                log(f"  ✅ Search service embedding: {duration:.1f}ms", "SUCCESS")
                return True
            else:
                log(f"  ❌ Search service embedding failed", "ERROR")
                return False
        except Exception as e:
            log(f"  ❌ Search service embedding failed: {e}", "ERROR")
            return False

    except Exception as e:
        log(f"  ❌ Search service test failed: {e}", "ERROR")
        return False

def run_performance_test():
    """Run a quick performance test."""
    log("Running performance test...", "TEST")

    try:
        from src.services.embedding.embedder import JinaV4Embedder
        embedder = JinaV4Embedder()

        # Test multiple embeddings to check consistency
        test_text = "definisi badan hukum dalam peraturan perundang-undangan"
        durations = []

        for i in range(3):
            start_time = time.time()
            embedding = embedder.embed_single(test_text, task="retrieval.query")
            duration = (time.time() - start_time) * 1000
            durations.append(duration)

            if not embedding or len(embedding) != 384:
                log(f"  ❌ Performance test failed on attempt {i+1}", "ERROR")
                return False

        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        log(f"  ✅ Performance: avg={avg_duration:.1f}ms, min={min_duration:.1f}ms, max={max_duration:.1f}ms", "SUCCESS")

        if avg_duration < 30000:  # Less than 30 seconds average
            log(f"  ✅ Performance acceptable (under 30s average)", "SUCCESS")
            return True
        else:
            log(f"  ⚠️ Performance slow but working ({avg_duration/1000:.1f}s average)", "WARNING")
            return True

    except Exception as e:
        log(f"  ❌ Performance test failed: {e}", "ERROR")
        return False

def main():
    """Run all tests."""
    log("🧪 TESTING JINA API EMERGENCY FIX", "TEST")
    log("=" * 50)

    tests = [
        ("Environment Configuration", test_environment),
        ("Timeout Settings", test_timeout_settings),
        ("Basic Embedding", test_basic_embedding),
        ("Legal Queries", test_legal_queries),
        ("Batch Embedding", test_batch_embedding),
        ("Search Service Integration", test_search_service),
        ("Performance Test", run_performance_test)
    ]

    results = []

    for test_name, test_func in tests:
        log(f"\n🧪 Running: {test_name}", "TEST")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                log(f"✅ {test_name}: PASSED", "SUCCESS")
            else:
                log(f"❌ {test_name}: FAILED", "ERROR")
        except Exception as e:
            log(f"❌ {test_name}: ERROR - {e}", "ERROR")
            results.append((test_name, False))

    # Summary
    log("\n" + "=" * 50)
    log("📊 TEST SUMMARY", "INFO")
    log("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    log(f"Tests Passed: {passed}/{total}", "SUCCESS" if passed == total else "WARNING")

    if passed < total:
        log("\nFailed Tests:", "ERROR")
        for test_name, result in results:
            if not result:
                log(f"  ❌ {test_name}", "ERROR")

    # Recommendations
    if passed == total:
        log("\n🎉 ALL TESTS PASSED!", "SUCCESS")
        log("Jina API emergency fix is working correctly.", "SUCCESS")
        log("You can now restart your API server and test contextual searches.", "INFO")
    elif passed >= 4:  # Most tests passed
        log("\n⚠️ MOSTLY WORKING", "WARNING")
        log("Core functionality is working but some issues remain.", "WARNING")
        log("You can proceed but monitor for errors.", "INFO")
    else:
        log("\n❌ EMERGENCY FIX NOT WORKING", "ERROR")
        log("Consider these solutions:", "ERROR")
        log("1. Check your JINA_API_KEY is valid", "ERROR")
        log("2. Increase timeout: EMBEDDING_REQUEST_TIMEOUT=120", "ERROR")
        log("3. Check network connectivity", "ERROR")
        log("4. Check Jina API status: https://status.jina.ai/", "ERROR")

    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        log("Test interrupted by user", "WARNING")
        sys.exit(130)
    except Exception as e:
        log(f"Test script failed: {e}", "ERROR")
        sys.exit(1)
