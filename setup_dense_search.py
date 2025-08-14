#!/usr/bin/env python3
"""
Dense Search Setup and Validation Script for Legal RAG System.

This script sets up and validates the new dense search implementation that replaces
the hybrid FTS+vector approach with pure vector search and citation parsing.

Run with: python setup_dense_search.py
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class DenseSearchSetup:
    """Setup and validation for dense search system."""

    def __init__(self):
        """Initialize setup with project paths."""
        self.project_root = Path(__file__).parent
        self.src_path = self.project_root / "src"
        self.setup_results = {}

        # Add src to Python path
        sys.path.insert(0, str(self.src_path))

    def print_header(self, title: str, char: str = "=") -> None:
        """Print formatted header."""
        print(f"\n{char * 60}")
        print(f"{title:^60}")
        print(f"{char * 60}")

    def print_step(self, step: str, status: str = "") -> None:
        """Print step with optional status."""
        if status:
            print(f"📋 {step}: {status}")
        else:
            print(f"📋 {step}")

    def run_command(self, command: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str]:
        """Run shell command and return success status and output."""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        self.print_step("Checking Python version")

        version = sys.version_info
        if version.major == 3 and version.minor >= 9:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
            return True
        else:
            print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)")
            return False

    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        self.print_step("Checking dependencies")

        required_packages = [
            "sqlalchemy",
            "alembic",
            "fastapi",
            "uvicorn",
            "psycopg2",
            "pgvector",
            "pydantic",
            "requests",
            "pytest"
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package}")
                missing_packages.append(package)

        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Install with: pip install -r requirements.txt")
            return False

        print("✅ All dependencies installed")
        return True

    def check_database_connection(self) -> bool:
        """Check database connectivity and pgvector extension."""
        self.print_step("Checking database connection")

        try:
            from src.config.settings import settings
            from src.db.session import get_db_session
            from sqlalchemy import text

            with get_db_session() as db:
                # Test basic connectivity
                result = db.execute(text("SELECT 1")).scalar()
                print("  ✅ Database connection successful")

                # Check pgvector extension
                try:
                    db.execute(text("SELECT vector_dims('[1,2,3]'::vector)")).scalar()
                    print("  ✅ pgvector extension available")
                except Exception:
                    print("  ❌ pgvector extension not found")
                    print("     Install with: CREATE EXTENSION vector;")
                    return False

                return True

        except Exception as e:
            print(f"  ❌ Database connection failed: {e}")
            print("     Check DATABASE_URL in .env file")
            return False

    def run_migration(self) -> bool:
        """Run database migration to dense search schema."""
        self.print_step("Running database migration")

        # Check if migration file exists
        migration_file = self.project_root / "src/db/migrations/versions/001_remove_fts_ordinal_dense_search.py"
        if not migration_file.exists():
            print("  ❌ Migration file not found")
            return False

        # Run migration
        success, output = self.run_command(["alembic", "upgrade", "head"])

        if success:
            print("  ✅ Migration completed successfully")
            return True
        else:
            print(f"  ❌ Migration failed: {output}")
            return False

    def validate_schema_changes(self) -> bool:
        """Validate that schema changes were applied correctly."""
        self.print_step("Validating schema changes")

        try:
            from src.db.session import get_db_session
            from sqlalchemy import text

            with get_db_session() as db:
                # Check that ordinal columns are removed
                ordinal_check = db.execute(text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'legal_units'
                    AND column_name IN ('ordinal_int', 'ordinal_suffix', 'seq_sort_key', 'content_vector')
                """)).fetchall()

                if ordinal_check:
                    print(f"  ❌ Old columns still exist: {[r[0] for r in ordinal_check]}")
                    return False
                else:
                    print("  ✅ Ordinal and FTS columns removed")

                # Check DocumentVector embedding dimension
                embedding_info = db.execute(text("""
                    SELECT atttypmod
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    WHERE c.relname = 'document_vectors'
                    AND a.attname = 'embedding'
                """)).scalar()

                if embedding_info == 384:
                    print("  ✅ DocumentVector embedding dimension is 384")
                else:
                    print(f"  ❌ DocumentVector embedding dimension is {embedding_info}, expected 384")
                    return False

                # Check HNSW index exists
                hnsw_index = db.execute(text("""
                    SELECT indexname
                    FROM pg_indexes
                    WHERE tablename = 'document_vectors'
                    AND indexname = 'idx_vec_embedding_hnsw'
                """)).scalar()

                if hnsw_index:
                    print("  ✅ HNSW index created")
                else:
                    print("  ❌ HNSW index not found")
                    return False

                return True

        except Exception as e:
            print(f"  ❌ Schema validation failed: {e}")
            return False

    def test_citation_parser(self) -> bool:
        """Test citation parser functionality."""
        self.print_step("Testing citation parser")

        try:
            from src.services.citation import parse_citation, is_explicit_citation, get_best_citation_match

            test_cases = [
                ("UU 8/2019 Pasal 6 ayat (2) huruf b", True, "UU", "8", 2019),
                ("PP No. 45 Tahun 2020 Pasal 12", True, "PP", "45", 2020),
                ("definisi badan hukum", False, None, None, None),
                ("sanksi pidana korupsi", False, None, None, None)
            ]

            passed = 0
            for query, should_be_citation, expected_form, expected_number, expected_year in test_cases:
                is_citation = is_explicit_citation(query, 0.60)

                if is_citation == should_be_citation:
                    if should_be_citation:
                        match = get_best_citation_match(query)
                        if (match and match.doc_form == expected_form and
                            match.doc_number == expected_number and
                            match.doc_year == expected_year):
                            passed += 1
                            print(f"    ✅ '{query[:30]}...'")
                        else:
                            print(f"    ❌ '{query[:30]}...' (incorrect parsing)")
                    else:
                        passed += 1
                        print(f"    ✅ '{query[:30]}...' (correctly not citation)")
                else:
                    print(f"    ❌ '{query[:30]}...' (detection failed)")

            if passed == len(test_cases):
                print("  ✅ Citation parser working correctly")
                return True
            else:
                print(f"  ❌ Citation parser failed {len(test_cases) - passed}/{len(test_cases)} tests")
                return False

        except Exception as e:
            print(f"  ❌ Citation parser test failed: {e}")
            return False

    def test_vector_search_service(self) -> bool:
        """Test vector search service initialization."""
        self.print_step("Testing vector search service")

        try:
            from src.services.search.vector_search import VectorSearchService
            from unittest.mock import Mock

            # Mock embedder to avoid API calls
            mock_embedder = Mock()
            mock_embedder.embed_texts.return_value = [[0.1] * 384]

            # Initialize service
            search_service = VectorSearchService(embedder=mock_embedder)

            # Test routing logic
            citation_query = "UU 8/2019 Pasal 6"
            contextual_query = "definisi badan hukum"

            # Mock the actual search methods to avoid database calls
            search_service._handle_explicit_citation = Mock(return_value=[])
            search_service._handle_contextual_search = Mock(return_value=[])

            # Test citation routing
            result1 = search_service.search(citation_query, k=5)
            if result1['metadata']['search_type'] == 'explicit_citation':
                print("    ✅ Citation query routed correctly")
            else:
                print("    ❌ Citation query routing failed")
                return False

            # Test contextual routing
            result2 = search_service.search(contextual_query, k=5)
            if result2['metadata']['search_type'] == 'contextual_semantic':
                print("    ✅ Contextual query routed correctly")
            else:
                print("    ❌ Contextual query routing failed")
                return False

            print("  ✅ Vector search service working correctly")
            return True

        except Exception as e:
            print(f"  ❌ Vector search service test failed: {e}")
            return False

    def test_natural_sorting(self) -> bool:
        """Test natural sorting utility."""
        self.print_step("Testing natural sorting")

        try:
            from src.utils.natural_sort import natural_sort_strings

            test_input = ['10', '2', '1', '3', '11', '20']
            expected_output = ['1', '2', '3', '10', '11', '20']

            result = natural_sort_strings(test_input)

            if result == expected_output:
                print("  ✅ Natural sorting working correctly")
                return True
            else:
                print(f"  ❌ Natural sorting failed: expected {expected_output}, got {result}")
                return False

        except Exception as e:
            print(f"  ❌ Natural sorting test failed: {e}")
            return False

    def start_api_server(self) -> bool:
        """Start API server for testing."""
        self.print_step("Starting API server")

        try:
            # Check if server is already running
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("  ✅ API server already running")
                    return True
            except:
                pass

            # Start server in background
            import subprocess
            import threading

            def run_server():
                subprocess.run([
                    sys.executable, "-m", "uvicorn",
                    "src.api.main:app",
                    "--host", "0.0.0.0",
                    "--port", "8000"
                ], cwd=self.project_root, capture_output=True)

            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()

            # Wait for server to start
            for i in range(10):
                time.sleep(1)
                try:
                    response = requests.get("http://localhost:8000/health", timeout=1)
                    if response.status_code == 200:
                        print("  ✅ API server started successfully")
                        return True
                except:
                    continue

            print("  ❌ API server failed to start")
            return False

        except Exception as e:
            print(f"  ❌ API server startup failed: {e}")
            return False

    def test_api_endpoints(self) -> bool:
        """Test API endpoints."""
        self.print_step("Testing API endpoints")

        try:
            # Test health endpoint
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code != 200:
                print("    ❌ Health endpoint failed")
                return False
            else:
                print("    ✅ Health endpoint working")

            # Test search endpoint with citation
            search_payload = {
                "query": "UU 8/2019 Pasal 6",
                "limit": 5,
                "use_reranking": False
            }

            response = requests.post(
                "http://localhost:8000/search",
                json=search_payload,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if 'results' in data and 'metadata' in data:
                    print("    ✅ Search endpoint working")
                    print(f"      Search type: {data['metadata'].get('search_type')}")
                else:
                    print("    ❌ Search endpoint returned invalid structure")
                    return False
            else:
                print(f"    ❌ Search endpoint failed with status {response.status_code}")
                return False

            print("  ✅ API endpoints working correctly")
            return True

        except Exception as e:
            print(f"  ❌ API endpoint test failed: {e}")
            return False

    def generate_summary_report(self) -> None:
        """Generate summary report of setup results."""
        self.print_header("SETUP SUMMARY REPORT")

        total_checks = len(self.setup_results)
        passed_checks = sum(self.setup_results.values())

        print(f"Setup Results: {passed_checks}/{total_checks} checks passed\n")

        for check_name, passed in self.setup_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {check_name.replace('_', ' ').title()}: {status}")

        print(f"\nOverall Status: {'✅ SUCCESS' if passed_checks == total_checks else '❌ INCOMPLETE'}")

        if passed_checks == total_checks:
            print("\n🎉 Dense search system is ready for use!")
            print("\nNext steps:")
            print("1. Ingest some documents: python src/ingestion.py --query 'UU 2019' --limit 10")
            print("2. Test search: curl 'http://localhost:8000/search?query=UU%208/2019%20Pasal%206'")
            print("3. Run golden tests: python -m pytest tests/integration/test_golden_ops.py")
        else:
            print("\n⚠️  Setup incomplete. Please resolve failed checks before proceeding.")
            print("\nFailed checks need to be addressed:")
            for check_name, passed in self.setup_results.items():
                if not passed:
                    print(f"  - {check_name.replace('_', ' ').title()}")

    def run_setup(self) -> bool:
        """Run complete setup and validation."""
        self.print_header("LEGAL RAG DENSE SEARCH SETUP")

        print("🚀 Setting up dense semantic search system...")
        print("   Replacing hybrid FTS+vector with pure vector search")
        print("   Adding citation parsing for explicit legal references")
        print("   Configuring 384-dimensional embeddings with HNSW indexing")

        # Run all setup steps
        setup_steps = [
            ("python_version", self.check_python_version),
            ("dependencies", self.check_dependencies),
            ("database_connection", self.check_database_connection),
            ("migration", self.run_migration),
            ("schema_validation", self.validate_schema_changes),
            ("citation_parser", self.test_citation_parser),
            ("vector_search_service", self.test_vector_search_service),
            ("natural_sorting", self.test_natural_sorting),
            ("api_server", self.start_api_server),
            ("api_endpoints", self.test_api_endpoints)
        ]

        for step_name, step_func in setup_steps:
            try:
                self.setup_results[step_name] = step_func()
            except Exception as e:
                print(f"❌ {step_name} failed with error: {e}")
                self.setup_results[step_name] = False

        # Generate summary
        self.generate_summary_report()

        return all(self.setup_results.values())


def main():
    """Main setup execution."""
    setup = DenseSearchSetup()

    try:
        success = setup.run_setup()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
