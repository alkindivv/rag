#!/usr/bin/env python3
"""
Main CLI entry point for Legal RAG system.

Provides commands for indexing documents, testing search, and system management.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.db.session import get_db_session, init_db, reset_db
from src.pipeline.indexer import LegalDocumentIndexer
from src.services.search.vector_search import VectorSearchService, SearchFilters
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def setup_cli_logging(verbose: bool = False) -> None:
    """Set up logging for CLI operations."""
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level, structured=False)  # Human-readable for CLI


def cmd_init_db(args: argparse.Namespace) -> int:
    """Initialize database tables."""
    try:
        logger.info("Initializing database...")

        if args.reset:
            logger.warning("Resetting database (this will delete all data!)")
            if not args.force:
                confirm = input("Are you sure? Type 'yes' to confirm: ")
                if confirm.lower() != 'yes':
                    logger.info("Database reset cancelled")
                    return 0
            reset_db()
            logger.info("Database reset complete")
        else:
            init_db()
            logger.info("Database initialization complete")

        return 0

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return 1


def cmd_index(args: argparse.Namespace) -> int:
    """Index legal documents from JSON files."""
    try:
        logger.info(f"Starting indexing from: {args.input_path}")

        # Initialize indexer
        indexer = LegalDocumentIndexer()

        input_path = Path(args.input_path)

        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            return 1

        # Process input
        if input_path.is_file():
            if not input_path.suffix == '.json':
                logger.error("Input file must be a JSON file")
                return 1

            success = indexer.index_json_file(input_path)
            if not success:
                logger.error("Failed to index file")
                return 1

        elif input_path.is_dir():
            stats = indexer.index_directory(input_path, args.pattern)

            logger.info(f"Indexing completed with stats: {stats}")

            if stats["errors"] > 0:
                logger.warning(f"Indexing completed with {stats['errors']} errors")
                return 1 if not args.continue_on_error else 0

        else:
            logger.error(f"Invalid input path: {input_path}")
            return 1

        logger.info("Indexing completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Test search functionality."""
    try:
        logger.info(f"Testing search with query: '{args.query}'")

        # Initialize search service
        search_service = VectorSearchService()

        # Build filters if provided
        filters = None
        if any([args.doc_forms, args.doc_years, args.doc_numbers]):
            filters = SearchFilters(
                doc_forms=args.doc_forms.split(',') if args.doc_forms else None,
                doc_years=[int(y) for y in args.doc_years.split(',')] if args.doc_years else None,
                doc_numbers=args.doc_numbers.split(',') if args.doc_numbers else None,
            )

        # Perform search
        response = search_service.search(
            query=args.query,
            k=args.limit,
            filters=filters,
            use_reranking=args.rerank,
            session_id="cli_test"
        )

        # Display results
        print(f"\nSearch Results for: '{args.query}'")
        print(f"Search Type: {response['metadata']['search_type']}")
        print(f"Total: {response['metadata']['total_results']} results")
        print(f"Reranked: {response['metadata']['reranking_used']}")
        print(f"Duration: {response['metadata']['duration_ms']:.2f}ms")

        if response['metadata'].get('error'):
            print(f"Error: {response['metadata']['error']}")
            return 1

        print("\nResults:")
        for i, result in enumerate(response["results"], 1):
            print(f"\n{i}. {result.citation_string}")
            print(f"   Document: {result.doc_form} {result.doc_number}/{result.doc_year}")
            print(f"   Score: {result.score:.3f} | Type: {result.unit_type}")

            if args.show_content:
                content = result.content[:200] + "..." if len(result.content) > 200 else result.content
                print(f"   Content: {content}")

        return 0

    except Exception as e:
        logger.error(f"Search test failed: {e}")
        return 1


def cmd_outline(args: argparse.Namespace) -> int:
    """Get document outline."""
    try:
        logger.info(f"Getting outline for document: {args.doc_id}")

        search_service = VectorSearchService()
        # Note: get_document_outline not implemented in VectorSearchService
        # This is a placeholder - would need to be implemented
        response = {"error": "Document outline not implemented in VectorSearchService yet"}

        if response.get('error'):
            logger.error(f"Error: {response['error']}")
            return 1

        doc = response["document"]
        print(f"\nDocument Outline: {doc['title']}")
        print(f"Form: {doc['form']} {doc['number']}/{doc['year']}")
        print(f"Status: {doc['status']}")
        print(f"Total Units: {response['total_units']}")
        print(f"Duration: {response['duration_ms']:.2f}ms")

        print("\nHierarchy:")
        for unit in response["outline"]:
            indent = "  " * (len(unit.get("hierarchy_path", "").split("/")) - 1)
            print(f"{indent}{unit['label']} - {unit['type']}")

            if args.include_content and unit.get('content'):
                content = unit['content'][:100] + "..." if len(unit['content']) > 100 else unit['content']
                print(f"{indent}  Content: {content}")

        return 0

    except Exception as e:
        logger.error(f"Outline retrieval failed: {e}")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Check system status and database health."""
    try:
        logger.info("Checking system status...")

        status = {"database": "unknown", "embedding": "unknown", "errors": []}

        # Check database connection
        try:
            with get_db_session() as db:
                from src.db.models import LegalDocument

                doc_count = db.query(LegalDocument).count()
                status["database"] = "connected"
                status["document_count"] = doc_count

                # Check for recent documents
                recent_docs = db.query(LegalDocument).order_by(
                    LegalDocument.created_at.desc()
                ).limit(5).all()

                status["recent_documents"] = [
                    {
                        "doc_id": doc.doc_id,
                        "title": doc.doc_title[:60] + "..." if len(doc.doc_title) > 60 else doc.doc_title,
                        "status": doc.doc_processing_status,
                        "created": doc.created_at.isoformat() if doc.created_at else None
                    }
                    for doc in recent_docs
                ]

        except Exception as e:
            status["database"] = "error"
            status["errors"].append(f"Database error: {e}")

        # Check embedding service
        try:
            from src.services.embedding.embedder import JinaV4Embedder

            embedder = JinaV4Embedder()
            test_embedding = embedder.embed_single("test")

            if test_embedding and len(test_embedding) == 384:
                status["embedding"] = "working"
                status["embedding_model"] = embedder.model
            else:
                status["embedding"] = "error"
                status["errors"].append("Embedding service returned invalid response")

        except Exception as e:
            status["embedding"] = "error"
            status["errors"].append(f"Embedding error: {e}")

        # Display status
        print("\nSystem Status:")
        print(f"Database: {status['database']}")
        print(f"Embedding: {status['embedding']}")

        if status.get("document_count") is not None:
            print(f"Documents indexed: {status['document_count']}")

        if status.get("embedding_model"):
            print(f"Embedding model: {status['embedding_model']}")

        if status.get("recent_documents"):
            print("\nRecent Documents:")
            for doc in status["recent_documents"]:
                print(f"  - {doc['doc_id']}: {doc['title']} ({doc['status']})")

        if status["errors"]:
            print("\nErrors:")
            for error in status["errors"]:
                print(f"  - {error}")
            return 1

        print("\nAll systems operational ✓")
        return 0

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run search benchmarks."""
    try:
        logger.info("Running search benchmarks...")

        search_service = VectorSearchService()

        # Test queries
        test_queries = [
            "pasal 1",
            "UU 4/2009",
            "pertambangan mineral",
            "batubara dan energi",
            "hilirisasi industri",
            "izin usaha pertambangan",
            "lingkungan hidup",
            "pajak dan retribusi"
        ]

        results = []

        for query in test_queries:
            start_time = time.time()

            response = search_service.search(
                query=query,
                k=10,
                use_reranking=args.rerank
            )

            duration = (time.time() - start_time) * 1000

            results.append({
                "query": query,
                "search_type": response["metadata"]["search_type"],
                "results_count": response["metadata"]["total_results"],
                "duration_ms": duration,
                "reranked": response["metadata"]["reranking_used"]
            })

            print(f"Query: '{query}' -> {response['metadata']['total_results']} results in {duration:.2f}ms ({response['metadata']['search_type']})")

        # Summary statistics
        avg_duration = sum(r["duration_ms"] for r in results) / len(results)
        total_results = sum(r["results_count"] for r in results)

        print("\nBenchmark Summary:")
        print(f"Queries tested: {len(test_queries)}")
        print(f"Average duration: {avg_duration:.2f}ms")
        print(f"Total results found: {total_results}")
        print(f"Reranking enabled: {args.rerank}")

        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Legal RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize database
  python -m src.main init-db

  # Index all JSON files in data/json
  python -m src.main index data/json

  # Index specific file
  python -m src.main index data/json/undang_undang_2_2025.json

  # Test search
  python -m src.main search "pasal 1 ayat 2"

  # Test search with filters
  python -m src.main search "pertambangan" --doc-forms UU --doc-years 2009,2025

  # Get document outline
  python -m src.main outline UU-2025-2

  # Check system status
  python -m src.main status

  # Run benchmarks
  python -m src.main benchmark --rerank
        """
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init DB command
    init_parser = subparsers.add_parser("init-db", help="Initialize database")
    init_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset database (delete all data)"
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reset without confirmation"
    )

    # Index command
    index_parser = subparsers.add_parser("index", help="Index legal documents")
    index_parser.add_argument(
        "input_path",
        help="Path to JSON file or directory containing JSON files"
    )
    index_parser.add_argument(
        "--pattern",
        default="*.json",
        help="File pattern for directory processing (default: *.json)"
    )
    index_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing even if some files fail"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Test search functionality")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results (default: 10)"
    )
    search_parser.add_argument(
        "--doc-forms",
        help="Comma-separated list of document forms (UU,PP,etc.)"
    )
    search_parser.add_argument(
        "--doc-years",
        help="Comma-separated list of document years"
    )
    search_parser.add_argument(
        "--doc-numbers",
        help="Comma-separated list of document numbers"
    )

    search_parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Disable reranking"
    )
    search_parser.add_argument(
        "--show-content",
        action="store_true",
        help="Show content preview in results"
    )

    # Outline command
    outline_parser = subparsers.add_parser("outline", help="Get document outline")
    outline_parser.add_argument("doc_id", help="Document ID (e.g., UU-2025-2)")
    outline_parser.add_argument(
        "--include-content",
        action="store_true",
        help="Include content in outline"
    )

    # Status command
    subparsers.add_parser("status", help="Check system status")
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run search benchmarks")
    benchmark_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable reranking in benchmarks"
    )

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    setup_cli_logging(args.verbose)

    # Route to appropriate command
    if args.command == "init-db":
        return cmd_init_db(args)
    elif args.command == "index":
        return cmd_index(args)
    elif args.command == "search":
        return cmd_search(args)
    elif args.command == "outline":
        return cmd_outline(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
