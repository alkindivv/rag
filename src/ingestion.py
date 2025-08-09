#!/usr/bin/env python3
"""Simple Document Ingestion - Direct orchestrator calls, no pipeline abstractions"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.services.crawler.crawler_orchestrator import CrawlerOrchestrator
from src.services.pdf import PDFOrchestrator
from src.config.crawler_config import CrawlerConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for ingestion."""
    parser = argparse.ArgumentParser(description="Simple Document Ingestion")
    parser.add_argument("--query", default="", help="Search query")
    parser.add_argument("--type", default="", help="Type of regulation")
    parser.add_argument("--year", default="", help="Year of regulation")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of documents")
    parser.add_argument("--output-dir", default="./data/json", help="Output directory for JSON files")
    parser.add_argument("--pdf-dir", default="./data/pdfs", help="Output directory for PDF files")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF processing")
    parser.add_argument("--txt-dir", help="Directory containing TXT files for batch processing")
    parser.add_argument(
        "--process-txt-batch",
        action="store_true",
        help="Process all TXT files in txt-dir that match existing JSON metadata",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


async def process_pdf_documents(
    crawler: CrawlerOrchestrator, pdf_processor: PDFOrchestrator, documents: List[Dict[str, Any]]
) -> int:
    """Process documents with available PDFs concurrently."""

    async def _process(doc: Dict[str, Any]) -> None:
        nonlocal processed_count
        if not doc.get("pdf_path"):
            return
        enhanced = await asyncio.to_thread(pdf_processor.process_document_complete, doc)
        if enhanced.get("doc_content"):
            await crawler.save_regulation_json(enhanced, update_existing=True)
            crawler.save_text_file(enhanced)
            doc.update(enhanced)
            processed_count += 1

    processed_count = 0
    await asyncio.gather(*(_process(d) for d in documents))
    return processed_count


async def run_ingestion(args: argparse.Namespace) -> None:
    """Execute the ingestion pipeline based on parsed arguments."""
    config = CrawlerConfig()
    config.setup_logging("DEBUG" if args.verbose else "INFO")

    try:
        if args.process_txt_batch and args.txt_dir:
            processed = await process_txt_batch(args.txt_dir, args.output_dir, args.pdf_dir)
            logger.info("BATCH TXT PROCESSING RESULTS")
            logger.info("TXT files processed: %d", processed)
            return

        crawler = CrawlerOrchestrator(output_dir=args.output_dir, pdf_dir=args.pdf_dir)
        documents = await crawler.crawl_regulations(
            query=args.query, regulation_type=args.type, year=args.year, limit=args.limit
        )

        logger.info("INGESTION RESULTS")
        logger.info("Documents crawled: %d", len(documents))

        pdfs_processed = 0
        if not args.no_pdf:
            pdf_processor = PDFOrchestrator()
            pdfs_processed = await process_pdf_documents(crawler, pdf_processor, documents)

        logger.info("PDFs processed: %d", pdfs_processed)
        logger.info("Processing: %s", "Enabled" if not args.no_pdf else "Disabled")

        if documents:
            for i, doc in enumerate(documents[:3], 1):
                title = doc.get("doc_title", doc.get("title", "Unknown"))
                has_content = "Yes" if doc.get("doc_content", doc.get("content")) else "No"
                logger.info("%d. %s (Content: %s)", i, title, has_content)
            if len(documents) > 3:
                logger.info("... and %d more documents", len(documents) - 3)

        json_files = len(list(Path(args.output_dir).glob("*.json"))) if Path(args.output_dir).exists() else 0
        pdf_files = len(list(Path(args.pdf_dir).glob("*.pdf"))) if Path(args.pdf_dir).exists() else 0
        logger.info("Files: %d JSON, %d PDF", json_files, pdf_files)
    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
    except Exception:
        logger.exception("Ingestion failed")
        sys.exit(1)


async def process_txt_batch(txt_dir: str, json_dir: str, pdf_dir: str) -> int:
    """Batch process TXT files based on filename matching."""
    txt_path = Path(txt_dir)
    json_path = Path(json_dir)

    if not txt_path.exists():
        logger.warning("TXT directory not found: %s", txt_dir)
        return 0

    txt_files = list(txt_path.glob("*.txt"))
    processed_count = 0

    pdf_processor = PDFOrchestrator()

    for txt_file in txt_files:
        try:
            base_name = txt_file.stem
            json_file = json_path / f"{base_name}.json"

            if not json_file.exists():
                logger.warning("No matching JSON found for: %s", txt_file.name)
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    regulation_data = json.load(f)
            except json.JSONDecodeError:
                logger.error("Invalid JSON in %s", json_file)
                continue

            with open(txt_file, "r", encoding="utf-8") as f:
                txt_content = f.read()

            logger.info("Processing: %s", txt_file.name)

            enhanced_doc = pdf_processor.process_txt_content(regulation_data, txt_content)

            from src.services.crawler.crawler_orchestrator import DateTimeEncoder
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(enhanced_doc, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)

            logger.info("Successfully processed: %s", txt_file.name)
            processed_count += 1
        except Exception:
            logger.exception("Error processing %s", txt_file.name)
            continue

    return processed_count


async def main() -> None:
    """Entry point for command line execution."""
    args = parse_args()
    await run_ingestion(args)


if __name__ == "__main__":
    asyncio.run(main())
