#!/usr/bin/env python3
"""
Simple Document Ingestion - Direct orchestrator calls, no pipeline abstractions
"""

import argparse
import asyncio
import sys
import json
from datetime import datetime
from pathlib import Path

from src.services.crawler.crawler_orchestrator import CrawlerOrchestrator
from src.services.pdf import PDFOrchestrator
from src.config.crawler_config import CrawlerConfig


async def main():
    """Simple ingestion using direct orchestrator calls"""
    parser = argparse.ArgumentParser(description='Simple Document Ingestion')
    parser.add_argument('--query', default='', help='Search query')
    parser.add_argument('--type', default='', help='Type of regulation')
    parser.add_argument('--year', default='', help='Year of regulation')
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of documents')
    parser.add_argument('--output-dir', default='./data/json', help='Output directory for JSON files')
    parser.add_argument('--pdf-dir', default='./data/pdfs', help='Output directory for PDF files')
    parser.add_argument('--no-pdf', action='store_true', help='Skip PDF processing')
    parser.add_argument('--txt-dir', help='Directory containing TXT files for batch processing')
    parser.add_argument('--process-txt-batch', action='store_true', help='Process all TXT files in txt-dir that match existing JSON metadata')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    config = CrawlerConfig()
    if args.verbose:
        config.setup_logging("DEBUG")
    else:
        config.setup_logging("INFO")

    try:
        # Check if batch TXT processing is requested
        if args.process_txt_batch and args.txt_dir:
            processed_count = await process_txt_batch(args.txt_dir, args.output_dir, args.pdf_dir)
            print(f"\n{'='*60}")
            print(f"BATCH TXT PROCESSING RESULTS")
            print(f"{'='*60}")
            print(f"TXT files processed: {processed_count}")
            print(f"{'='*60}")
            return
        # Step 1: Crawl documents
        crawler = CrawlerOrchestrator(output_dir=args.output_dir, pdf_dir=args.pdf_dir)
        documents = await crawler.crawl_regulations(
            query=args.query,
            regulation_type=args.type,
            year=args.year,
            limit=args.limit
        )

        print(f"\n{'='*60}")
        print(f"INGESTION RESULTS")
        print(f"{'='*60}")
        print(f"Documents crawled: {len(documents)}")

        # Step 2: Process PDFs if enabled
        pdfs_processed = 0
        if not args.no_pdf:
            pdf_processor = PDFOrchestrator()

            for doc in documents:
                if doc.get('pdf_path'):
                    enhanced_doc = pdf_processor.process_document_complete(doc)
                    if enhanced_doc.get('doc_content'):
                        # Update files
                        await crawler.save_regulation_json(enhanced_doc, update_existing=True)
                        crawler.save_text_file(enhanced_doc)
                        pdfs_processed += 1
                        doc.update(enhanced_doc)

        print(f"PDFs processed: {pdfs_processed}")
        print(f"Processing: {'Enabled' if not args.no_pdf else 'Disabled'}")
        print(f"{'='*60}")

        # Show sample documents
        if documents:
            print(f"\nProcessed documents:")
            for i, doc in enumerate(documents[:3], 1):
                title = doc.get('doc_title', doc.get('title', 'Unknown'))
                has_content = 'Yes' if doc.get('doc_content', doc.get('content')) else 'No'
                print(f"{i}. {title} (Content: {has_content})")

            if len(documents) > 3:
                print(f"... and {len(documents) - 3} more documents")

        # File summary
        json_files = len(list(Path(args.output_dir).glob("*.json"))) if Path(args.output_dir).exists() else 0
        pdf_files = len(list(Path(args.pdf_dir).glob("*.pdf"))) if Path(args.pdf_dir).exists() else 0
        print(f"\nFiles: {json_files} JSON, {pdf_files} PDF")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("Ingestion interrupted by user")
    except Exception as e:
        print(f"Ingestion failed: {str(e)}")
        sys.exit(1)


async def process_txt_batch(txt_dir: str, json_dir: str, pdf_dir: str) -> int:
    """Batch process TXT files berdasarkan filename matching"""
    txt_path = Path(txt_dir)
    json_path = Path(json_dir)

    if not txt_path.exists():
        print(f"TXT directory not found: {txt_dir}")
        return 0

    # Get all TXT files
    txt_files = list(txt_path.glob("*.txt"))
    processed_count = 0

    pdf_processor = PDFOrchestrator()

    for txt_file in txt_files:
        try:
            # Extract base filename (without extension)
            base_name = txt_file.stem  # undang_undang_1_2025

            # Look for matching JSON file
            json_file = json_path / f"{base_name}.json"

            if not json_file.exists():
                print(f"⚠️  No matching JSON found for: {txt_file.name}")
                continue

            # Load existing JSON metadata
            with open(json_file, 'r', encoding='utf-8') as f:
                regulation_data = json.load(f)

            # Read TXT content
            with open(txt_file, 'r', encoding='utf-8') as f:
                txt_content = f.read()

            print(f"🔄 Processing: {txt_file.name}")

            # Process TXT content (build tree, etc)
            enhanced_doc = pdf_processor.process_txt_content(regulation_data, txt_content)

            # Save updated JSON
            from src.services.crawler.crawler_orchestrator import DateTimeEncoder
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_doc, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)

            print(f"✅ Successfully processed: {txt_file.name}")
            processed_count += 1

        except Exception as e:
            print(f"❌ Error processing {txt_file.name}: {str(e)}")
            continue

    return processed_count


if __name__ == "__main__":
    asyncio.run(main())
