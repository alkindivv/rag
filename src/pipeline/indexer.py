"""
Comprehensive indexer for Legal RAG system.

Processes JSON documents from crawler/PDF output and stores in database with embeddings.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..config.settings import settings
from ..db.models import (
    DocForm,
    DocStatus,
    LegalDocument,
    LegalUnit,
    Subject,
    UnitType,
)
from ..db.session import get_db_session
from src.services.embedding.embedder import JinaV4Embedder, ConfigError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class LegalDocumentIndexer:
    """
    Indexes legal documents from JSON into database with embeddings.

    Processes crawler/PDF output JSON files and creates:
    - LegalDocument records with metadata
    - LegalUnit records for document hierarchy
    - DocumentVector records with embeddings for pasal-level content
    - Subject associations
    """

    def __init__(self, embedder: Optional[JinaV4Embedder] = None):
        """
        Initialize indexer.

        Args:
            embedder: Optional embedder instance. Creates new one if None.
        """
        self.embedder = embedder or JinaV4Embedder()
        self.stats = {
            "documents_processed": 0,
            "units_created": 0,
            "embeddings_created": 0,
            "errors": 0,
        }

    def index_json_file(self, json_path: Path) -> bool:
        """
        Index a single JSON file.

        Args:
            json_path: Path to JSON file to process

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Processing JSON file: {json_path}")

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            with get_db_session() as db:
                success = self._process_document(db, data, json_path)
                if success:
                    self.stats["documents_processed"] += 1
                    logger.info(f"Successfully processed: {json_path}")
                else:
                    self.stats["errors"] += 1
                    logger.error(f"Failed to process: {json_path}")

                return success

        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
            self.stats["errors"] += 1
            return False

    def index_directory(self, json_dir: Path, pattern: str = "*.json") -> Dict[str, Any]:
        """
        Index all JSON files in a directory.

        Args:
            json_dir: Directory containing JSON files
            pattern: File pattern to match (default: "*.json")

        Returns:
            Dict with processing statistics
        """
        logger.info(f"Indexing directory: {json_dir} with pattern: {pattern}")

        json_files = list(json_dir.glob(pattern))
        if not json_files:
            logger.warning(f"No files found in {json_dir} matching {pattern}")
            return self.stats

        total_files = len(json_files)
        logger.info(f"Found {total_files} files to process")

        for i, json_path in enumerate(json_files, 1):
            logger.info(f"Processing file {i}/{total_files}: {json_path.name}")
            self.index_json_file(json_path)

        logger.info(f"Indexing complete. Stats: {self.stats}")
        return self.stats

    def _process_document(self, db: Session, data: Dict[str, Any], source_file: Path) -> bool:
        """
        Process a single document from JSON data.

        Args:
            db: Database session
            data: JSON document data
            source_file: Source file path for logging

        Returns:
            bool: True if successful
        """
        try:
            # Check if document already exists
            existing_doc = db.query(LegalDocument).filter_by(
                doc_id=data.get("doc_id")
            ).first()

            if existing_doc:
                logger.info(f"Document {data.get('doc_id')} already exists, skipping")
                return True

            # Create document record
            doc = self._create_document_record(data)
            db.add(doc)
            db.flush()  # Get the ID for foreign keys

            # Process subjects
            if data.get("doc_subject"):
                self._process_subjects(db, doc, data["doc_subject"])

            # Process document tree
            if data.get("document_tree"):
                units, pasal_units = self._process_document_tree(
                    db, doc, data["document_tree"]
                )
                self.stats["units_created"] += len(units)

                # Create embeddings on legal_units.embedding (granular)
                embeddings_created = self._create_embeddings(db, doc, units)
                self.stats["embeddings_created"] += embeddings_created

            return True

        except Exception as e:
            logger.error(f"Error processing document from {source_file}: {e}")
            db.rollback()
            return False

    def _create_document_record(self, data: Dict[str, Any]) -> LegalDocument:
        """Create LegalDocument record from JSON data."""

        # Parse enum values
        # Normalize document form and status with graceful fallback
        try:
            doc_form = DocForm(data.get("doc_form", "LAINNYA").upper())
        except ValueError:
            logger.warning(
                f"Unknown doc_form '{data.get('doc_form')}' - defaulting to LAINNYA"
            )
            doc_form = DocForm.LAINNYA

        try:
            doc_status = DocStatus(data.get("doc_status", "Berlaku"))
        except ValueError:
            logger.warning(
                f"Unknown doc_status '{data.get('doc_status')}' - defaulting to BERLAKU"
            )
            doc_status = DocStatus.BERLAKU

        # Parse dates
        date_fields = ["doc_date_enacted", "doc_date_promulgated", "doc_date_effective"]
        dates = {}
        for field in date_fields:
            date_str = data.get(field)
            if date_str:
                try:
                    dates[field] = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    logger.warning(f"Invalid date format for {field}: {date_str}")
                    dates[field] = None

        # Create document record
        doc = LegalDocument(
            doc_source=data.get("doc_source", ""),
            doc_type=data.get("doc_type", ""),
            doc_title=data.get("doc_title", ""),
            doc_id=data.get("doc_id"),
            doc_number=data.get("doc_number", ""),
            doc_form=doc_form,
            doc_form_short=data.get("doc_form_short", ""),
            doc_year=int(data.get("doc_year", 0)),
            doc_teu=data.get("doc_teu"),
            doc_place_enacted=data.get("doc_place_enacted"),
            doc_language=data.get("doc_language", "Bahasa Indonesia"),
            doc_location=data.get("doc_location"),
            doc_field=data.get("doc_field"),
            doc_relationships=data.get("relationships"),
            doc_uji_materi=data.get("uji_materi"),
            doc_status=doc_status,
            doc_detail_url=data.get("detail_url"),
            doc_source_url=data.get("source_url"),
            doc_pdf_url=data.get("pdf_url"),
            doc_uji_materi_pdf_url=data.get("uji_materi_pdf_url"),
            doc_pdf_path=data.get("pdf_path"),
            doc_text_path=data.get("text_path"),
            doc_content=data.get("doc_content"),
            doc_content_length=len(data.get("doc_content") or ""),
            doc_processing_status=data.get("doc_processing_status", "indexed"),
            doc_last_updated=datetime.now(),
            **dates
        )

        return doc

    def _process_subjects(self, db: Session, doc: LegalDocument, subjects: List[str]) -> None:
        """Process and associate subjects with document."""
        for subject_name in subjects:
            if not subject_name.strip():
                continue

            # Get or create subject
            subject = db.query(Subject).filter_by(name=subject_name).first()
            if not subject:
                subject = Subject(name=subject_name)
                db.add(subject)
                db.flush()

            # Associate with document
            if subject not in doc.subjects:
                doc.subjects.append(subject)

    def _process_document_tree(
        self,
        db: Session,
        doc: LegalDocument,
        tree: Dict[str, Any]
    ) -> Tuple[List[LegalUnit], List[LegalUnit]]:
        """
        Process document tree recursively.

        Returns:
            Tuple of (all_units_created, pasal_units_for_embedding)
        """
        units = []
        pasal_units = []  # pasal-level units for vector embedding
        processed_unit_ids: set[str] = set()

        def process_node(
            node: Dict[str, Any],
            parent_db_id: Optional[str] = None,
        ) -> None:
            """Recursively process tree nodes."""

            # Map type to enum
            unit_type_map = {
                "dokumen": UnitType.DOKUMEN,
                "buku": UnitType.BUKU,
                "bab": UnitType.BAB,
                "bagian": UnitType.BAGIAN,
                "paragraf": UnitType.PARAGRAF,
                "pasal": UnitType.PASAL,
                "ayat": UnitType.AYAT,
                "huruf": UnitType.HURUF,
                "angka": UnitType.ANGKA,
            }

            node_type = node.get("type", "dokumen")
            unit_type = unit_type_map.get(node_type, UnitType.DOKUMEN)

            unit_id = node.get("unit_id", "")
            if unit_id in processed_unit_ids:
                logger.warning(f"Duplicate unit_id detected: {unit_id}; skipping")
                return
            processed_unit_ids.add(unit_id)

            # Pure-consumer: use orchestrator fields as-is
            # Content: prefer orchestrator 'content' for pasal, else 'local_content'
            content_value = node.get("content") or node.get("local_content")

            unit = LegalUnit(
                document_id=doc.id,
                unit_id=unit_id,
                unit_type=unit_type,
                number_label=node.get("number_label"),
                label_display=node.get("label_display"),
                title=node.get("title"),
                content=content_value,
                local_content=node.get("local_content"),
                display_text=node.get("display_text"),
                # bm25_body deprecated: do not populate
                citation_string=node.get("citation_string"),
                # Legacy parent_* fields are deprecated: do not write
            )

            # Persist unit_path as ltree using server-side function
            unit_path_str = node.get("unit_path")
            if unit_path_str:
                unit.unit_path = func.text2ltree(unit_path_str)

            # Parent link: assign actual parent's DB UUID if provided
            if parent_db_id:
                unit.parent_unit_id = parent_db_id

            db.add(unit)
            units.append(unit)

            # Collect pasal units for embedding only
            if unit_type == UnitType.PASAL:
                pasal_units.append(unit)

            for child in node.get("children", []):
                process_node(child, unit.id)

        # Start processing from root
        if tree.get("children"):
            for child in tree["children"]:
                process_node(child)
        else:
            # Process root if no children
            process_node(tree)

        # Flush to ensure all units are saved
        db.flush()

        return units, pasal_units

    # Removed all in-Python aggregation/formatting helpers to enforce pure-consumer behavior

    def _create_embeddings(
        self,
        db: Session,
        doc: LegalDocument,
        units: List[LegalUnit]
    ) -> int:
        """
        Create vector embeddings directly on legal_units.embedding (384-dim).

        Args:
            db: Database session
            doc: Document record
            units: List of LegalUnit objects (all granularities)

        Returns:
            Number of embeddings created
        """
        if not units:
            return 0

        try:
            # Prepare content per granular rule
            unit_db_ids: List[str] = []  # store ORM pk id for update
            contents: List[str] = []

            def is_skip_text(text: Optional[str]) -> bool:
                if not text:
                    return True
                stripped = text.strip()
                if not stripped:
                    return True
                # Skip boilerplate markers
                return stripped.lower() in {"dihapus.", "diubah.", "dicabut."}

            for unit in units:
                # Choose content based on unit_type
                if unit.unit_type in {UnitType.AYAT, UnitType.HURUF, UnitType.ANGKA}:
                    chosen = unit.local_content
                else:
                    chosen = unit.content

                if is_skip_text(chosen):
                    continue

                unit_db_ids.append(unit.id)
                contents.append(chosen)

            if not contents:
                logger.warning("No valid unit content found for embedding")
                return 0

            # Get embeddings using passage task (384 dims)
            embeddings = self.embedder.embed_texts(contents, task="retrieval.passage", dims=384)

            created = 0
            for unit_pk, content, embedding in zip(unit_db_ids, contents, embeddings):
                if embedding is None or len(embedding) != 384:
                    logger.warning(f"Failed to create valid 384-dim embedding for unit {unit_pk}")
                    continue
                # Update in-place
                db.query(LegalUnit).filter(LegalUnit.id == unit_pk).update({
                    LegalUnit.embedding: embedding
                }, synchronize_session=False)
                created += 1

            logger.info(f"Created {created} embeddings for document {doc.doc_id}")
            return created

        except Exception as e:
            logger.error(f"Error creating embeddings for document {doc.doc_id}: {e}")
            return 0

    # Removed hierarchy extraction helper; no longer needed after removing DocumentVector


def main():
    """CLI entry point for indexing."""
    import argparse

    parser = argparse.ArgumentParser(description="Index legal documents from JSON")
    parser.add_argument(
        "input_path",
        help="Path to JSON file or directory containing JSON files"
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="File pattern for directory processing (default: *.json)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=getattr(logging, args.log_level))

    # Create indexer
    indexer = LegalDocumentIndexer()

    # Process input
    input_path = Path(args.input_path)

    if input_path.is_file():
        success = indexer.index_json_file(input_path)
        if success:
            print(f"Successfully indexed: {input_path}")
        else:
            print(f"Failed to index: {input_path}")
            exit(1)
    elif input_path.is_dir():
        stats = indexer.index_directory(input_path, args.pattern)
        print(f"Indexing complete: {stats}")
        if stats["errors"] > 0:
            exit(1)
    else:
        print(f"Invalid path: {input_path}")
        exit(1)


if __name__ == "__main__":
    main()
