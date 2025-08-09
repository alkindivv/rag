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

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..config.settings import settings
from ..db.models import (
    DocForm,
    DocStatus,
    DocumentVector,
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
            "vectors_created": 0,
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
                units, pasal_contents = self._process_document_tree(
                    db, doc, data["document_tree"]
                )
                self.stats["units_created"] += len(units)

                # Create embeddings for pasal-level content
                vectors_created = self._create_embeddings(db, doc, pasal_contents)
                self.stats["vectors_created"] += vectors_created

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
    ) -> Tuple[List[LegalUnit], Dict[str, str]]:
        """
        Process document tree recursively.

        Returns:
            Tuple of (units_created, pasal_content_dict)
        """
        units = []
        pasal_contents = {}  # unit_id -> full_content for embedding
        processed_unit_ids: set[str] = set()

        def process_node(node: Dict[str, Any], parent_pasal_id: Optional[str] = None) -> None:
            """Recursively process tree nodes."""
            logger.debug(f"Processing node: type={node.get('type')}, unit_id={node.get('unit_id')}")

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
                "angka_amandement": UnitType.ANGKA_AMANDEMENT,
            }

            node_type = node.get("type", "dokumen")
            unit_type = unit_type_map.get(node_type, UnitType.DOKUMEN)

            unit_id = node.get("unit_id", "")
            if unit_id in processed_unit_ids:
                logger.warning(f"Duplicate unit_id detected: {unit_id}; skipping")
                return
            processed_unit_ids.add(unit_id)

            # Create unit
            unit = LegalUnit(
                document_id=doc.id,
                unit_type=unit_type,
                unit_id=unit_id,
                number_label=node.get("number_label"),
                ordinal_int=node.get("ordinal_int", 0),
                ordinal_suffix=node.get("ordinal_suffix", ""),
                label_display=node.get("label_display"),
                seq_sort_key=node.get("seq_sort_key"),
                title=node.get("title"),
                content=self._build_full_content(node),
                local_content=node.get("local_content"),
                display_text=node.get("display_text"),
                bm25_body=node.get("bm25_body") or node.get("local_content", ""),
                path=node.get("path"),
                citation_string=node.get("citation_string"),
                parent_pasal_id=parent_pasal_id,
                hierarchy_path=self._build_hierarchy_path(node.get("path", [])),
            )

            db.add(unit)
            units.append(unit)

            # If this is a pasal, collect content for embedding
            current_pasal_id = parent_pasal_id
            if unit_type == UnitType.PASAL:
                current_pasal_id = unit.unit_id
                pasal_contents[unit.unit_id] = unit.content or ""

            # Process children
            for child in node.get("children", []):
                process_node(child, current_pasal_id)

        # Start processing from root
        logger.debug(f"Processing document tree: {tree.get('unit_id')} with {len(tree.get('children', []))} children")
        if tree.get("children"):
            for i, child in enumerate(tree["children"]):
                logger.debug(f"Processing child {i}: {child.get('type')} {child.get('unit_id')}")
                process_node(child)
        else:
            # Process root if no children
            logger.debug("Processing root node (no children)")
            process_node(tree)

        # Update FTS vectors
        db.flush()
        self._update_fts_vectors(db, units)

        return units, pasal_contents

    def _build_full_content(self, node: Dict[str, Any]) -> str:
        """Build full content including children."""
        content_parts = []

        # Add local content
        if node.get("local_content"):
            content_parts.append(node["local_content"])

        # Add children content recursively
        for child in node.get("children", []):
            child_content = self._build_full_content(child)
            if child_content:
                content_parts.append(child_content)

        return "\n\n".join(content_parts)

    def _build_hierarchy_path(self, path: List[Dict[str, Any]]) -> str:
        """Build text hierarchy path from path array."""
        if not path:
            return ""

        path_parts = []
        for item in path:
            if item.get("label"):
                path_parts.append(item["label"])

        return " / ".join(path_parts)

    def _update_fts_vectors(self, db: Session, units: List[LegalUnit]) -> None:
        """Update full-text search vectors for units."""
        for unit in units:
            if unit.bm25_body:
                # Update content_vector using PostgreSQL's to_tsvector
                update_query = text("""
                    UPDATE legal_units
                    SET content_vector = to_tsvector('indonesian', :content)
                    WHERE id = :unit_id
                """)
                db.execute(update_query, {
                    "content": unit.bm25_body,
                    "unit_id": unit.id
                })

    def _create_embeddings(
        self,
        db: Session,
        doc: LegalDocument,
        pasal_contents: Dict[str, str]
    ) -> int:
        """
        Create vector embeddings for pasal-level content.

        Args:
            db: Database session
            doc: Document record
            pasal_contents: Dict mapping unit_id to content

        Returns:
            Number of vectors created
        """
        if not pasal_contents:
            return 0

        try:
            # Prepare content for embedding
            unit_ids = list(pasal_contents.keys())
            contents = list(pasal_contents.values())

            # Get embeddings using passage task for document content
            embeddings = self.embedder.embed_passages(contents, dims=settings.embedding_dim)

            vectors_created = 0
            for unit_id, content, embedding in zip(unit_ids, contents, embeddings):
                if embedding is None or len(embedding) != settings.embedding_dim:
                    logger.warning(f"Failed to create valid embedding for {unit_id}")
                    continue

                # Extract hierarchy info from unit_id
                hierarchy_info = self._extract_hierarchy_info(unit_id)

                vector = DocumentVector(
                    document_id=doc.id,
                    unit_id=unit_id,
                    content_type="pasal",
                    embedding=embedding,
                    embedding_model=self.embedder.model,
                    doc_form=doc.doc_form,
                    doc_year=doc.doc_year,
                    doc_number=doc.doc_number,
                    doc_status=doc.doc_status,
                    hierarchy_path=hierarchy_info.get("hierarchy_path", ""),
                    pasal_number=hierarchy_info.get("pasal_number"),
                    bab_number=hierarchy_info.get("bab_number"),
                    ayat_number=hierarchy_info.get("ayat_number"),
                    token_count=len(content.split()),
                    char_count=len(content),
                )

                db.add(vector)
                vectors_created += 1

            logger.info(f"Created {vectors_created} embeddings for document {doc.doc_id}")
            return vectors_created

        except Exception as e:
            logger.error(f"Error creating embeddings for document {doc.doc_id}: {e}")
            return 0

    def _extract_hierarchy_info(self, unit_id: str) -> Dict[str, Optional[str]]:
        """Extract hierarchy information from unit_id."""
        info = {
            "hierarchy_path": "",
            "pasal_number": None,
            "bab_number": None,
            "ayat_number": None,
        }

        # Parse unit_id like "UU-2025-2/pasal-1" or "UU-2025-2/bab-1/pasal-5"
        parts = unit_id.split("/")
        path_parts = []

        for part in parts[1:]:  # Skip document part
            if part.startswith("bab-"):
                info["bab_number"] = part.replace("bab-", "")
                path_parts.append(f"Bab {info['bab_number']}")
            elif part.startswith("pasal-"):
                info["pasal_number"] = part.replace("pasal-", "")
                path_parts.append(f"Pasal {info['pasal_number']}")
            elif part.startswith("ayat-"):
                info["ayat_number"] = part.replace("ayat-", "")
                path_parts.append(f"Ayat {info['ayat_number']}")

        info["hierarchy_path"] = " / ".join(path_parts)
        return info


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
