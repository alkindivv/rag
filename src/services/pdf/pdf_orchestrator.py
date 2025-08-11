#!/usr/bin/env python3
"""
Simple PDF Orchestrator - Clean and Lean
Uses unified extractor with minimal complexity
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime
from dataclasses import dataclass, field

from .extractor import UnifiedPDFExtractor, ExtractionResult
from ...utils.citation import build_citation_string

logger = logging.getLogger(__name__)


@dataclass
class LegalNode:
    """Simple node for legal document structure."""
    type: str
    number: str
    title: str = ""
    content: str = ""
    children: List['LegalNode'] = field(default_factory=list)
    level: int = 0


# Indonesian legal hierarchy patterns - for tree structure
HIERARCHY_PATTERNS = {
    "buku": (r"^\s*BUKU\s+([IVXLC]+)", 1),
    "bab": (r"^\s*BAB\s+([IVX]+[A-Z]*)", 2),
    "bagian": (r"^\s*BAGIAN\s+(?:KE\s*)?(\w+)", 3),
    "paragraf": (r"^\s*PARAGRAF\s+(?:KE\s*)?(\w+)", 4),
    "pasal": (r"^\s*Pasal\s+(\d+[A-Z]*)", 5),
    "ayat": (r"^\s*\(\s*(\d+)\s*\)", 6),
    "huruf": (r"^\s*([a-z])\.\s*(.*)", 7),
    "angka": (r"^\s*(\d{1,2})\.\s*(.*)", 7.5),  # Flexible level for amendments and numbered items
}


class PDFOrchestrator:
    """Simple PDF orchestrator using unified extractor"""

    def __init__(self):
        """Initialize with unified extractor"""
        self.extractor = UnifiedPDFExtractor()
        # Pre-compile regex patterns for performance
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for name, (pattern, _) in HIERARCHY_PATTERNS.items()
        }
        logger.info("PDFOrchestrator initialized with unified extractor and tree structure support")

    def extract_text(self, file_path: str) -> ExtractionResult:
        """Extract text from PDF - simple delegation"""
        return self.extractor.extract_text(file_path)

    def process_document_complete(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete document processing with PDF extraction and text cleaning.
        """
        enhanced_data = document_data.copy()

        # Check if PDF exists
        pdf_path = document_data.get('pdf_path')
        if not pdf_path or not Path(pdf_path).exists():
            logger.warning(f"No PDF available for document: {document_data.get('title', 'Unknown')}")
            return enhanced_data

        try:
            # Extract text from PDF
            pdf_result = self.extract_text(pdf_path)

            if not pdf_result.success:
                logger.warning(f"PDF extraction failed for: {pdf_path}")
                return enhanced_data

            # Clean extracted text
            cleaned_text = self._clean_text(pdf_result.text)

            # Build document tree structure
            document_tree = self._build_document_tree(cleaned_text)

            # Convert tree to JSON structure matching specification
            doc_id = document_data.get('doc_id', 'document')
            doc_title = document_data.get('doc_title', document_data.get('title', 'Document'))
            enhanced_tree = self._serialize_tree(document_tree, doc_id, doc_title)

            # Augment cross references by resolving cited units' contents
            try:
                self._augment_cross_references(enhanced_tree)
            except Exception as e:
                logger.warning(f"Cross-reference augmentation failed: {e}")

            # Add PDF content, tree structure, and metadata
            enhanced_data['doc_content'] = cleaned_text
            enhanced_data['document_tree'] = enhanced_tree if enhanced_tree else None
            enhanced_data['doc_processing_status'] = 'pdf_processed'
            enhanced_data['pdf_extraction_metadata'] = {
                'method': pdf_result.method,
                'confidence': pdf_result.confidence,
                'processing_time': pdf_result.processing_time,
                'text_length': len(cleaned_text) if cleaned_text else 0,
                'original_text_length': len(pdf_result.text) if pdf_result.text else 0,
                'page_count': pdf_result.page_count,
                'tree_nodes_count': self._count_tree_nodes(document_tree) if document_tree else 0
            }

            logger.info(f"Successfully processed PDF: {pdf_path} ({pdf_result.method})")
            return enhanced_data

        except Exception as e:
            logger.error(f"PDF processing failed for {pdf_path}: {str(e)}")
            return enhanced_data

    def process_txt_content(self, document_data: Dict[str, Any], txt_content: str) -> Dict[str, Any]:
        """Process TXT content instead of PDF - for manual input"""
        enhanced_data = document_data.copy()

        try:
            # Clean TXT content (optional, tergantung kualitas TXT)
            cleaned_text = self._clean_text(txt_content)

            # Build document tree dari TXT
            document_tree = self._build_document_tree(cleaned_text)

            # Convert tree
            doc_id = document_data.get('doc_id', 'document')
            doc_title = document_data.get('doc_title', document_data.get('title', 'Document'))
            enhanced_tree = self._serialize_tree(document_tree, doc_id, doc_title)

            try:
                self._augment_cross_references(enhanced_tree)
            except Exception as e:
                logger.warning(f"Cross-reference augmentation failed: {e}")

            # Update data dengan TXT content & tree
            enhanced_data['doc_content'] = cleaned_text
            enhanced_data['document_tree'] = enhanced_tree if enhanced_tree else None
            enhanced_data['pdf_extraction_metadata'] = {
                'method': 'manual_txt_input',
                'confidence': 100.0,  # Manual input = perfect confidence
                'processing_time': 0.0,
                'text_length': len(cleaned_text) if cleaned_text else 0,
                'original_text_length': len(txt_content) if txt_content else 0,
                'page_count': 0,  # TXT doesn't have pages
                'tree_nodes_count': self._count_tree_nodes(document_tree) if document_tree else 0
            }
            enhanced_data['doc_processing_status'] = 'txt_processed'
            enhanced_data['last_updated'] = datetime.now().isoformat()

            logger.info(f"Successfully processed TXT content: {document_data.get('title', 'Unknown')}")
            return enhanced_data

        except Exception as e:
            logger.error(f"TXT content processing failed: {str(e)}")
            return enhanced_data

    def process_existing_pdfs(self, pdf_directory: str) -> List[Dict[str, Any]]:
        """
        Process existing PDF files in directory.
        """
        pdf_dir = Path(pdf_directory)
        documents = []

        if not pdf_dir.exists():
            logger.error(f"PDF directory does not exist: {pdf_dir}")
            return documents

        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")

        for i, pdf_path in enumerate(pdf_files, 1):
            try:
                logger.info(f"Processing PDF {i}/{len(pdf_files)}: {pdf_path.name}")

                # Create minimal document data
                document_data = {
                    'title': pdf_path.stem.replace('_', ' ').title(),
                    'form': 'Unknown',
                    'number': 'Unknown',
                    'year': 2024,
                    'pdf_path': str(pdf_path)
                }

                # Process the document
                processed_doc = self.process_document_complete(document_data)

                if processed_doc.get('doc_content'):
                    documents.append(processed_doc)
                    logger.info(f"Successfully processed: {pdf_path.name}")
                else:
                    logger.warning(f"No content extracted from: {pdf_path.name}")

            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {str(e)}")

        logger.info(f"Processed {len(documents)} PDFs successfully")
        return documents

    def _clean_text(self, text: str) -> str:
        """Clean extracted text for legal documents"""
        if not text:
            return ""

        try:
            # Import text cleaner if available
            from src.utils.text_cleaner import TextCleaner
            text_cleaner = TextCleaner()
            return text_cleaner.clean_legal_document_comprehensive(text)
        except ImportError:
            logger.warning("TextCleaner not available, using basic cleaning")
            # return self._basic_clean(text)

    # def _basic_clean(self, text: str) -> str:
    #     """Basic text cleaning fallback"""
    #     import re

    #     # Remove excessive whitespace
    #     text = re.sub(r'\s+', ' ', text)

    #     # Remove page breaks and form feeds
    #     text = re.sub(r'[\f\r]+', '\n', text)

    #     # Clean up line breaks
    #     text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

    #     return text.strip()

    def _build_document_tree(self, text: str) -> LegalNode:
        """Build structured document tree from legal text."""
        root = LegalNode(type="document", number="root", title="Document Root")

        if not text or not text.strip():
            return root

        lines = text.splitlines()
        current_stack = [root]  # Stack to track current hierarchy
        content_buffer = []
        in_amendment_section = False  # Track if we're in amendment/ketentuan section

        for line in lines:
            line = line.strip()
            if not line or len(line) < 2:
                continue

            # Skip common noise patterns
            if (line.startswith("REPUBLIK INDONESIA") or
                line in ["DENGAN RAHMAT TUHAN YANG MAHA ESA"] or
                re.match(r"^\s*Page\s+\d+\s*$", line) or
                len(line) <= 3):
                continue

            # Detect amendment/ketentuan sections
            if (re.search(r'disisipkan|diubah|dicabut|ditambahkan|ketentuan', line, re.IGNORECASE) or
                re.search(r'\d+\.\s+(di\s+antara|diantara|setelah|sebelum)', line, re.IGNORECASE)):
                in_amendment_section = True

            # Check for structural elements
            matched_node = self._match_hierarchy_pattern(line, in_amendment_section)
            if matched_node:
                # Flush any accumulated content to current node
                self._flush_content_buffer(content_buffer, current_stack[-1])
                content_buffer = []

                # Adaptive level adjustment for angka in amendments
                if matched_node.type == "angka" and in_amendment_section:
                    # Check if we're currently under a pasal (should be nested)
                    if (len(current_stack) > 1 and
                        current_stack[-1].type == "pasal"):
                        # This angka should be nested under the pasal
                        matched_node.level = 7.5  # Standard angka level
                    else:
                        # This is a standalone amendment provision
                        matched_node.level = 4  # Between paragraf and pasal

                # Reset amendment section flag for major structural elements
                if matched_node.type in ["buku", "bab", "bagian"]:
                    in_amendment_section = False

                # Standard hierarchy management
                while (len(current_stack) > 1 and
                       current_stack[-1].level >= matched_node.level):
                    current_stack.pop()

                # Add new node as child of current stack top
                current_stack[-1].children.append(matched_node)
                current_stack.append(matched_node)
            else:
                # Accumulate content
                content_buffer.append(line)

        # Flush remaining content
        self._flush_content_buffer(content_buffer, current_stack[-1])

        # Build full content for pasal nodes after tree is complete
        self._build_full_content_for_pasal(root)

        return root

    def _match_hierarchy_pattern(self, line: str, in_amendment_section: bool = False) -> LegalNode:
        """Match line against hierarchy patterns with amendment context awareness."""
        # Try patterns in hierarchical order
        pattern_order = ["buku", "bab", "bagian", "paragraf", "pasal", "ayat", "angka", "huruf"]

        for element_type in pattern_order:
            if element_type not in self._compiled_patterns:
                continue

            pattern = self._compiled_patterns[element_type]
            match = pattern.match(line)
            if match:
                level = HIERARCHY_PATTERNS[element_type][1]
                number = match.group(1).strip()

                # Special handling for angka - check if it's an amendment provision
                if element_type == "angka" and in_amendment_section:
                    # Check if this looks like a high-level amendment provision
                    content = match.group(2).strip() if len(match.groups()) > 1 else ""
                    if (re.search(r'disisipkan|diubah|dicabut|ditambahkan|ketentuan.*diatur', content, re.IGNORECASE) or
                        re.search(r'(di\s+antara|diantara|setelah|sebelum)\s+pasal', content, re.IGNORECASE) or
                        re.search(r'peraturan.*berlaku|tetap\s+berlaku', content, re.IGNORECASE)):
                        # This is a high-level amendment provision
                        level = 4  # Between paragraf and pasal level
                    # Otherwise keep default level for nested items under pasal

                # Handle list items with inline content
                if element_type in ["huruf", "angka"] and len(match.groups()) > 1:
                    content = match.group(2).strip() if match.group(2) else ""

                    # Create appropriate title format
                    if element_type == "huruf":
                        title = f"{number}. {content}" if content else f"{number}."
                    elif element_type == "angka":
                        title = f"{number}. {content}" if content else f"{number}."
                    else:
                        title = line.strip()

                    return LegalNode(
                        type=element_type,
                        number=number,
                        title=title,
                        content=content if content and len(content) > 5 else "",
                        level=level
                    )
                else:
                    # For structural elements without inline content
                    return LegalNode(
                        type=element_type,
                        number=number.upper() if element_type in ["bab", "buku"] else number,
                        title=line.strip(),
                        level=level
                    )

        return None

    def _flush_content_buffer(self, buffer: List[str], target_node: LegalNode):
        """Add buffered content to target node."""
        if not buffer:
            return

        content = " ".join(buffer).strip()
        if not content or len(content) < 5:
            return

        # Clean up common artifacts
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'^\s*[,;.]\s*', '', content)

        # For list items, avoid duplication
        if target_node.type in ["huruf", "angka"]:
            title_content = ""
            if ". " in target_node.title:
                title_content = target_node.title.split(". ", 1)[1]

            if title_content and content.lower().startswith(title_content.lower()[:30]):
                return

            if target_node.content:
                if content not in target_node.content and len(content) > 10:
                    target_node.content += " " + content
            else:
                target_node.content = content if content != title_content else ""
        else:
            # For structural elements, append content normally
            if target_node.content:
                target_node.content += "\n\n" + content
            else:
                target_node.content = content

    def _serialize_tree(
        self,
        node: LegalNode,
        doc_id: str,
        doc_title: str,
        parent_unit_id: Optional[str] = None,
        path: Optional[List[Dict[str, str]]] = None,
        pasal_id: Optional[str] = None,
        used_ids: Optional[set[str]] = None,
        current_ayat_id: Optional[str] = None,
        current_huruf_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Serialize LegalNode tree to specification-compliant structure."""

        if used_ids is None:
            used_ids = set()

        if path is None:
            path = [{"type": "dokumen", "label": doc_title, "unit_id": doc_id}]

        # Root document node -> emit TreeNode-compliant root
        if node.type == "document":
            # Ensure the root doc_id is reserved to avoid collisions
            used_ids.add(doc_id)

            root_node = {
                "type": "dokumen",
                "unit_id": doc_id,
                "number_label": None,
                "ordinal_int": None,
                "ordinal_suffix": "",
                "label_display": doc_title,
                "seq_sort_key": None,
                "citation_string": doc_title,
                "path": path,
                "title": doc_title,
                "content": None,
                "children": [
                    self._serialize_tree(child, doc_id, doc_title, doc_id, path, None, used_ids)
                    for child in node.children
                ],
            }

            return root_node

        parent_unit_id = parent_unit_id or doc_id
        base_unit_id = f"{parent_unit_id}/{node.type}-{node.number}"
        unit_id = base_unit_id
        counter = 2
        while unit_id in used_ids:
            unit_id = f"{base_unit_id}-{counter}"
            counter += 1
        used_ids.add(unit_id)
        number_label = node.number
        ordinal_int, ordinal_suffix = self._split_number_label(number_label)
        label_display = self._build_label_display(node.type, number_label)
        seq_sort_key = f"{ordinal_int:04d}|{ordinal_suffix}"
        current_entry = {"type": node.type, "label": label_display, "unit_id": unit_id}
        current_path = path + [current_entry]
        citation = build_citation_string(current_path, doc_title)

        data = {
            "type": node.type,
            "unit_id": unit_id,
            "number_label": number_label,
            "ordinal_int": ordinal_int,
            "ordinal_suffix": ordinal_suffix,
            "label_display": label_display,
            "seq_sort_key": seq_sort_key,
            "citation_string": citation,
            "path": current_path,
        }

        if node.type in ["bab", "bagian", "paragraf", "pasal"]:
            data["title"] = node.title

        if node.type == "pasal":
            data["content"] = node.content
            data["tags_semantik"] = []
            data["entities"] = []

        if node.type in ["ayat", "huruf", "angka"]:
            local_content = self._extract_inline_content(node)
            # Base linkage to pasal
            data.update(
                {
                    "parent_pasal_id": pasal_id,
                    "local_content": local_content,
                    "display_text": f"{label_display} {local_content}".strip(),
                    "bm25_body": local_content,
                    "span": None,
                }
            )
            # Additional strict parent links per level
            if node.type == "huruf":
                data["parent_ayat_id"] = current_ayat_id
            if node.type == "angka":
                data["parent_ayat_id"] = current_ayat_id
                data["parent_huruf_id"] = current_huruf_id

        child_pasal_id = pasal_id
        child_ayat_id = current_ayat_id
        child_huruf_id = current_huruf_id
        if node.type == "pasal":
            child_pasal_id = unit_id
            # reset lower anchors when entering a new pasal
            child_ayat_id = None
            child_huruf_id = None
        elif node.type == "ayat":
            child_ayat_id = unit_id
            child_huruf_id = None
        elif node.type == "huruf":
            child_huruf_id = unit_id

        data["children"] = [
            self._serialize_tree(
                child,
                doc_id,
                doc_title,
                unit_id,
                current_path,
                child_pasal_id,
                used_ids,
                child_ayat_id,
                child_huruf_id,
            )
            for child in node.children
        ]

        return data

    def _split_number_label(self, label: str) -> Tuple[int, str]:
        """Split number label into integer and suffix."""
        if not label:
            return 0, ""
        if re.fullmatch(r"[IVXLCDM]+", label):
            return self._roman_to_int(label), ""
        match = re.match(r"(\d+)([A-Za-z]*)", label)
        if match:
            return int(match.group(1)), match.group(2)
        if len(label) == 1 and label.isalpha():
            return ord(label.lower()) - 96, ""
        return 0, ""

    def _roman_to_int(self, roman: str) -> int:
        """Convert Roman numeral to integer."""
        roman_vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        result = 0
        prev = 0
        for ch in reversed(roman.upper()):
            val = roman_vals.get(ch, 0)
            if val < prev:
                result -= val
            else:
                result += val
                prev = val
        return result

    def _build_label_display(self, node_type: str, number_label: str) -> str:
        """Create display label for a node."""
        if node_type == "pasal":
            return f"Pasal {number_label}"
        if node_type == "bab":
            return f"BAB {number_label}"
        if node_type == "bagian":
            return f"BAGIAN {number_label}"
        if node_type == "paragraf":
            return f"PARAGRAF {number_label}"
        if node_type == "ayat":
            return f"({number_label})"
        if node_type in ["huruf", "angka"]:
            return f"{number_label}."
        return number_label


    def _extract_inline_content(self, node: LegalNode) -> str:
        """Extract inline content from node title if content missing."""
        if node.content:
            return node.content
        label = self._build_label_display(node.type, node.number)
        if node.title.startswith(label):
            return node.title[len(label):].strip()
        return node.title

    def _count_tree_nodes(self, node: LegalNode) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in node.children:
            count += self._count_tree_nodes(child)
        return count

    def _build_full_content_for_pasal(self, node: LegalNode):
        """Build full content for pasal nodes including all children content."""
        if node.type == "pasal":
            # Build full content for this pasal
            full_content_parts = []

            # Add pasal title and existing content
            if node.title and node.title != f"Pasal {node.number}":
                pasal_content = node.title.replace(f"Pasal {node.number}", "").strip()
                if pasal_content:
                    full_content_parts.append(pasal_content)

            if node.content:
                full_content_parts.append(node.content)

            # Add all children content recursively
            def collect_child_content(child_node, indent=""):
                child_parts = []

                if child_node.type == "ayat":
                    # Extract ayat content from title
                    ayat_text = child_node.title
                    if ayat_text.startswith(f"({child_node.number})"):
                        ayat_content = ayat_text[len(f"({child_node.number})"):].strip()
                        if ayat_content:
                            child_parts.append(f"{indent}({child_node.number}) {ayat_content}")

                    if child_node.content:
                        child_parts.append(f"{indent}{child_node.content}")

                elif child_node.type in ["huruf", "angka"]:
                    # Extract content from title
                    if child_node.type == "huruf":
                        prefix = f"{child_node.number}."
                    else:  # angka
                        prefix = f"{child_node.number}."

                    if child_node.title.startswith(prefix):
                        item_content = child_node.title[len(prefix):].strip()
                        if item_content:
                            child_parts.append(f"{indent}{prefix} {item_content}")

                    if child_node.content and child_node.content != item_content:
                        child_parts.append(f"{indent}{child_node.content}")

                else:
                    # Other node types
                    if child_node.title:
                        child_parts.append(f"{indent}{child_node.title}")
                    if child_node.content:
                        child_parts.append(f"{indent}{child_node.content}")

                # Process grandchildren with increased indent
                for grandchild in child_node.children:
                    child_parts.extend(collect_child_content(grandchild, indent + "  "))

                return child_parts

            # Collect content from all children
            for child in node.children:
                full_content_parts.extend(collect_child_content(child))

            # Update the pasal's content with full content
            node.content = "\n".join(full_content_parts)

        # Recursively process all children
        for child in node.children:
            self._build_full_content_for_pasal(child)

    # ------------------------
    # Cross-reference resolver
    # ------------------------
    def _augment_cross_references(self, root: Dict[str, Any]) -> None:
        """
        Traverse the serialized tree, detect references like
        'Pasal 6 ayat (1) huruf b' or 'Pasal 6 ayat (1)' or 'Pasal 6',
        and attach the referenced units' contents to the node as:
          - reference_contents: [ { 'ref': str, 'unit_id': str, 'content': str } ]
          - reference_content: str (if exactly one reference found)
        """
        if not root:
            return

        index = self._build_unit_index(root)

        # Regex captures: pasal, optional ayat, optional huruf
        ref_pattern = re.compile(
            r"Pasal\s+(?P<pasal>\d+[A-Za-z]*)"  # Pasal number with optional suffix
            r"(?:\s+ayat\s*\(\s*(?P<ayat>\d+)\s*\))?"  # optional ayat
            r"(?:\s+huruf\s+(?P<huruf>[a-z]))?",
            flags=re.IGNORECASE,
        )

        # Relative references (within current context):
        #   - ayat (N)  [optionally preceded by legal phrasing]
        rel_ayat_pattern = re.compile(
            r"(?:sebagaimana\s+dimaksud(?:\s+dalam|\s+pada)?\s*)?ayat\s*\(\s*(?P<ayat_only>\d+)\s*\)",
            flags=re.IGNORECASE,
        )
        #   - huruf x  [optionally preceded by legal phrasing]
        rel_huruf_pattern = re.compile(
            r"(?:sebagaimana\s+dimaksud(?:\s+dalam|\s+pada)?\s*)?huruf\s+(?P<huruf_only>[a-z])",
            flags=re.IGNORECASE,
        )

        def extract_text_fields(node_dict: Dict[str, Any]) -> List[Tuple[str, str]]:
            """Return list of (field_name, text) to scan for references."""
            fields = []
            t = node_dict.get("type")
            if t == "pasal":
                if node_dict.get("content"):
                    fields.append(("content", node_dict["content"]))
            elif t in ("ayat", "huruf", "angka"):
                # Prefer local_content; fall back to display_text
                if node_dict.get("local_content"):
                    fields.append(("local_content", node_dict["local_content"]))
                elif node_dict.get("display_text"):
                    fields.append(("display_text", node_dict["display_text"]))
            else:
                # For structural nodes, scan title if present
                if node_dict.get("title"):
                    fields.append(("title", node_dict["title"]))
            return fields

        def resolve_one(match: re.Match) -> Optional[Dict[str, str]]:
            pasal = match.group("pasal")
            ayat = match.group("ayat")
            huruf = match.group("huruf")
            target = self._find_unit_by_ref(index, pasal, ayat, huruf)
            if not target:
                return None
            content = self._get_unit_content(target)
            if not content:
                return None
            ref_str = match.group(0)
            return {"ref": ref_str, "unit_id": target.get("unit_id"), "content": content}

        def walk(node_dict: Dict[str, Any], current_pasal: Optional[str] = None, current_ayat: Optional[str] = None):
            # Maintain traversal context based on node type
            t = node_dict.get("type")
            if t == "pasal" and node_dict.get("number_label"):
                current_pasal = str(node_dict.get("number_label"))
                current_ayat = None
            elif t == "ayat" and node_dict.get("number_label") and current_pasal:
                current_ayat = str(node_dict.get("number_label"))

            # Find references in text fields
            refs: List[Dict[str, str]] = []
            seen: set[tuple] = set()
            for _, text in extract_text_fields(node_dict):
                txt = text or ""
                # 1) Absolute references: Pasal N [ayat (M) [huruf x]]
                for m in ref_pattern.finditer(txt):
                    resolved = resolve_one(m)
                    if resolved:
                        key = (resolved["unit_id"], resolved["ref"])
                        if key not in seen:
                            seen.add(key)
                            refs.append(resolved)
                # 2) Relative ayat: ayat (M) -> (current_pasal, M)
                for m in rel_ayat_pattern.finditer(txt):
                    if current_pasal:
                        ay = m.group("ayat_only")
                        target = index["ayat"].get((str(current_pasal), str(ay)))
                        if target:
                            content = self._get_unit_content(target)
                            if content:
                                ref_str = m.group(0)
                                resolved = {"ref": ref_str, "unit_id": target.get("unit_id"), "content": content}
                                key = (resolved["unit_id"], resolved["ref"])
                                if key not in seen:
                                    seen.add(key)
                                    refs.append(resolved)
                # 3) Relative huruf: huruf x -> (current_pasal, current_ayat, x)
                for m in rel_huruf_pattern.finditer(txt):
                    if current_pasal and current_ayat:
                        hf = m.group("huruf_only").lower()
                        target = index["huruf"].get((str(current_pasal), str(current_ayat), hf))
                        if target:
                            content = self._get_unit_content(target)
                            if content:
                                ref_str = m.group(0)
                                resolved = {"ref": ref_str, "unit_id": target.get("unit_id"), "content": content}
                                key = (resolved["unit_id"], resolved["ref"])
                                if key not in seen:
                                    seen.add(key)
                                    refs.append(resolved)

            if refs:
                node_dict["reference_contents"] = refs
                if len(refs) == 1:
                    node_dict["reference_content"] = refs[0]["content"]
            # Recurse
            for child in node_dict.get("children", []) or []:
                walk(child, current_pasal, current_ayat)

        walk(root)

    def _build_unit_index(self, root: Dict[str, Any]) -> Dict[str, Any]:
        """Build lookup indices for pasal/ayat/huruf by their numbers."""
        index = {
            "pasal": {},  # pasal_num -> node
            "ayat": {},   # (pasal_num, ayat_num) -> node
            "huruf": {},  # (pasal_num, ayat_num, huruf) -> node
        }

        def walk(node_dict: Dict[str, Any], current_pasal: Optional[str] = None, current_ayat: Optional[str] = None):
            t = node_dict.get("type")
            label = node_dict.get("label_display") or ""
            # normalize extract number token
            num = node_dict.get("number_label")
            if t == "pasal" and num:
                current_pasal = str(num)
                index["pasal"][current_pasal] = node_dict
                current_ayat = None
            elif t == "ayat" and num and current_pasal:
                current_ayat = str(num)
                index["ayat"][ (current_pasal, current_ayat) ] = node_dict
            elif t == "huruf" and num and current_pasal and current_ayat:
                index["huruf"][ (current_pasal, str(current_ayat), str(num).lower()) ] = node_dict

            for child in node_dict.get("children", []) or []:
                walk(child, current_pasal, current_ayat)

        walk(root)
        return index

    def _find_unit_by_ref(self, index: Dict[str, Any], pasal: str, ayat: Optional[str], huruf: Optional[str]) -> Optional[Dict[str, Any]]:
        pasal_key = str(pasal)
        target = index["pasal"].get(pasal_key)
        if not target:
            return None
        if ayat:
            target = index["ayat"].get((pasal_key, str(ayat)))
            if not target:
                return None
            if huruf:
                target = index["huruf"].get((pasal_key, str(ayat), str(huruf).lower()))
                if not target:
                    return None
        return target

    def _get_unit_content(self, node_dict: Dict[str, Any]) -> str:
        """Extract the most relevant text for a unit node."""
        t = node_dict.get("type")
        if t == "pasal":
            return node_dict.get("content") or ""
        if t in ("ayat", "huruf", "angka"):
            return node_dict.get("local_content") or node_dict.get("display_text") or ""
        # structural: return title
        return node_dict.get("title") or ""

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            'extractor_methods': len(self.extractor.methods),
            'available_methods': [method[0] for method in self.extractor.methods],
            'tree_patterns': len(self._compiled_patterns),
            'hierarchy_levels': list(HIERARCHY_PATTERNS.keys()),
            'status': 'ready'
        }
