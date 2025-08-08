"""
Pattern Manager for Legal Document Processing
Centralized regex patterns for Indonesian legal documents

This module provides a unified interface for all regex patterns used in
legal document processing, eliminating duplication and ensuring consistency.

Author: Refactored Architecture
Purpose: Centralized pattern management following DRY principles
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    """Types of legal document patterns."""
    CHAPTER = "chapter"
    ARTICLE = "article"
    VERSE = "verse"
    DEFINITION = "definition"
    HIERARCHICAL = "hierarchical"
    NUMBERED = "numbered"
    LEGAL_REFERENCE = "legal_reference"
    DOCUMENT_TYPE = "document_type"


@dataclass
class MatchResult:
    """Result of pattern matching."""
    text: str
    start: int
    end: int
    groups: Dict[str, str]
    confidence: float = 1.0


@dataclass
class ChapterMatch:
    """Matched chapter information."""
    number: str
    title: str
    roman_numeral: bool = True
    confidence: float = 1.0


@dataclass
class ArticleMatch:
    """Matched article information."""
    number: str
    title: str = ""
    content: str = ""
    confidence: float = 1.0


@dataclass
class VerseMatch:
    """Matched verse information."""
    number: str
    content: str
    parent_article: str = ""
    confidence: float = 1.0


@dataclass
class HierarchicalMatch:
    """Matched hierarchical item (a, b, c)."""
    letter: str
    content: str
    level: int = 1
    confidence: float = 1.0


@dataclass
class NumberedMatch:
    """Matched numbered item (1, 2, 3)."""
    number: str
    content: str
    format_type: str = "parenthesis"  # parenthesis, period, bracket
    confidence: float = 1.0


@dataclass
class LegalReferenceMatch:
    """Matched legal document reference."""
    type: str  # UU, PP, Perpres, etc.
    number: str
    year: str
    full_text: str
    confidence: float = 1.0


class PatternManager:
    """
    Centralized pattern manager for Indonesian legal documents.

    Provides consistent regex patterns and matching functions for:
    - BAB (Chapters)
    - Pasal (Articles)
    - Ayat (Verses)
    - Hierarchical lists (a, b, c)
    - Legal document references
    - Document type identification
    """

    def __init__(self):
        """Initialize pattern manager with compiled patterns."""
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile all regex patterns for performance."""

        # Chapter patterns (BAB)
        self.chapter_patterns = {
            'roman': re.compile(
                r'\bBAB\s+([IVX]+)\s*[:\-]?\s*([A-Z\s]+?)(?=\n|$|BAB|\bPasal)',
                re.IGNORECASE | re.MULTILINE
            ),
            'numeric': re.compile(
                r'\bBAB\s+(\d+)\s*[:\-]?\s*([A-Z\s]+?)(?=\n|$|BAB|\bPasal)',
                re.IGNORECASE | re.MULTILINE
            )
        }

        # Article patterns (Pasal)
        self.article_patterns = {
            'simple': re.compile(
                r'\bPasal\s+(\d+(?:[A-Z])?)\s*([^\n]*)',
                re.IGNORECASE
            ),
            'with_title': re.compile(
                r'\bPasal\s+(\d+(?:[A-Z])?)\s*[:\-]?\s*([^\n]+?)(?=\n|$|\(1\)|BAB|\bPasal)',
                re.IGNORECASE | re.MULTILINE
            ),
            'full_hierarchy': re.compile(
                r'\bPasal\s+(\d+(?:[A-Z])?)\s*(?:ayat\s*\(?\s*(\d+)\s*\)?)?\s*(?:huruf\s*([a-z]+))?\s*(?:angka\s*(\d+)\)?)?\s*(?:alinea\s*([a-z]+)\)?)?',
                re.IGNORECASE
            )
        }

        # Verse patterns (Ayat)
        self.verse_patterns = {
            'parenthesis': re.compile(
                r'\((\d+)\)\s*([^\n(]+?)(?=\n\(\d+\)|\n[A-Z]|\n$|$)',
                re.MULTILINE | re.DOTALL
            ),
            'bracket': re.compile(
                r'\[(\d+)\]\s*([^\n\[]+?)(?=\n\[\d+\]|\n[A-Z]|\n$|$)',
                re.MULTILINE | re.DOTALL
            )
        }

        # Hierarchical patterns (a, b, c)
        self.hierarchical_patterns = {
            'letter_period': re.compile(
                r'^([a-z])\.\s*([^\n;]+?)(?=\s*[a-z]\.|$|;)',
                re.MULTILINE | re.IGNORECASE
            ),
            'letter_parenthesis': re.compile(
                r'([a-z])\)\s*([^\n)]+?)(?=\s*[a-z]\)|$)',
                re.MULTILINE | re.IGNORECASE
            )
        }

        # Numbered list patterns
        self.numbered_patterns = {
            'parenthesis': re.compile(
                r'(\d+)\)\s*([^\n)]+?)(?=\s*\d+\)|$)',
                re.MULTILINE
            ),
            'period': re.compile(
                r'^(\d+)\.\s*([^\n.]+?)(?=\s*\d+\.|$)',
                re.MULTILINE
            )
        }

        # Legal reference patterns
        self.legal_reference_patterns = {
            'standard': re.compile(
                r'\b(UU|PP|Perpres|Peraturan\s+Presiden|Peraturan\s+Pemerintah|Undang-Undang)\s*(?:No\.?|Nomor)?\s*(\d+(?:[A-Z])?)\s*(?:Tahun|th\.?)?\s*(\d{4})',
                re.IGNORECASE
            ),
            'ministerial': re.compile(
                r'\b(Permen|Peraturan\s+Menteri)\s+([A-Za-z\s]+)\s*(?:No\.?|Nomor)?\s*(\d+(?:[A-Z])?)\s*(?:Tahun|th\.?)?\s*(\d{4})',
                re.IGNORECASE
            ),
            'regional': re.compile(
                r'\b(Perda|Peraturan\s+Daerah)\s*(?:Provinsi|Kabupaten|Kota)?\s*([A-Za-z\s]*)\s*(?:No\.?|Nomor)?\s*(\d+(?:[A-Z])?)\s*(?:Tahun|th\.?)?\s*(\d{4})',
                re.IGNORECASE
            )
        }

        # Document type patterns
        self.document_type_patterns = {
            'full_title': re.compile(
                r'(UNDANG-UNDANG|PERATURAN\s+PEMERINTAH|PERATURAN\s+PRESIDEN|PERATURAN\s+MENTERI)\s+(?:REPUBLIK\s+INDONESIA\s+)?(?:NOMOR|NO\.?)\s*(\d+(?:[A-Z])?)\s*TAHUN\s*(\d{4})',
                re.IGNORECASE | re.MULTILINE
            ),
            'abbreviated': re.compile(
                r'\b(UU|PP|Perpres|Permen|Perda)\s*(?:No\.?|Nomor)?\s*(\d+(?:[A-Z])?)\s*(?:Tahun|th\.?)?\s*(\d{4})',
                re.IGNORECASE
            )
        }

        # Definition section patterns
        self.definition_patterns = {
            'standard': re.compile(
                r'Dalam\s+(?:Undang-Undang|Peraturan|undang-undang|peraturan)\s+ini\s+yang\s+dimaksud\s+dengan',
                re.IGNORECASE
            ),
            'section_header': re.compile(
                r'\b(?:KETENTUAN\s+UMUM|DEFINISI|PENGERTIAN)\b',
                re.IGNORECASE
            )
        }

        # Cleaning patterns for text processing
        self.noise_patterns = {
            # Ultra-comprehensive SK No patterns for all OCR variations
            'sk_basic': re.compile(
                r'SK\s+No\.?\s*\d+[A-Z]*(?:\s*\([^)]*\))?\s*',
                re.IGNORECASE
            ),
            'sk_ocr_corrupted': re.compile(
                r'SK\s*No\.?\s*[Il1lO0]+[Il1lO0A-Z\d]*[A-Z]*(?:\s*[DSAZ])*\s*',
                re.IGNORECASE
            ),
            'sk_spaced_dotted': re.compile(
                r'SK[\s\.]*No[\s\.]*[Il1lO0\d]+[Il1lO0A-Z\d]*[A-Z]*\s*',
                re.IGNORECASE
            ),
            'sk_with_parentheses': re.compile(
                r'SK[\s\.]*No[\s\.]*[Il1lO0\d]+[Il1lO0A-Z\d]*[A-Z]*\.?\s*\([^)]*\)\s*',
                re.IGNORECASE
            ),
            'sk_line_artifacts': re.compile(
                r'^\s*SK[\s\.]*No[\s\.]*[Il1lO0\d]+[Il1lO0A-Z\d]*[A-Z]*.*$',
                re.IGNORECASE | re.MULTILINE
            ),
            'sk_garbled_inline': re.compile(
                r'SK[\s\.]*No[\s\.]*[Il1lO0\d]+[Il1lO0A-Z\d]*[A-Z]*(?:\s*[DSAZ])*(?:\s*\([^)]*\))?\s*',
                re.IGNORECASE
            ),
            'page_artifacts': re.compile(
                r'-\s*\d+\s*-|^-\d+-$|^-\d+[a-z]*-$|\[?(?:Page|Hal(?:aman)?)\s*\d+(?:\s*of\s*\d+)?\]?',
                re.IGNORECASE | re.MULTILINE
            ),
            'garbled_text': re.compile(
                r'(?:T\{!TTilTTTIT\'T\'\]ITSrJ|rrflrFlflxllf\.r\]Ilsnl|EfltrtrLlf,EEtrtrEln|E\.-l-\+\{t\]\{Il|EETITIilIilTI|REPTIELIK|FRESIDEN|REFUEUK)',
                re.IGNORECASE
            ),
            'ocr_noise': re.compile(
                r'(?:EETITIilIilTI|TIr:I\+Tf\.I\{Il|llrr-\{rT;trIllilTlTatrtltrtrIrtr|aftfd\'TFITli|nrrFF\[iriN\]|[A-Z]{1,3}\s+lfr\'o)',
                re.IGNORECASE
            ),
            'republic_variations': re.compile(
                r'(?:REPTIBLIK|REPUEUK|REPUEIJK|R\]EPUELIK)\s*(?:TNDONESIA|INOONESIA|INDOXESIA|INDOHESIA|IITIDONESIA|INDONESIA)',
                re.IGNORECASE
            ),
            'president_variations': re.compile(
                r'(?:FRESIDEN|REX\'UEUK|FN\.\s*ESIDEN)\s*(?:REPUBLIK|REPTIBLIK|REPUEUK|IITIDONESIA|TNDONESIA|INDOXESIA|INDOHESIA)?',
                re.IGNORECASE
            ),
            'page_numbers': re.compile(
                r'-[tl]\d+[a-z]*-|-L\d+[a-z]*-',
                re.IGNORECASE
            ),
            'ocr_symbols': re.compile(
                r'(?:rrfl\s*Tf:\s*[\+]?Jn!|irrIrLtrtITII\{\'\s*If\{jm|r[-!]\s*\?\s*r[\*]?TiT[;]?\s*ill|\[r[\$]ilrr[\$]5f\]t|Et--l[,]?\s*FTf\.\s*I\{Il|lrl-FtTiIiN|iEiT[;]?\s*FIEtrN|ITTITFTfI\'IITTI[-\{],?\s*TTT5N)',
                re.IGNORECASE
            ),
            'stubborn_ocr': re.compile(
                r'r[-!]\s*\?\s*r[\*]?TiT[;]?\s*ill',
                re.IGNORECASE
            ),
            'remaining_garbage': re.compile(
                r'(?:-to\d+-|r[-!]\s*\?\s*r[\*]?TiT[;]?\s*ill|ITTITFTfI\'IITTI[-\{],?\s*TTT5N|PRESIDEN\s+REPUBLIK\s+TNDONESIA)',
                re.IGNORECASE
            ),
            'page_artifacts_extended': re.compile(
                r'-[tloL]\d+[a-z]*-|\{\d+\s*[-L]\d+[-]?',
                re.IGNORECASE
            ),
            'ocr_patterns_complex': re.compile(
                r'(?:INEEtrtrEIn|EIIEEIf|INIIEtrtrEIn)',
                re.IGNORECASE
            ),
            'numeric_artifacts': re.compile(
                r'^\d+[A-Z]\s*(?:EIIEEIf|INEEtrtrEIn|FN\.\s*ESIDEN)?.*$',
                re.IGNORECASE | re.MULTILINE
            ),
            'mixed_garbage': re.compile(
                r'\{\d+\s+(?:INEEtrtrEIn|INIIEtrtrEIn)\s*[-L]\d+[-]?',
                re.IGNORECASE
            ),
            'additional_republic_variants': re.compile(
                r'(?:REPUBI\.\s*IK|REFI\.\s*IEU\(|REFI\.\s*IBUI\()\s*INDONESI?A?',
                re.IGNORECASE
            ),
            'complex_ocr_artifacts': re.compile(
                r'(?:Efltrtrf,\s*I\]IEENtrEIA|r[-!]\s*\?\s*r[\*]?TiT[;]?\s*ill)',
                re.IGNORECASE
            ),
            'parentheses_garbage': re.compile(
                r'\.\(\s*i[-]ili\s*rIrL\s*\'tr',
                re.IGNORECASE
            ),
            'document_headers': re.compile(
                r'(?:PRESIDEN\s+)?REPUBLIK\s+INDONESIA\s*',
                re.IGNORECASE
            ),
            'excessive_punctuation': re.compile(r'[,\s]{3,}|\.{4,}|\-{4,}'),
            'form_fields': re.compile(r'____+|\.\.\.\.+|\[\s*\]'),
            'single_letters': re.compile(r'^\s*[A-Z]\s*$', re.MULTILINE)
        }

        # Whitespace patterns for normalization
        self.whitespace_patterns = {
            'multiple_spaces': re.compile(r' {2,}'),
            'multiple_newlines': re.compile(r'\n{3,}'),
            'trailing_spaces': re.compile(r' +$', re.MULTILINE),
            'leading_spaces': re.compile(r'^ +', re.MULTILINE),
            'mixed_whitespace': re.compile(r'[ \t]+')
        }

        # OCR correction patterns
        self.ocr_correction_patterns = {
            'republic_variations': re.compile(r'REPUB?LI[CK]?\s*INDONESIA', re.IGNORECASE),
            'president_variations': re.compile(r'PRES[Il]DEN\s*REPUBLIK', re.IGNORECASE),
            'garbled_headers': re.compile(r'[A-Z]{3,}[Il1]{2,}[A-Z]{3,}', re.IGNORECASE)
        }

        # BAB fixing patterns
        self.bab_fix_patterns = {
            'no_space': re.compile(r'BAB([IVXLCDM]+)\s*\.{3,}'),
            'spaced_ellipsis': re.compile(r'BAB\s+([IVXLCDM]+)\s*\.\s*\.\s*\.'),
            'corrupted_bab': re.compile(r'BAB([IVXLCDM]+)(?=\s|$)')
        }

        # Pasal number fixing patterns
        self.pasal_fix_patterns = {
            'no_space_digits': re.compile(r'\bPasal(\d+[A-Za-z]?)\b'),
            'embedded_ocr_full': re.compile(r'\bPasal\s*([Il1lO0]*\d*[Il1lO0]*\d*[Il1lO0]*[A-Za-z]?)\b'),
            'pasa_typo': re.compile(r'\bPasa7(\d+)\b'),
            'space_in_number': re.compile(r'\bPasal\s+(\d+)\s+([Il1])\s*\b'),
            'ayat_ocr': re.compile(r'\bl(\d+)l\b'),
            'ayat_broken': re.compile(r'\((\d)[Il1]\b'),
            'newline_missing': re.compile(r'(Pasal\s+\d+[A-Za-z]?)\s+([A-Z][a-z]+)'),
            'complex_spacing': re.compile(r'\bPasal\s+(\d+)\s+(\d+)\s+([Il1])\b'),
            'ocr_in_middle': re.compile(r'\bPasal\s*([Il1])(\d+)([Il1])\b')
        }

    def find_chapters(self, text: str) -> List[ChapterMatch]:
        """
        Find all chapters (BAB) in the text.

        Args:
            text: Legal document text

        Returns:
            List of ChapterMatch objects
        """
        chapters = []

        # Try Roman numeral pattern first
        for match in self.chapter_patterns['roman'].finditer(text):
            chapters.append(ChapterMatch(
                number=match.group(1),
                title=match.group(2).strip(),
                roman_numeral=True,
                confidence=0.95
            ))

        # Try numeric pattern for mixed documents
        for match in self.chapter_patterns['numeric'].finditer(text):
            chapters.append(ChapterMatch(
                number=match.group(1),
                title=match.group(2).strip(),
                roman_numeral=False,
                confidence=0.90
            ))

        # Remove duplicates and sort by position
        unique_chapters = []
        seen_numbers = set()

        for chapter in sorted(chapters, key=lambda x: text.find(f"BAB {x.number}")):
            if chapter.number not in seen_numbers:
                unique_chapters.append(chapter)
                seen_numbers.add(chapter.number)

        return unique_chapters

    def find_articles(self, text: str) -> List[ArticleMatch]:
        """
        Find all articles (Pasal) in the text.

        Args:
            text: Legal document text

        Returns:
            List of ArticleMatch objects
        """
        articles = []

        # Use title-aware pattern first
        for match in self.article_patterns['with_title'].finditer(text):
            articles.append(ArticleMatch(
                number=match.group(1),
                title=match.group(2).strip() if match.group(2) else "",
                confidence=0.90
            ))

        # Fallback to simple pattern for articles without clear titles
        if not articles:
            for match in self.article_patterns['simple'].finditer(text):
                articles.append(ArticleMatch(
                    number=match.group(1),
                    title=match.group(2).strip() if match.group(2) else "",
                    confidence=0.80
                ))

        # Remove duplicates
        unique_articles = []
        seen_numbers = set()

        for article in articles:
            if article.number not in seen_numbers:
                unique_articles.append(article)
                seen_numbers.add(article.number)

        return unique_articles

    def find_verses(self, text: str) -> List[VerseMatch]:
        """
        Find all verses (ayat) in the text.

        Args:
            text: Legal document text

        Returns:
            List of VerseMatch objects
        """
        verses = []

        # Find parenthesis format: (1), (2), etc.
        for match in self.verse_patterns['parenthesis'].finditer(text):
            verses.append(VerseMatch(
                number=match.group(1),
                content=match.group(2).strip(),
                confidence=0.95
            ))

        # Find bracket format: [1], [2], etc.
        for match in self.verse_patterns['bracket'].finditer(text):
            verses.append(VerseMatch(
                number=match.group(1),
                content=match.group(2).strip(),
                confidence=0.85
            ))

        return verses

    def find_hierarchical_items(self, text: str) -> List[HierarchicalMatch]:
        """
        Find hierarchical list items (a, b, c format).

        Args:
            text: Legal document text

        Returns:
            List of HierarchicalMatch objects
        """
        items = []

        # Find letter with period: a., b., c.
        for match in self.hierarchical_patterns['letter_period'].finditer(text):
            items.append(HierarchicalMatch(
                letter=match.group(1).lower(),
                content=match.group(2).strip(),
                confidence=0.90
            ))

        # Find letter with parenthesis: a), b), c)
        for match in self.hierarchical_patterns['letter_parenthesis'].finditer(text):
            items.append(HierarchicalMatch(
                letter=match.group(1).lower(),
                content=match.group(2).strip(),
                confidence=0.85
            ))

        return items

    def find_numbered_items(self, text: str) -> List[NumberedMatch]:
        """
        Find numbered list items.

        Args:
            text: Legal document text

        Returns:
            List of NumberedMatch objects
        """
        items = []

        # Find numbered with parenthesis: 1), 2), 3)
        for match in self.numbered_patterns['parenthesis'].finditer(text):
            items.append(NumberedMatch(
                number=match.group(1),
                content=match.group(2).strip(),
                format_type="parenthesis",
                confidence=0.90
            ))

        # Find numbered with period: 1., 2., 3.
        for match in self.numbered_patterns['period'].finditer(text):
            items.append(NumberedMatch(
                number=match.group(1),
                content=match.group(2).strip(),
                format_type="period",
                confidence=0.85
            ))

        return items

    def find_legal_references(self, text: str) -> List[LegalReferenceMatch]:
        """
        Find legal document references in text.

        Args:
            text: Legal document text

        Returns:
            List of LegalReferenceMatch objects
        """
        references = []

        # Standard references (UU, PP, Perpres)
        for match in self.legal_reference_patterns['standard'].finditer(text):
            references.append(LegalReferenceMatch(
                type=self._normalize_document_type(match.group(1)),
                number=match.group(2),
                year=match.group(3),
                full_text=match.group(0),
                confidence=0.95
            ))

        # Ministerial regulations
        for match in self.legal_reference_patterns['ministerial'].finditer(text):
            references.append(LegalReferenceMatch(
                type="PERMEN",
                number=match.group(3),
                year=match.group(4),
                full_text=match.group(0),
                confidence=0.90
            ))

        # Regional regulations
        for match in self.legal_reference_patterns['regional'].finditer(text):
            references.append(LegalReferenceMatch(
                type="PERDA",
                number=match.group(3),
                year=match.group(4),
                full_text=match.group(0),
                confidence=0.85
            ))

        return references

    def is_definition_section(self, text: str) -> bool:
        """
        Check if text contains definition section indicators.

        Args:
            text: Text to check

        Returns:
            True if text appears to be a definition section
        """
        for pattern in self.definition_patterns.values():
            if pattern.search(text):
                return True
        return False

    def detect_document_type(self, text: str) -> Optional[str]:
        """
        Detect the type of legal document from text.

        Args:
            text: Document text

        Returns:
            Document type (UU, PP, etc.) or None if not detected
        """
        # Try full title pattern first
        match = self.document_type_patterns['full_title'].search(text)
        if match:
            return self._normalize_document_type(match.group(1))

        # Try abbreviated pattern
        match = self.document_type_patterns['abbreviated'].search(text)
        if match:
            return self._normalize_document_type(match.group(1))

        return None

    def _normalize_document_type(self, raw_type: str) -> str:
        """
        Normalize document type to standard abbreviation.

        Args:
            raw_type: Raw document type from text

        Returns:
            Normalized document type
        """
        type_mapping = {
            'undang-undang': 'UU',
            'undang undang': 'UU',
            'uu': 'UU',
            'peraturan pemerintah': 'PP',
            'pp': 'PP',
            'peraturan presiden': 'PERPRES',
            'perpres': 'PERPRES',
            'peraturan menteri': 'PERMEN',
            'permen': 'PERMEN',
            'peraturan daerah': 'PERDA',
            'perda': 'PERDA'
        }

        normalized = raw_type.lower().strip()
        return type_mapping.get(normalized, raw_type.upper())

    def extract_article_hierarchy(self, text: str, article_number: str) -> Dict[str, Any]:
        """
        Extract complete hierarchy for a specific article.

        Args:
            text: Legal document text
            article_number: Article number to extract

        Returns:
            Dictionary containing verses, hierarchical items, etc.
        """
        # Find article boundaries
        article_pattern = rf'\bPasal\s+{re.escape(article_number)}\b.*?(?=\bPasal\s+\d+|\bBAB\s+|$)'
        article_match = re.search(article_pattern, text, re.IGNORECASE | re.DOTALL)

        if not article_match:
            return {}

        article_text = article_match.group(0)

        return {
            'verses': self.find_verses(article_text),
            'hierarchical_items': self.find_hierarchical_items(article_text),
            'numbered_items': self.find_numbered_items(article_text),
            'legal_references': self.find_legal_references(article_text)
        }

    def validate_legal_structure(self, text: str) -> Dict[str, Any]:
        """
        Validate overall legal document structure.

        Args:
            text: Legal document text

        Returns:
            Validation results with confidence scores
        """
        chapters = self.find_chapters(text)
        articles = self.find_articles(text)
        verses = self.find_verses(text)

        # Calculate structure confidence
        has_chapters = len(chapters) > 0
        has_articles = len(articles) > 0
        has_verses = len(verses) > 0
        has_definition = self.is_definition_section(text)

        structure_score = sum([has_chapters, has_articles, has_verses, has_definition]) / 4

        return {
            'is_valid_legal_document': structure_score > 0.5,
            'confidence': structure_score,
            'chapters_found': len(chapters),
            'articles_found': len(articles),
            'verses_found': len(verses),
            'has_definitions': has_definition,
            'document_type': self.detect_document_type(text)
        }

    def clean_noise_artifacts(self, text: str) -> str:
        """
        Clean noise artifacts from text using compiled patterns.
        Surgically removes SK No artifacts while preserving legal content.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Process line by line for surgical SK No removal
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            original_line = line.strip()

            # Skip empty lines
            if not original_line:
                continue

            # Check if line is ONLY SK artifacts (no legal content)
            has_legal_content = any(legal_marker in line.upper() for legal_marker in [
                'PASAL', 'BAB', 'AYAT', 'KETENTUAN', 'SANKSI', 'UNDANG-UNDANG',
                'REPUBLIK', 'INDONESIA', 'TENTANG', 'NOMOR', 'TAHUN'
            ])

            # Check if line contains list markers (a., b., (1), etc.)
            has_list_markers = bool(re.search(r'[a-z]\.\s|\(\d+\)\s|\d+\.\s', line))

            # If line has legal content or list markers, preserve it but clean SK artifacts
            if has_legal_content or has_list_markers:
                # Remove SK artifacts surgically from the line
                for pattern_name, pattern in self.noise_patterns.items():
                    if 'sk_' in pattern_name:
                        line = pattern.sub('', line)

                # Apply other cleaning patterns
                for pattern_name, pattern in self.noise_patterns.items():
                    if 'sk_' not in pattern_name:
                        line = pattern.sub('', line)

                # Keep the line if it still has content after cleaning
                if line.strip():
                    cleaned_lines.append(line)
            else:
                # Check if line is purely SK artifacts
                is_sk_only = any(pattern.search(original_line) for pattern_name, pattern in self.noise_patterns.items() if 'sk_' in pattern_name)

                if not is_sk_only:
                    # Apply all cleaning patterns to non-SK lines
                    for pattern in self.noise_patterns.values():
                        line = pattern.sub('', line)

                    # Keep the line if it has content after cleaning
                    if line.strip():
                        cleaned_lines.append(line)

        # Rejoin and apply final inline SK cleanup
        text = '\n'.join(cleaned_lines)

        # Final pass: remove any remaining SK artifacts inline
        for pattern_name, pattern in self.noise_patterns.items():
            if 'sk_' in pattern_name:
                text = pattern.sub('', text)

        return text

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text using compiled patterns.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        for pattern_name, pattern in self.whitespace_patterns.items():
            if pattern_name == 'multiple_spaces':
                text = pattern.sub(' ', text)
            elif pattern_name == 'multiple_newlines':
                text = pattern.sub('\n\n', text)
            elif pattern_name in ['trailing_spaces', 'leading_spaces']:
                text = pattern.sub('', text)
            elif pattern_name == 'mixed_whitespace':
                text = pattern.sub(' ', text)
        return text

    def fix_bab_patterns(self, text: str) -> str:
        """
        Fix corrupted BAB patterns using compiled patterns.

        Args:
            text: Text with potentially corrupted BAB patterns

        Returns:
            Text with fixed BAB patterns
        """
        # Fix no space between BAB and number
        text = self.bab_fix_patterns['no_space'].sub(r'BAB \1', text)
        text = self.bab_fix_patterns['corrupted_bab'].sub(r'BAB \1', text)

        # Fix spaced BAB with ellipsis
        text = self.bab_fix_patterns['spaced_ellipsis'].sub(r'BAB \1', text)

        # Fix specific corrupted patterns
        corrupted_mappings = {
            'BABII': 'BAB II', 'BABIII': 'BAB III', 'BABIV': 'BAB IV',
            'BABV': 'BAB V', 'BABVI': 'BAB VI', 'BABVII': 'BAB VII',
            'BABVIII': 'BAB VIII', 'BABIX': 'BAB IX', 'BABX': 'BAB X',
            'BABXI': 'BAB XI', 'BABXII': 'BAB XII', 'BABXIII': 'BAB XIII',
            'BABXIV': 'BAB XIV', 'BABXV': 'BAB XV', 'BABXVI': 'BAB XVI',
            'BABXVII': 'BAB XVII', 'BABXVIII': 'BAB XVIII', 'BABXIX': 'BAB XIX',
            'BABXX': 'BAB XX', 'BABXXI': 'BAB XXI', 'BABXXII': 'BAB XXII'
        }

        for corrupted, fixed in corrupted_mappings.items():
            text = text.replace(corrupted, fixed)

        return text

    def apply_ocr_corrections(self, text: str) -> str:
        """
        Apply OCR corrections using compiled patterns.

        Args:
            text: Text with OCR errors

        Returns:
            Text with OCR corrections applied
        """
        text = self.ocr_correction_patterns['republic_variations'].sub('REPUBLIK INDONESIA', text)
        text = self.ocr_correction_patterns['president_variations'].sub('PRESIDEN REPUBLIK', text)
        text = self.ocr_correction_patterns['garbled_headers'].sub('', text)
        return text

    def fix_legal_reference_formatting(self, text: str) -> str:
        """
        Fix legal reference formatting to keep references together.

        Args:
            text: Text with potentially broken legal references

        Returns:
            Text with fixed legal reference formatting
        """
        # Keep "Pasal X ayat Y" together
        text = re.sub(
            r'(Pasal\s+\d+[A-Za-z]?)\s*\n\s*(ayat\s*\([^)]+\))',
            r'\1 \2',
            text,
            flags=re.IGNORECASE
        )

        # Keep "Pasal X huruf Y" together
        text = re.sub(
            r'(Pasal\s+\d+[A-Za-z]?)\s*\n\s*(huruf\s+[a-z])',
            r'\1 \2',
            text,
            flags=re.IGNORECASE
        )

        # Keep "Pasal X angka Y" together
        text = re.sub(
            r'(Pasal\s+\d+[A-Za-z]?)\s*\n\s*(angka\s+\d+)',
            r'\1 \2',
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r'(Pasal\s+\d+[A-Za-z]?)\s*\n\s*(huruf\s+[a-z])\s*\n\s*(\d{1,3})',
            r'\1 \2 \3',
            text,
            flags=re.IGNORECASE
        )

        # Keep reference ranges together: "Pasal X sampai dengan Pasal Y"
        text = re.sub(
            r'(Pasal\s+\d+[A-Za-z]?)\s*\n\s*(sampai\s+dengan)\s*\n\s*(Pasal\s+\d+[A-Za-z]?)',
            r'\1 \2 \3',
            text,
            flags=re.IGNORECASE
        )

        # Keep "dimaksud dalam Pasal X" together
        text = re.sub(
            r'(dimaksud\s+dalam)\s*\n\s*(Pasal\s+\d+[A-Za-z]?)',
            r'\1 \2',
            text,
            flags=re.IGNORECASE
        )

        # Keep "sebagaimana dimaksud dalam Pasal X ayat Y" together
        text = re.sub(
            r'(sebagaimana\s+dimaksud\s+dalam)\s*\n\s*(Pasal\s+\d+[A-Za-z]?(?:\s+ayat\s*\([^)]+\))?)',
            r'\1 \2',
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r'(sebagaimana\s+dimaksud\s+dalam)\s*\n\s*'
            r'(Pasal\s+\d+[A-Za-z]?'
            r'(?:\s+ayat\s+\d+)?'
            r'(?:\s+huruf\s+\w+)?'
            r'(?:\s+angka\s+\d+)?'
            r')',
            r'\1 \2',
            text,
            flags=re.IGNORECASE
        )

        return text

    def fix_ayat_formatting(self, text: str) -> str:
        """
        Fix ayat formatting issues.

        Args:
            text: Text with ayat formatting issues

        Returns:
            Text with fixed ayat formatting
        """
        # Fix "Pasal 1 (1)" -> "Pasal 1\n(1)" (ayat should be on new line)
        text = re.sub(r'(Pasal\s+\d+[A-Za-z]?)\s+\((\d+)\)', r'\1\n(\2)', text)

        # Fix broken ayat: "(21" -> "(2)"
        text = re.sub(r'\((\d)[1l]\s', r'(\1) ', text)

        # Fix spaced parentheses: "( 1 )" -> "(1)"
        text = re.sub(r'\(\s*(\d+)\s*\)', r'(\1)', text)

        return text

    def fix_broken_legal_patterns(self, text: str) -> str:
        """
        Fix pasal patterns broken across lines.

        Args:
            text: Text with broken legal patterns

        Returns:
            Text with fixed legal patterns
        """
        # Fix "Pasal\n390" -> "Pasal 390" (critical for missing numbers)
        text = re.sub(r'Pasal\s*\n\s*(\d+[A-Za-z]?)', r'Pasal \1', text, flags=re.MULTILINE)

        # Fix "Pasal 3\n90" -> "Pasal 390" (page break splitting numbers)
        text = re.sub(r'Pasal\s+(\d+)\s*\n\s*(\d+)', r'Pasal \1\2', text, flags=re.MULTILINE)

        # Fix concatenated headers: "text.BAB IV" -> "text.\n\nBAB IV"
        text = re.sub(r'(\w)\.\s*(BAB\s+[IVXLCDM]+)', r'\1.\n\n\2', text)
        text = re.sub(r'(\w)\.\s*(BAGIAN\s+[A-Za-z]+)', r'\1.\n\n\2', text)
        text = re.sub(r'(\w)\.\s*(Paragraf\s*\d*)', r'\1.\n\n\2', text)

        # Fix pasal with trailing content: "Pasal 1 Tindak" -> separate properly
        text = re.sub(r'(Pasal\s+\d+[A-Za-z]?)\s+([A-Z][a-z]+)', r'\1\n\2', text)

        return text

    def fix_pasal_numbers(self, text: str) -> str:
        """
        Fix Pasal number formatting issues including OCR errors and spacing.

        Args:
            text: Text with Pasal formatting issues

        Returns:
            Text with fixed Pasal numbers
        """
        # Fix "Pasa7" typos first
        text = self.pasal_fix_patterns['pasa_typo'].sub(r'Pasal \1', text)

        # Fix complex spacing like "Pasal 1 1 I" -> "Pasal 111"
        text = self.pasal_fix_patterns['complex_spacing'].sub(r'Pasal \1\2\3', text)

        # Fix OCR in middle like "Pasal l7l" -> "Pasal 171"
        def fix_ocr_in_middle(match):
            char1 = match.group(1).replace('l', '1').replace('I', '1')
            num = match.group(2)
            char3 = match.group(3).replace('l', '1').replace('I', '1')
            return f'Pasal {char1}{num}{char3}'

        text = self.pasal_fix_patterns['ocr_in_middle'].sub(fix_ocr_in_middle, text)

        # Fix space in numbers like "Pasal 2 I" -> "Pasal 21"
        def fix_space_in_number(match):
            num1 = match.group(1)
            char = match.group(2).replace('l', '1').replace('I', '1')
            return f'Pasal {num1}{char}'

        text = self.pasal_fix_patterns['space_in_number'].sub(fix_space_in_number, text)

        # Fix embedded OCR characters
        def fix_ocr_digits(match):
            pasal_num = match.group(1)
            pasal_num = pasal_num.replace('l', '1').replace('I', '1').replace('O', '0')
            return f'Pasal {pasal_num}'

        text = self.pasal_fix_patterns['embedded_ocr_full'].sub(fix_ocr_digits, text)

        # Fix ayat OCR errors like "l2l" -> "(2)"
        text = self.pasal_fix_patterns['ayat_ocr'].sub(r'(\1)', text)

        # Fix broken ayat like "(21" -> "(2)"
        text = self.pasal_fix_patterns['ayat_broken'].sub(r'(\1)', text)

        # Fix missing newlines between Pasal and content (handle OCR O as well)
        text = re.sub(r'(Pasal\s+\d+[A-Za-z]?)\s+([A-Z][a-z]+)', r'\1\n\2', text)

        # Fix remaining no-space patterns
        text = self.pasal_fix_patterns['no_space_digits'].sub(r'Pasal \1', text)

        return text

    def get_pattern_statistics(self) -> Dict[str, int]:
        """
        Get statistics about compiled patterns.

        Returns:
            Dictionary with pattern counts and information
        """
        return {
            'chapter_patterns': len(self.chapter_patterns),
            'article_patterns': len(self.article_patterns),
            'verse_patterns': len(self.verse_patterns),
            'hierarchical_patterns': len(self.hierarchical_patterns),
            'numbered_patterns': len(self.numbered_patterns),
            'legal_reference_patterns': len(self.legal_reference_patterns),
            'document_type_patterns': len(self.document_type_patterns),
            'definition_patterns': len(self.definition_patterns),
            'noise_patterns': len(self.noise_patterns),
            'whitespace_patterns': len(self.whitespace_patterns),
            'ocr_correction_patterns': len(self.ocr_correction_patterns),
            'bab_fix_patterns': len(self.bab_fix_patterns),
            'pasal_fix_patterns': len(self.pasal_fix_patterns)
        }
