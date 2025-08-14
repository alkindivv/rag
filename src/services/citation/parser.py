"""
Citation parser for Indonesian legal documents.

Handles explicit legal references like:
- UU 8/2019 Pasal 6 ayat (2) huruf b
- PP No. 45 Tahun 2020 Pasal 12
- Pasal 15 ayat (1)
- UU No. 4/2009 Pasal 121 Ayat 1
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from ...config.settings import settings
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CitationMatch:
    """Structured citation match result."""

    # Document identification
    doc_form: Optional[str] = None  # UU, PP, PERPU, etc.
    doc_number: Optional[str] = None  # 8, 45, etc.
    doc_year: Optional[int] = None  # 2019, 2020, etc.

    # Unit hierarchy
    pasal_number: Optional[str] = None  # 6, 121, 15A, etc.
    ayat_number: Optional[str] = None  # 1, 2, etc.
    huruf_letter: Optional[str] = None  # a, b, c, etc.
    angka_number: Optional[str] = None  # 1, 2, 3, etc.

    # Match metadata
    confidence: float = 0.0  # 0.0-1.0 confidence score
    matched_text: str = ""  # Original matched text
    is_complete: bool = False  # Has both document and unit info

    def to_dict(self) -> Dict[str, Union[str, int, float, bool]]:
        """Convert to dictionary for JSON serialization."""
        return {
            'doc_form': self.doc_form,
            'doc_number': self.doc_number,
            'doc_year': self.doc_year,
            'pasal_number': self.pasal_number,
            'ayat_number': self.ayat_number,
            'huruf_letter': self.huruf_letter,
            'angka_number': self.angka_number,
            'confidence': self.confidence,
            'matched_text': self.matched_text,
            'is_complete': self.is_complete
        }


class LegalCitationParser:
    """
    Parser for Indonesian legal citations with comprehensive pattern matching.

    Supports various citation formats commonly used in Indonesian legal documents,
    from formal citations to partial references.
    """

    # Citation patterns ordered by specificity (most specific first)
    CITATION_PATTERNS = [
        # Complete citations with document and unit reference
        {
            'name': 'complete_uu_formal',
            'pattern': r'(?:UU|Undang-Undang)\s+(?:No\.?\s*)?(\d+)(?:/|\s+[Tt]ahun\s+)(\d{4})\s+[Pp]asal\s+(\d+[A-Z]*)\s*(?:[Aa]yat\s*\(?(\d+)\)?)?(?:\s*[Hh]uruf\s*([a-z]))?(?:\s*[Aa]ngka\s*(\d+))?',
            'confidence': 0.95,
            'groups': ['doc_number', 'doc_year', 'pasal_number', 'ayat_number', 'huruf_letter', 'angka_number'],
            'doc_form': 'UU'
        },
        {
            'name': 'complete_pp_formal',
            'pattern': r'(?:PP|Peraturan\s+Pemerintah)\s+(?:No\.?\s*)?(\d+)(?:/|\s+[Tt]ahun\s+)(\d{4})\s+[Pp]asal\s+(\d+[A-Z]*)\s*(?:[Aa]yat\s*\(?(\d+)\)?)?(?:\s*[Hh]uruf\s*([a-z]))?(?:\s*[Aa]ngka\s*(\d+))?',
            'confidence': 0.95,
            'groups': ['doc_number', 'doc_year', 'pasal_number', 'ayat_number', 'huruf_letter', 'angka_number'],
            'doc_form': 'PP'
        },
        {
            'name': 'complete_perpu_formal',
            'pattern': r'(?:PERPU|Peraturan\s+Pemerintah\s+Pengganti\s+Undang-Undang)\s+(?:No\.?\s*)?(\d+)(?:/|\s+[Tt]ahun\s+)(\d{4})\s+[Pp]asal\s+(\d+[A-Z]*)\s*(?:[Aa]yat\s*\(?(\d+)\)?)?(?:\s*[Hh]uruf\s*([a-z]))?(?:\s*[Aa]ngka\s*(\d+))?',
            'confidence': 0.95,
            'groups': ['doc_number', 'doc_year', 'pasal_number', 'ayat_number', 'huruf_letter', 'angka_number'],
            'doc_form': 'PERPU'
        },

        # Short form citations
        {
            'name': 'short_uu_citation',
            'pattern': r'UU\s+(\d+)/(\d{4})\s+[Pp]asal\s+(\d+[A-Z]*)\s*(?:[Aa]yat\s*\(?(\d+)\)?)?(?:\s*[Hh]uruf\s*([a-z]))?(?:\s*[Aa]ngka\s*(\d+))?',
            'confidence': 0.90,
            'groups': ['doc_number', 'doc_year', 'pasal_number', 'ayat_number', 'huruf_letter', 'angka_number'],
            'doc_form': 'UU'
        },
        {
            'name': 'short_pp_citation',
            'pattern': r'PP\s+(\d+)/(\d{4})\s+[Pp]asal\s+(\d+[A-Z]*)\s*(?:[Aa]yat\s*\(?(\d+)\)?)?(?:\s*[Hh]uruf\s*([a-z]))?(?:\s*[Aa]ngka\s*(\d+))?',
            'confidence': 0.90,
            'groups': ['doc_number', 'doc_year', 'pasal_number', 'ayat_number', 'huruf_letter', 'angka_number'],
            'doc_form': 'PP'
        },

        # Document-only citations (no specific unit)
        {
            'name': 'document_uu_only',
            'pattern': r'(?:UU|Undang-Undang)\s+(?:No\.?\s*)?(\d+)(?:/|\s+[Tt]ahun\s+)(\d{4})(?!\s+[Pp]asal)',
            'confidence': 0.70,
            'groups': ['doc_number', 'doc_year'],
            'doc_form': 'UU'
        },
        {
            'name': 'document_pp_only',
            'pattern': r'(?:PP|Peraturan\s+Pemerintah)\s+(?:No\.?\s*)?(\d+)(?:/|\s+[Tt]ahun\s+)(\d{4})(?!\s+[Pp]asal)',
            'confidence': 0.70,
            'groups': ['doc_number', 'doc_year'],
            'doc_form': 'PP'
        },

        # Unit-only citations (no document context)
        {
            'name': 'pasal_only_detailed',
            'pattern': r'[Pp]asal\s+(\d+[A-Z]*)\s*(?:[Aa]yat\s*\(?(\d+)\)?)?(?:\s*[Hh]uruf\s*([a-z]))?(?:\s*[Aa]ngka\s*(\d+))?',
            'confidence': 0.60,
            'groups': ['pasal_number', 'ayat_number', 'huruf_letter', 'angka_number'],
            'doc_form': None
        },
        {
            'name': 'ayat_only',
            'pattern': r'[Aa]yat\s*\(?(\d+)\)?(?:\s*[Hh]uruf\s*([a-z]))?(?:\s*[Aa]ngka\s*(\d+))?',
            'confidence': 0.40,
            'groups': ['ayat_number', 'huruf_letter', 'angka_number'],
            'doc_form': None
        },
        {
            'name': 'huruf_only',
            'pattern': r'[Hh]uruf\s*([a-z])(?:\s*[Aa]ngka\s*(\d+))?',
            'confidence': 0.30,
            'groups': ['huruf_letter', 'angka_number'],
            'doc_form': None
        },

        # Flexible patterns for various formats
        {
            'name': 'flexible_citation',
            'pattern': r'(\w+)\s+(\d+)/(\d{4})\s*[•·]?\s*[Pp]asal\s+(\d+[A-Z]*)',
            'confidence': 0.75,
            'groups': ['doc_form', 'doc_number', 'doc_year', 'pasal_number'],
            'doc_form': None  # Will be extracted from group
        }
    ]

    # Valid document forms
    VALID_DOC_FORMS = {
        'UU', 'PP', 'PERPU', 'PERPRES', 'POJK', 'PERMEN', 'PERDA', 'SE'
    }

    def __init__(self):
        """Initialize citation parser with compiled regex patterns."""
        self.compiled_patterns = []

        for pattern_config in self.CITATION_PATTERNS:
            try:
                compiled = re.compile(pattern_config['pattern'], re.IGNORECASE | re.MULTILINE)
                self.compiled_patterns.append({
                    **pattern_config,
                    'compiled': compiled
                })
            except re.error as e:
                logger.warning(f"Failed to compile pattern {pattern_config['name']}: {e}")

        logger.info(f"Initialized LegalCitationParser with {len(self.compiled_patterns)} patterns")

    def parse_citation(self, text: str) -> List[CitationMatch]:
        """
        Parse text for legal citations.

        Args:
            text: Input text to parse for citations

        Returns:
            List of CitationMatch objects ordered by confidence
        """
        if not text or not text.strip():
            return []

        matches = []
        text = text.strip()

        for pattern_config in self.compiled_patterns:
            compiled_pattern = pattern_config['compiled']

            for match in compiled_pattern.finditer(text):
                citation = self._extract_citation(match, pattern_config)
                if citation and self._validate_citation(citation):
                    matches.append(citation)

        # Remove duplicates and sort by confidence
        unique_matches = self._deduplicate_matches(matches)
        return sorted(unique_matches, key=lambda x: x.confidence, reverse=True)

    def is_explicit_citation(self, text: str, min_confidence: float = 0.60) -> bool:
        """
        Check if text contains explicit legal citations.

        Args:
            text: Input text to check
            min_confidence: Minimum confidence threshold

        Returns:
            True if explicit citations found above threshold
        """
        matches = self.parse_citation(text)
        return any(match.confidence >= min_confidence for match in matches)

    def get_best_match(self, text: str) -> Optional[CitationMatch]:
        """
        Get the most complete and highest confidence citation match.

        Args:
            text: Input text to parse

        Returns:
            Best CitationMatch or None if no valid matches
        """
        matches = self.parse_citation(text)
        if not matches:
            return None

        # Priority 1: Try to merge complementary matches
        merged_match = self._merge_complementary_matches(matches)
        if merged_match:
            return merged_match

        # Priority 2: Select most complete match (not just highest confidence)
        return self._select_most_complete_match(matches)

    def _merge_complementary_matches(self, matches: List[CitationMatch]) -> Optional[CitationMatch]:
        """Merge document info with unit info from separate matches."""
        doc_info = None
        unit_info = None

        for match in matches:
            # Document info match
            if match.doc_form and match.doc_number and match.doc_year:
                if not doc_info or match.confidence > doc_info.confidence:
                    doc_info = match

            # Unit info match
            if match.pasal_number or match.ayat_number or match.huruf_letter:
                if not unit_info or match.confidence > unit_info.confidence:
                    unit_info = match

        # Merge if we have both components
        if doc_info and unit_info:
            merged = CitationMatch(
                doc_form=doc_info.doc_form,
                doc_number=doc_info.doc_number,
                doc_year=doc_info.doc_year,
                pasal_number=unit_info.pasal_number,
                ayat_number=unit_info.ayat_number,
                huruf_letter=unit_info.huruf_letter,
                angka_number=unit_info.angka_number,
                confidence=(doc_info.confidence + unit_info.confidence) / 2,
                matched_text=f"{unit_info.matched_text} {doc_info.matched_text}",
                is_complete=True
            )
            return merged

        return None

    def _select_most_complete_match(self, matches: List[CitationMatch]) -> CitationMatch:
        """Select match with most complete information."""
        def completeness_score(match):
            score = 0
            if match.doc_form: score += 3
            if match.doc_number: score += 3
            if match.doc_year: score += 3
            if match.pasal_number: score += 2
            if match.ayat_number: score += 1
            if match.huruf_letter: score += 1
            if match.angka_number: score += 1
            return score

        # Sort by completeness score, then confidence
        sorted_matches = sorted(matches,
            key=lambda m: (completeness_score(m), m.confidence),
            reverse=True
        )

        return sorted_matches[0]

    def _extract_citation(self, match: re.Match, pattern_config: Dict) -> Optional[CitationMatch]:
        """Extract CitationMatch from regex match."""
        try:
            groups = match.groups()
            matched_text = match.group(0)

            citation = CitationMatch(
                matched_text=matched_text,
                confidence=pattern_config['confidence']
            )

            # Extract values based on group mapping
            group_names = pattern_config['groups']
            for i, group_name in enumerate(group_names):
                if i < len(groups) and groups[i]:
                    value = groups[i].strip()

                    if group_name == 'doc_form':
                        # Normalize document form
                        citation.doc_form = self._normalize_doc_form(value)
                    elif group_name == 'doc_number':
                        citation.doc_number = value
                    elif group_name == 'doc_year':
                        try:
                            citation.doc_year = int(value)
                        except ValueError:
                            continue
                    elif group_name == 'pasal_number':
                        citation.pasal_number = value
                    elif group_name == 'ayat_number':
                        citation.ayat_number = value
                    elif group_name == 'huruf_letter':
                        citation.huruf_letter = value.lower()
                    elif group_name == 'angka_number':
                        citation.angka_number = value

            # Set doc_form from pattern config if not extracted
            if not citation.doc_form and pattern_config['doc_form']:
                citation.doc_form = pattern_config['doc_form']

            # Determine if citation is complete
            citation.is_complete = bool(
                citation.doc_form and citation.doc_number and
                citation.doc_year and citation.pasal_number
            )

            return citation

        except Exception as e:
            logger.warning(f"Error extracting citation from match: {e}")
            return None

    def _normalize_doc_form(self, doc_form: str) -> str:
        """Normalize document form to standard abbreviation."""
        doc_form = doc_form.upper().strip()

        # Handle common variations
        mappings = {
            'UNDANG-UNDANG': 'UU',
            'UNDANG': 'UU',
            'PERATURAN PEMERINTAH': 'PP',
            'PERATURAN PEMERINTAH PENGGANTI UNDANG-UNDANG': 'PERPU',
            'PERATURAN PRESIDEN': 'PERPRES',
            'PERATURAN MENTERI': 'PERMEN',
            'PERATURAN DAERAH': 'PERDA',
            'SURAT EDARAN': 'SE'
        }

        return mappings.get(doc_form, doc_form)

    def _validate_citation(self, citation: CitationMatch) -> bool:
        """Validate citation has reasonable values."""
        if not citation:
            return False

        # Check doc_form is valid if present
        if citation.doc_form and citation.doc_form not in self.VALID_DOC_FORMS:
            return False

        # Check year is reasonable if present
        if citation.doc_year and (citation.doc_year < 1945 or citation.doc_year > 2030):
            return False

        # Must have at least some content
        has_content = any([
            citation.doc_form,
            citation.pasal_number,
            citation.ayat_number,
            citation.huruf_letter,
            citation.angka_number
        ])

        return has_content

    def _deduplicate_matches(self, matches: List[CitationMatch]) -> List[CitationMatch]:
        """Remove duplicate matches based on extracted content."""
        if not matches:
            return []

        unique_matches = []
        seen_signatures = set()

        for match in matches:
            # Create signature for deduplication
            signature = (
                match.doc_form,
                match.doc_number,
                match.doc_year,
                match.pasal_number,
                match.ayat_number,
                match.huruf_letter,
                match.angka_number
            )

            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_matches.append(match)

        return unique_matches


# Module-level convenience functions
_parser = None

def get_citation_parser() -> LegalCitationParser:
    """Get singleton citation parser instance."""
    global _parser
    if _parser is None:
        _parser = LegalCitationParser()
    return _parser


def parse_citation(text: str) -> List[CitationMatch]:
    """Parse text for legal citations using singleton parser."""
    return get_citation_parser().parse_citation(text)


def is_explicit_citation(text: str, min_confidence: float = 0.60) -> bool:
    """Check if text contains explicit legal citations."""
    return get_citation_parser().is_explicit_citation(text, min_confidence)


def get_best_citation_match(text: str) -> Optional[CitationMatch]:
    """Get the best citation match from text."""
    return get_citation_parser().get_best_match(text)
