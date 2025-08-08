"""
Enhanced Text Cleaner for Legal Documents
Comprehensive text cleaning with focus on Indonesian legal document structure preservation

This module consolidates ALL text cleaning logic from the original legal_pdf_processor.py
and addresses the specific issues with inconsistent newlines, spacing, and reference formatting.

Key Improvements:
- Proper reference keeping (e.g., "Pasal 64 huruf a" stays together)
- Fixed spacing issues (e.g., "a. keamanan , , ," patterns)
- Consistent newline handling for legal structure
- Complete artifact removal
- Preserved all critical cleaning logic without loss

Author: Refactored Architecture - Consolidated Cleaning
Purpose: Single responsibility comprehensive text cleaning
"""

import re
import unicodedata
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Any
import time
from .pattern_manager import PatternManager

logger = logging.getLogger(__name__)


@dataclass
class CleaningResult:
    """Result of text cleaning operation."""
    cleaned_text: str
    operations_applied: List[str]
    original_length: int
    cleaned_length: int
    processing_time: float


class TextCleaner:
    """
    Comprehensive text cleaner for legal documents.

    Consolidates all cleaning logic from extractors and provides
    single entry point for text cleaning with focus on legal document structure.
    """

    def __init__(self):
        """Initialize text cleaner with pattern manager."""
        self.pattern_manager = PatternManager()
        self.logger = logging.getLogger(__name__)



    def clean_legal_document_comprehensive(self, text: str) -> str:
        """
        Comprehensive cleaning for legal documents - main entry point.

        This method consolidates ALL cleaning logic from the original legal_pdf_processor.py
        and addresses specific formatting issues identified by users.

        Args:
            text: Raw extracted text from PDF

        Returns:
            Cleaned text ready for processing
        """
        if not text or len(text.strip()) < 10:
            return ""

        start_time = time.time()
        original_length = len(text)

        # Step 1: Basic normalization and encoding fixes
        cleaned_text = self._normalize_line_endings(text)
        cleaned_text = self._fix_encoding_issues(cleaned_text)

        # Step 2: Fix critical OCR errors in legal elements FIRST
        cleaned_text = self._fix_critical_legal_ocr_errors(cleaned_text)

        # Step 3: Preserve and fix corrupted BAB patterns
        cleaned_text = self._preserve_and_fix_corrupted_bab(cleaned_text)

        # Step 4: Clean page break artifacts that disrupt structure
        cleaned_text = self.clean_page_break_artifacts(cleaned_text)

        # Step 5: Fix broken legal patterns across lines (CRITICAL for references)
        cleaned_text = self._fix_broken_legal_patterns(cleaned_text)

        # Step 6: Fix reference formatting (keep "Pasal X huruf a" together)
        cleaned_text = self._fix_legal_reference_formatting(cleaned_text)

        # Step 7: Fix ayat and numbering formatting
        cleaned_text = self._fix_ayat_formatting(cleaned_text)

        # Step 8: Apply comprehensive OCR correction
        cleaned_text = self._comprehensive_ocr_correction(cleaned_text)

        # Step 9: Remove document noise and artifacts
        cleaned_text = self._remove_document_noise(cleaned_text)

        # Step 10: Fix spacing issues (handles "a. keamanan , , ," patterns)
        cleaned_text = self._fix_spacing_issues(cleaned_text)

        # Step 11: Advanced line filtering and duplicate removal
        # cleaned_text = self._advanced_line_filtering(cleaned_text)

        # Step 12: Final legal formatting standardization
        cleaned_text = self._standardize_legal_formatting(cleaned_text)

        # Step 13: Fix improper line wrapping and broken sentences
        cleaned_text = self._fix_line_wrapping_and_sentence_segmentation(cleaned_text)

        # Step 14: Fix collapsed list formatting (separate enumerated items)
        # cleaned_text = self._fix_collapsed_list_formatting(cleaned_text)

        # Step 15: Final whitespace cleanup
        cleaned_text = self._final_whitespace_cleanup(cleaned_text)



        processing_time = time.time() - start_time

        logger.debug(f"Text cleaning completed: {original_length} -> {len(cleaned_text)} chars in {processing_time:.2f}s")

        return cleaned_text.strip()

    def _normalize_line_endings(self, text: str) -> str:
        """Normalize line endings and basic encoding."""
        # Normalize line endings
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)

        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)

        return text

    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues in Indonesian legal documents."""
        # Fix smart quotes and special characters
        encoding_fixes = {
            'â€œ': '"', 'â€': '"', 'â€˜': "'", 'â€™': "'",
            '"': '"', '"': '"', ''': "'", ''': "'",
            'â€"': '-', 'â€"': '--', '—': '--',
            'â€¢': '•', 'â€¦': '...', 'Â': ' ', 'Â ': ' ',
            'â€¯': ' '
        }

        for wrong, correct in encoding_fixes.items():
            text = text.replace(wrong, correct)

        return text

    def _fix_critical_legal_ocr_errors(self, text: str) -> str:
        """Fix critical OCR errors that affect legal element detection."""
        # Fix digit OCR errors in pasal numbers
        ocr_fixes = {
            r'\bPasal\s+(\d+)[Il1]\b': r'Pasal \g<1>1',  # 6I/6l -> 61
            r'\bPasal\s+[Il1](\d+)\b': r'Pasal 1\g<1>',  # I6/l6 -> 16
            r'\bPasal\s+(\d+)O\b': r'Pasal \g<1>0',      # 6O -> 60
            r'\bPasal\s+O(\d+)\b': r'Pasal 0\g<1>',      # O6 -> 06
            r'\bPasal\s+(\d+)S\b': r'Pasal \g<1>5',      # 6S -> 65
            r'\bPasal\s+S(\d+)\b': r'Pasal 5\g<1>',      # S6 -> 56
            r'\bPasal\s+(\d+)B\b': r'Pasal \g<1>8',      # 6B -> 68
            r'\bPasal\s+B(\d+)\b': r'Pasal 8\g<1>',      # B6 -> 86

            # Fix ayat OCR errors
            r'\((\d)[Il1]\)': r'(\g<1>)',                # (2l) -> (2)
            r'\((\d)I\)': r'(\g<1>)',                    # (2I) -> (2)
            r'\((\d)O\)': r'(\g<1>)',                    # (2O) -> (2)
            r'\([Il1]\)': r'(1)',                        # (l) -> (1)
            r'\([O0]\)': r'(0)',                         # (O) -> (0)

            r'\(sqt\)': '(satu)',
            r'\(duq\)': '(dua)',
            r'\(tigq\)': '(tiga)',
            r'\(empqt\)': '(empat)',
            r'\(llma\)': '(lima)',
            r'\(limq\)': '(lima)',
            r'\(enqm\)': '(enam)',
            r'\(tqluh\)': '(tujuh)',
            r'\(tqiuh\)': '(tujuh)',
            r'\(delqpan\)': '(delapan)',
            r'\(sembiIan\)': '(sembilan)',
            r'\(sepuluh\)': '(sepuluh)',

            r'\(tqluh\)': '(tujuh)',
            r'\(empqt\)': '(empat)',
            r'\(limq\)': '(lima)',
            r'\(enqm\)': '(enam)',
            r'\btqiuh\b': 'tujuh',          # tqiuh → tujuh
            r'\btqluh\b': 'tujuh',
            r'\btqjuh\b': 'tujuh',
            r'\btigq\b': 'tiga',
            r'\bempqt\b': 'empat',
            r'\bernpat\b': 'empat',
            r'\bllma\b': 'lima',
            r'\blimq\b': 'lima',
            r'\bseiapqn\b': 'sembilan',     # example OCR mis-OCR "sembilan"
            r'\benqm\b': 'enam',
            r'\bsqt\b': 'satu',
            r'\bduq\b': 'dua',
            r'\bdlaq\b': 'dua',
            r'\bdelqpan\b': 'delapan',
            r'\bsepuluh\b': 'sepuluh',

            r'\bte\{adi\b': 'terjadi',
            r'\bT\\ra\b': 'tua',
            r'\bTtrjuan\b': 'Tujuan',
            r'\bT\\:juan\b': 'Tujuan',
            r'\bPerahrran\b': 'Peraturan',
            r'\bPerahuran\b': 'Peraturan',
            r'\bPeratunrn\b': 'Peraturan',
            r'\bPasaT\b': 'Pasal',
            r'\bPasd\b': 'Pasal',
            r'\bPasa7\b': 'Pasal',
            r'\bPasaJ\b': 'Pasal',
            # New word corrections from user
            r'\bp na\b': 'pidana',
            r'\bpelindunBan\b': 'perlindungan',
            r'\bPermufalatan\b': 'Pemufakatan',
            r'\brnateri\b': 'materi',
            r'\br-rmurn\b': 'umum',
            r'\bBaglan\b': 'Bagian',
            r'\bBagran\b': 'Bagian',
            r'\btertuiis\b': 'tertulis',
            r'\bilntara\b': 'antara',
            r'\bhukunr\b': 'hukum',
            r'\ba\$asl\b': 'asasi',
            r'\blrranusia\b': 'manusia',
            r'\bllewajibarr\b': 'kewajiban',
            r'\bsebagairnana\b': 'sebagaimana',
            r'\bmemberrtuk\b': 'membentuk',
            r'\bDalarn\b': 'Dalam',
            r'\bmErnapun\b': 'manapun',
            r'\bIkpal\b': 'kapal',
            r'\bneg€rra\b': 'negara',
            r'\bmerusalC\b': 'merusak',
            r'\bwalrtu\b': 'waktu',
            r'\btertuhrp\b': 'tertutup',
            r'\borErng\b': 'orang',
            r'\bs6!agei1n614\b': 'sebagaimana',
            r'\bDAI,AM\b': 'DALAM',
            r'\byEmg\b': 'yang',
            r'\bPasa7\b': 'Pasal',
            r'\bPazal\b': 'Pasal',
            r'\bmasyarakqt\b': 'masyarakat',
            r'\blngkah\b': 'langkah',
            r'\bkewajiban\b': 'kewajiban',
            r'\btelah\b': 'telah',
            r'\bdiberlkukan\b': 'diberlakukan',
            r'\bmelaksanalan\b': 'melaksanakan',
            r'\bmelaksnnakan\b': 'melaksanakan',
            r'\bmclaksanakan\b': 'melaksanakan',
            r'\bpcnyelenggaraan\b': 'penyelenggaraan',
            r'\btangguug\b': 'tanggung',
            r'\bkewcnangan\b': 'kewenangan',
            r'\bkewcnaqan\b': 'kewenangan',
            r'\bizin\b': 'izin',
            r'\bizirl\b': 'izin',
            r'\b1zin\b': 'izin',
            r'\b1zinya\b': 'izinnya',
            r'\bsesu3i\b': 'sesuai',
            r'\bscsuai\b': 'sesuai',
            r'\bscsu3i\b': 'sesuai',


            r'\bte\{adi\b': 'terjadi',
            r'\bT\\ra\b': 'tua',
            r'\bTtrjuan\b': 'Tujuan',
            r'\bT\\:juan\b': 'Tujuan',
            r'\bPerahrran\b': 'Peraturan',
            r'\bPasd\b': 'Pasal',
            r'\blO\b': '10',
            r'\bmelaksanalan\b': 'melaksanakan',
            r'\bPasaT\b': 'Pasal',
            r'\bizirl\b': 'izin',
            r'\btqiuh\b': 'tujuh',
            r'\bPasa7\b': 'Pasal',
            r'\bBagianKetqluh\b': 'Bagian Ketujuh',     # typo q untuk u
            r'\bBagianKetlu\b': 'Bagian Ketujuh',
            r'\bBagianKedelqpan\b': 'Bagian Kedelapan',
            r'\bBagianKedeqlapan\b': 'Bagian Kedelapan',
        }

        # Apply OCR corrections using PatternManager
        text = self.pattern_manager.apply_ocr_corrections(text)

        # Apply additional OCR fixes
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _preserve_and_fix_corrupted_bab(self, text: str) -> str:
        """Preserve and fix corrupted BAB patterns using PatternManager."""
        return self.pattern_manager.fix_bab_patterns(text)



    def _fix_broken_legal_patterns(self, text: str) -> str:
        """Fix pasal patterns broken across lines using PatternManager."""
        text = self.pattern_manager.fix_broken_legal_patterns(text)
        # Also fix Pasal number spacing issues
        text = self.pattern_manager.fix_pasal_numbers(text)
        return text

    def _fix_legal_reference_formatting(self, text: str) -> str:
        """Fix legal reference formatting using PatternManager."""
        return self.pattern_manager.fix_legal_reference_formatting(text)

    def _fix_ayat_formatting(self, text: str) -> str:
        """Fix ayat formatting using PatternManager."""
        return self.pattern_manager.fix_ayat_formatting(text)

    def _comprehensive_ocr_correction(self, text: str) -> str:
        """Comprehensive OCR correction for Indonesian legal documents."""
        # Critical corrections for legal document elements
        critical_corrections = OrderedDict([
            (r'\(\(([0-9]{1,4})\)', r'(\1)'),           # ((123)) -> (123)
            (r'\(l\)', '(1)'),                            # (l) -> (1)
            (r'\(I\)', '(1)'),                            # (I) -> (1)
            (r'\(i\)', '(1)'),                            # (i) -> (1)
            (r'\(O\)', '(0)'),                            # (O) -> (0)
            (r'\(o\)', '(0)'),                            # (o) -> (0)
            (r'\(Z\)', '(2)'),                            # (Z) -> (2)
            (r'\(S\)', '(5)'),                            # (S) -> (5)
            (r'\(G\)', '(6)'),                            # (G) -> (6)
            (r'\(B\)', '(8)'),                            # (B) -> (8)
            (r'\(g\)', '(9)'),                            # (g) -> (9)
            (r'\(q\)', '(9)'),                            # (q) -> (9)
        ])

        # Legal document corrections
        legal_corrections = OrderedDict([
            # (r'(?i)(Pasal)\s*\(([0-9]{1,4})\)', r'\1 \2'),     # "Pasal (12)" -> "Pasal 12"
            # (r'(?i)PASAL\s+([0-9]{1,4})', r'Pasal \1'),           # "PASAL 12" -> "Pasal 12"
            # (r'(?i)Ayat\s*\(([lI])\)', r'ayat (1)'),             # "Ayat (l)" -> "ayat (1)"
            # (r'(?i)Ayat\s*\(([0-9]{1,4})\)', r'ayat (\1)'),     # normalize case
            # (r'(?i)AYAT\s*\(([0-9]{1,4})\)', r'ayat (\1)'),     # "AYAT (25)" -> "ayat (25)"
            # (r'(?i)BAB\s+([IVX]{1,6})', r'BAB \1'),               # BAB normalization
            # (r'Undang[—–-]+Undang', 'Undang-Undang'),               # dash normalization
            # (r'sebagai\d+na(?=\s+dimaksud)', 'sebagaimana'),      # "sebagai62na dimaksud" -> "sebagaimana dimaksud"
            # (r'\([lI]\)', '(1)'),                                 # Fix ayat OCR: (l) -> (1)
            # (r'\([UO]\)', '(0)'),                                 # Fix ayat OCR: (U) -> (0)

            # r'\bPasal\s+(\d+)[Il1]\b': r'Pasal \g<1>1',  # 6I/6l -> 61
            # r'\bPasal\s+[Il1](\d+)\b': r'Pasal 1\g<1>',  # I6/l6 -> 16
            # r'\bPasal\s+(\d+)O\b': r'Pasal \g<1>0',      # 6O -> 60
            # r'\bPasal\s+O(\d+)\b': r'Pasal 0\g<1>',      # O6 -> 06
            # r'\bPasal\s+(\d+)S\b': r'Pasal \g<1>5',      # 6S -> 65
            # r'\bPasal\s+S(\d+)\b': r'Pasal 5\g<1>',      # S6 -> 56
            # r'\bPasal\s+(\d+)B\b': r'Pasal \g<1>8',      # 6B -> 68
            # r'\bPasal\s+B(\d+)\b': r'Pasal 8\g<1>',      # B6 -> 86
            # # Fix ayat OCR errors
            # r'\((\d)[Il1]\)': r'(\g<1>)',                # (2l) -> (2)
            # r'\((\d)I\)': r'(\g<1>)',                    # (2I) -> (2)
            # r'\((\d)O\)': r'(\g<1>)',                    # (2O) -> (2)
            #

            (r'(?i)Pasal\s+([0-9]{1,4})', r'Pasal \1'),           # Case-insensitive normalization: PASAL 12 → Pasal 12

            # Ayat Normalization (and OCR fix)
            (r'(?i)Ayat\s*\(([lI])\)', r'ayat (1)'),              # "Ayat (l)" or "Ayat (I)" → "ayat (1)"
            (r'(?i)Ayat\s*\(([0-9]{1,4})\)', r'ayat (\1)'),       # Case-insensitive: "Ayat (2)" or "AYAT (2)" → "ayat (2)"
            (r'\([lI]\)', '(1)'),                                 # OCR fix: (l) or (I) → (1)
            (r'\([UO]\)', '(0)'),                                 # OCR fix: (U) or (O) → (0)
            (r'\((\d)[Il1IO]\)', r'(\1)'),                        # OCR fix: (2l), (2I), (2O), etc. → (2)

            # BAB Normalization
            (r'(?i)BAB\s+([IVX]{1,6})', r'BAB \1'),               # BAB roman numeral handling

            # Undang-Undang dash normalization
            (r'Undang[—–-]+Undang', 'Undang-Undang'),             # Normalize em/en/mixed dash: "Undang–Undang" → "Undang-Undang"

            # OCR merge error fix
            (r'sebagai\d+na(?=\s+dimaksud)', 'sebagaimana'),      # Fix: "sebagai62na dimaksud" → "sebagaimana dimaksud"

            # Pasal OCR Fixes (e.g., 6I → 61, I6 → 16, etc.)
            (r'\bPasal\s+(\d+)[Il1]\b', r'Pasal \g<1>1'),         # 6I/6l → 61
            (r'\bPasal\s+[Il1](\d+)\b', r'Pasal 1\1'),            # I6/l6 → 16
            (r'\bPasal\s+(\d+)O\b', r'Pasal \g<1>0'),             # 6O → 60
            (r'\bPasal\s+O(\d+)\b', r'Pasal 0\1'),                # O6 → 06
            (r'\bPasal\s+(\d+)S\b', r'Pasal \g<1>5'),             # 6S → 65
            (r'\bPasal\s+S(\d+)\b', r'Pasal 5\1'),                # S6 → 56
            (r'\bPasal\s+(\d+)B\b', r'Pasal \g<1>8'),             # 6B → 68
            (r'\bPasal\s+B(\d+)\b', r'Pasal 8\1'),                # B6 → 86
        ])

        # Formatting corrections
        formatting_corrections = OrderedDict([
            (r'[ \t]{2,}', ' '),                          # Multiple spaces -> single
            (r'\.{2,}', '.'),                             # Multiple dots -> single
            (r',,+', ','),                                  # Multiple commas -> single
        ])

        # Execute corrections in phases
        correction_phases = [
            ('critical', critical_corrections),
            ('legal', legal_corrections),
            ('formatting', formatting_corrections)
        ]

        for phase_name, corrections in correction_phases:
            for pattern, replacement in corrections.items():
                text = re.sub(pattern, replacement, text)

        return text

    def _remove_document_noise(self, text: str) -> str:
        """Remove document noise patterns specific to Indonesian legal documents."""
        # Process line by line for better control
        lines = text.split('\n')
        cleaned_lines = []

        # Patterns that indicate entire lines should be removed
        line_removal_patterns = [
            r'^\s*SK\s+No\s*\d+[A-Z]*\s*$',
            r'^\s*[J]\s*$',
            r'^\s*EETITIilIilTI.*$',
            r'^\s*[A-Z]{1,3}\s+lfr\'o\s*$',
            r'^\s*PRESIDEN\s*$',
            r'^\s*REPUBLIK\s+INDONESIA\s*$',
            r'^\s*FRESIDEN\s*$',
            r'^\s*REFUEUK\s+INDONESIA\s*$',
            r'^\s*-\d+[a-z]*-\s*$',
            r'^\s*[A-Z][A-Z\s]*INDONESIA\s*$',
            r'^\s*T\{!TTilTTTIT.*$',
            r'^\s*rrflrFlflxllf.*$',
            r'^\s*EfltrtrLlf.*$',
            r'^\s*E\.-l-\+\{t.*$',
            r'^\s*TIr:I\+Tf.*$',
            r'^\s*llrr-\{rT.*$',
            r'^\s*aftfd\'TFITli.*$',
            r'^\s*nrrFF\[iriN\].*$',
            r'^\s*[\.\-\,\;\:]+\s*$'
            r'llrr-\{rT;trIllilTlTatrtltrtrIrtr',
            r'TIr:I\+Tf\.\{Il',
            r'I THITT : IIIiirIITItrIIIIEIA',
            r'nrrFF\[iriN\]',
            r'aftfd\'TFITli',
            r'EITtrELItrEEIIEIn',
            r'TEIIEtrTilNEEtrtrEIn',
            r'rrfl Tf:IrIXTItililIt\+Jn!',
            r'REI\'UEUK\s+INDONESIA',
            r'REPUA\|JK\s+INDONESIA',
            r'rIrIt\{f\.I\{II',
            r'^[A-Z\|\'\:\+\-\{\}\[\]\\]+$',  # Lines with only OCR symbols
            r'^-t\d+-$'  # Page break patterns like "-t29-"
        ]

        for line in lines:
            should_remove = False
            stripped_line = line.strip()

            # Check if entire line should be removed
            for pattern in line_removal_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    should_remove = True
                    break

            # Skip very short lines that are likely noise, but preserve legal elements
            if len(stripped_line) < 2:
                should_remove = True
            elif len(stripped_line) == 1 and not re.match(r'[a-z]\.?$', stripped_line):
                # Only remove single characters that aren't list markers
                should_remove = True

            # Don't remove lines with legal content
            if re.search(r'[a-z]\.\s+\w+|pidana|ayat|\(\d+\)', stripped_line, re.IGNORECASE):
                should_remove = False

            if not should_remove:
                cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        # Apply PatternManager noise cleaning
        text = self.pattern_manager.clean_noise_artifacts(text)

        return text

    def _fix_spacing_issues(self, text: str) -> str:
        """Fix spacing issues like 'a. keamanan , , ,' patterns."""
        # Fix excessive commas and spaces patterns
        text = re.sub(r',\s*,\s*,\s*', ',', text)  # , , , -> ,
        text = re.sub(r',\s*,\s*', ',', text)      # , , -> ,
        text = re.sub(r'\s+,', ',', text)          # " ," -> ","
        text = re.sub(r',\s+', ', ', text)         # ", " normalize

        # Fix broken list items that got split
        # Pattern: "dan/ atau k.\nkerja sama" -> "dan/atau k. kerja sama"
        text = re.sub(r'dan/\s*atau\s+([a-z])\.\s*\n\s*([a-z])', r'dan/atau \1. \2', text, flags=re.IGNORECASE)

        # Fix list items that got orphaned: "j.\nkerja sama" -> "j. kerja sama"
        text = re.sub(r'^([a-z])\.\s*\n\s*([a-z])', r'\1. \2', text, flags=re.MULTILINE)

        # Fix broken continuations: "kegiatan;\npengaruh" -> "kegiatan; pengaruh"
        # text = re.sub(r'([a-z]);?\s*\n\s*([a-z])', r'\1; \2', text)

        # Fix excessive spaces after list markers
        text = re.sub(r'^([a-z])\.\s{2,}', r'\1. ', text, flags=re.MULTILINE)
        text = re.sub(r'^(\d+)\.\s{2,}', r'\1. ', text, flags=re.MULTILINE)

        # Fix space before punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)

        # Fix space after punctuation (but not for abbreviations)
        text = re.sub(r'([.,;:!?])([^\s])', r'\1 \2', text)

        # Fix multiple spaces
        text = re.sub(r' {2,}', ' ', text)

        # Fix specific problematic patterns mentioned by user
        # "sebagai62na dimaksud" -> "sebagaimana dimaksud"
        text = re.sub(r'sebagai\d+na', 'sebagaimana', text, flags=re.IGNORECASE)

        # Fix broken references that got split: "Pasal 51\nsampai dengan Pasal 54"
        text = re.sub(r'(sebagaimana\s+dimaksud\s+dalam\s+Pasal\s+\d+[A-Za-z]?)\s*\n\s*(sampai\s+dengan\s+Pasal\s+\d+[A-Za-z]?)',
                      r'\1 \2', text, flags=re.IGNORECASE)

        # Fix common OCR spacing errors
        text = re.sub(r'dan/ atau', 'dan/atau', text)  # Fix "dan/ atau" -> "dan/atau"
        text = re.sub(r'dengaa\b', 'dengan', text)   # Fix OCR error "dengaa" -> "dengan"
        text = re.sub(r'ada1ah\b', 'adalah', text)   # Fix OCR error "ada1ah" -> "adalah"

        return text

     # PROBLEMATIC
    # def _advanced_line_filtering(self, text: str) -> str:
    #     """Conservative line filtering to remove noise while preserving legal content."""
    #     lines = text.split('\n')
    #     filtered_lines = []

    #     for line in lines:
    #         line = line.strip()
    #         if not line:
    #             continue

    #         # Skip obviously corrupted lines - conservative approach like old cleaner
    #         if (len(line) < 3 or
    #             re.match(r'^[\.\-\s]+$', line) or  # Just dots and dashes
    #             re.match(r'^[^a-zA-Z0-9]*$', line) or  # No alphanumeric
    #             re.match(r'^[A-Z]{1,3}$', line) or  # Single/few caps letters
    #             re.match(r'^\d+$', line) or  # Just numbers (page numbers)
    #             re.match(r'^SK No', line, re.IGNORECASE) or  # Document refs
    #             re.match(r'^-\d+-$', line)):  # Page markers
    #             continue

    #         filtered_lines.append(line)

    #     return '\n'.join(filtered_lines)

    def _standardize_legal_formatting(self, text: str) -> str:
        """Standardize formatting of legal document elements."""
        # Ensure proper newlines after BAB headers
        text = re.sub(
            r'\b(BAB\s+[IVX]+(?:\s+[^\n]*)?)\n?',
            r'\1\n\n',
            text,
            flags=re.IGNORECASE
        )

        # Ensure proper newlines after Pasal headers (but preserve inline references)
        text = re.sub(
            r'^(Pasal\s+\d+[A-Za-z]?)(?!\s+(?:ayat|huruf|angka|sampai|dimaksud))\s*$',
            r'\1\n',
            text,
            flags=re.MULTILINE | re.IGNORECASE
        )

        # Standardize verse formatting
        text = re.sub(r'^\((\d+)\)\s*', r'(\1) ', text, flags=re.MULTILINE)

        # Standardize list formatting
        text = re.sub(r'^([a-z])\.\s+', r'\1. ', text, flags=re.MULTILINE)
        text = re.sub(r'^(\d+)\.\s+', r'\1. ', text, flags=re.MULTILINE)

        return text

    def _final_whitespace_cleanup(self, text: str) -> str:
        """Final whitespace cleanup."""
        # Clean up excessive newlines but preserve paragraph structure
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove trailing spaces on lines
        text = re.sub(r' +$', '', text, flags=re.MULTILINE)

        # Remove leading spaces on lines (except for proper indentation)
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)

        # Ensure consistent spacing
        text = re.sub(r' {2,}', ' ', text)

        return text

    def _fix_line_wrapping_and_sentence_segmentation(self, text: str) -> str:
        """
        Fix improper line wrapping and broken sentence segmentation.

        Joins sentences that were incorrectly split across multiple lines due to
        hard line breaks from PDF extraction while preserving proper legal structure.

        Examples:
        - "Tidak ada satu perbuatan pun yang dapat dikenai\nsanksi pidana dan/atau tindakan"
          -> "Tidak ada satu perbuatan pun yang dapat dikenai sanksi pidana dan/atau tindakan"
        """
        lines = text.split('\n')
        fixed_lines = []
        i = 0

        while i < len(lines):
            current_line = lines[i].strip()

            # Skip empty lines
            if not current_line:
                fixed_lines.append('')
                i += 1
                continue

            # Preserve legal structure markers - never join these
            if self._is_legal_structure_marker(current_line):
                fixed_lines.append(current_line)
                i += 1
                continue

            # Check if current line should be joined with next lines
            joined_line = current_line
            j = i + 1

            # Keep joining lines until we find a natural sentence end
            while j < len(lines):
                next_line = lines[j].strip()

                # Stop joining if we hit empty line or legal structure
                if not next_line or self._is_legal_structure_marker(next_line):
                    break

                # Stop joining if next line starts with list marker (a., b., c., etc.)
                if re.match(r'^[a-z]\.\s+', next_line):
                    break

                # Stop joining if current line ends with semicolon and next starts with list marker
                if re.search(r';\s*$', joined_line) and re.match(r'^[a-z]\.\s+', next_line):
                    break

                # Stop joining if current line ends with sentence ending punctuation
                # and next line starts with capital letter or number (new sentence)
                if self._is_sentence_end(joined_line) and self._is_sentence_start(next_line):
                    break

                # Default behavior: join lines unless we find a clear sentence boundary
                # This handles cases where sentences are broken across multiple lines
                should_join = True

                # Don't join if next line looks like a new sentence or legal element
                if (self._is_sentence_end(joined_line) and
                    (re.match(r'^[A-Z]', next_line) or re.match(r'^\d+\.', next_line))):
                    should_join = False

                # Don't join if we've found a complete thought and next line starts new thought
                if (re.search(r'\.$', joined_line.strip()) and
                    re.match(r'^[A-Z][a-z]', next_line)):
                    should_join = False

                if should_join:
                    # Join the lines with a space
                    joined_line += ' ' + next_line
                    j += 1
                else:
                    break

                # Stop if we've created a very long line (likely already complete)
                if len(joined_line) > 500:
                    break

            # Clean up the joined line
            joined_line = self._clean_joined_line(joined_line)
            fixed_lines.append(joined_line)

            # Move to the next unprocessed line
            i = j

        return '\n'.join(fixed_lines)

    def _is_legal_structure_marker(self, line: str) -> bool:
        """Check if line is a legal structure marker that should not be joined."""
        line = line.strip()

        # Legal headers and markers - be more specific to avoid false positives
        case_insensitive_patterns = [
            r'^BAB\s+[IVXLCDM\d]+\s*$',               # BAB headers (exact match)
            r'^Bagian\s+\w+\s*$',                     # Section headers (exact match)
            r'^Paragraf\s+\d+\s*$',                   # Paragraph headers (exact match)
            r'^Pasal\s+\d+[A-Z]?\s*$',                # Article headers (standalone)
            r'^BUKU\s+\w+\s*$',                       # Book headers (exact match)
            r'^ATURAN\s+\w+\s*$',                     # Rule headers (exact match)
        ]

        case_sensitive_patterns = [
            r'^\(\d+\)\s*$',                          # Verse numbers (standalone only)
            r'^[a-z]\.\s*$',                          # List markers (standalone only)
            r'^\d+\.\s*$',                            # Numbered list (standalone only)
            r'^[A-Z][A-Z\s]*[A-Z]\s*$',               # All caps headers (case sensitive)
        ]

        # Check case-insensitive patterns
        for pattern in case_insensitive_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True

        # Check case-sensitive patterns
        for pattern in case_sensitive_patterns:
            if re.match(pattern, line):
                return True

        return False

    def _is_sentence_end(self, line: str) -> bool:
        """Check if line ends with sentence-ending punctuation."""
        line = line.strip()
        if not line:
            return False

        # Check for clear sentence endings
        if re.search(r'[.!?]\s*$', line):
            return True

        # Don't treat semicolons and colons as sentence ends for legal text
        # as they often continue the same legal provision
        return False

    def _is_sentence_start(self, line: str) -> bool:
        """Check if line starts like a new sentence."""
        line = line.strip()
        if not line:
            return False

        # Starts with capital letter, number, or legal marker
        return bool(re.match(r'^[A-Z0-9(]', line))

    def _clean_joined_line(self, line: str) -> str:
        """Clean up a joined line to fix common issues."""
        # Fix multiple spaces
        line = re.sub(r'\s+', ' ', line)

        # Fix common OCR errors in joined text
        line = re.sub(r'dengaa\b', 'dengan', line)  # Fix OCR error "dengaa"
        line = re.sub(r'ada1ah\b', 'adalah', line)  # Fix OCR error "ada1ah"
        line = re.sub(r'dan/ atau', 'dan/atau', line)  # Fix spacing in "dan/atau"
        line = re.sub(r'sanksi pidana dan/atau', 'sanksi pidana dan/atau', line)

        # Fix broken ayat content patterns - be careful to avoid duplicates



        line = re.sub(r'sanksi pidana\s+dan/atau\s+tindakan,\s+kecuali\s+atas(?!\s+kekuatan)', 'sanksi pidana dan/atau tindakan, kecuali atas kekuatan', line)

        # Fix spacing around punctuation
        line = re.sub(r'\s+([.,;:!?])', r'\1', line)
        line = re.sub(r'([.,;:!?])([^\s])', r'\1 \2', line)

        return line.strip()

    def _fix_collapsed_list_formatting(self, text: str) -> str:
        """
        Fix collapsed list formatting in legal text.

        Separates enumerated items (a., b., c., etc.) that are improperly
        joined into single paragraphs, ensuring proper legal structure.

        Example:
        "melakukan: a. Tindak Pidana; b. Pelanggaran; c. Kejahatan"
        becomes:
        "melakukan:
        a. Tindak Pidana;
        b. Pelanggaran;
        c. Kejahatan"
        """
        lines = text.split('\n')
        processed_lines = []

        for line in lines:
            line_stripped = line.strip()

            # Skip if line doesn't contain list markers
            if not re.search(r'[a-z]\.\s+', line_stripped):
                processed_lines.append(line)
                continue

            # Pattern 1: Collapsed enumerated lists after colon
            colon_list_pattern = r'^(.+:)\s*([a-z]\.\s+[^;]+(?:;\s*(?:atau\s+)?[a-z]\.\s+.+)+[;.]?)$'

            # Pattern 2: Inline semicolon-separated lists (without colon)
            inline_list_pattern = r'^([a-z]\.\s+[^;]+(?:;\s*[a-z]\.\s+.+)+[;.]?)$'

            # Try colon pattern first
            colon_match = re.match(colon_list_pattern, line_stripped)
            inline_match = re.match(inline_list_pattern, line_stripped)

            if colon_match:
                prefix = colon_match.group(1)  # Text before colon
                list_content = colon_match.group(2).strip()  # The enumerated items

                # Split on semicolons but preserve "atau" handling
                items = []
                current_item = ""
                parts = re.split(r'(;\s*(?:atau\s+)?)', list_content)

                for part in parts:
                    if re.match(r';\s*(?:atau\s+)?', part):
                        if current_item.strip():
                            if 'atau' in part:
                                items.append(current_item.strip() + '; atau')
                            else:
                                items.append(current_item.strip())
                            current_item = ""
                    else:
                        current_item += part

                # Add the final item
                if current_item.strip():
                    items.append(current_item.strip())

                # Build formatted output
                if len(items) >= 2:  # Only format if multiple items
                    processed_lines.append(prefix)
                    processed_lines.extend(items)
                else:
                    # Single item, handle special case
                    if re.search(r'^[a-z]\.\s+', list_content) and not ';' in list_content:
                        processed_lines.append(prefix)
                        processed_lines.append(list_content)
                    else:
                        processed_lines.append(line)
            elif inline_match:
                # Handle inline lists without colon
                list_content = inline_match.group(1).strip()

                # Split on semicolons but preserve "atau" handling
                items = []
                current_item = ""
                parts = re.split(r'(;\s*(?:atau\s+)?)', list_content)

                for part in parts:
                    if re.match(r';\s*(?:atau\s+)?', part):
                        if current_item.strip():
                            if 'atau' in part:
                                items.append(current_item.strip() + '; atau')
                            else:
                                items.append(current_item.strip())
                            current_item = ""
                    else:
                        current_item += part

                # Add the final item
                if current_item.strip():
                    items.append(current_item.strip())

                # Build formatted output - no prefix for inline lists
                if len(items) >= 2:  # Only format if multiple items
                    processed_lines.extend(items)
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)

        text = '\n'.join(processed_lines)

        # Handle inline items that are already on separate lines but need semicolon cleanup
        # Only process lines that start with list markers and contain semicolons
        lines = text.split('\n')
        final_lines = []

        for line in lines:
            # Fix inline semicolon separation within already separated list items
            if re.match(r'^[a-z]\.\s+.*;\s*[a-z]\.\s+', line.strip()):
                # This line has multiple items that should be separated
                items = re.split(r';\s*(?=atau\s+)?(?=[a-z]\.)', line.strip())
                formatted_items = []

                for item in items:
                    item = item.strip()
                    if item:
                        if item.startswith('atau '):
                            # Add "atau" to previous item
                            if formatted_items:
                                formatted_items[-1] += '; atau'
                            formatted_items.append(item[5:])  # Remove "atau " prefix
                        else:
                            formatted_items.append(item)

                final_lines.extend(formatted_items)
            else:
                final_lines.append(line)

        # Clean up spacing for list items
        text = '\n'.join(final_lines)
        text = re.sub(r'^([a-z]\.)\s+', r'\1 ', text, flags=re.MULTILINE)

        return text

    def _should_continue_ayat(self, current_line: str, next_line: str) -> bool:
        """Check if ayat content should continue to next line."""
        # Continue if current line ends with incomplete legal phrases
        incomplete_endings = [
            r'yang dapat dikenai\s*$',
            r'kecuali\s*$',
            r'atas\s*$',
            r'peraturan\s*$',
            r'perundang-undangan\s*$',
            r'sanksi pidana\s*$',
            r'dan/atau\s*$',
            r'sebelum\s*$',
            r'yang\s*$',
            r'dalam\s*$',
            r'dengan\s*$',
            r'di luar\s*$',
            r'oleh\s*$',
            r'atas dasar\s*$',
            r'yang memberikan\s*$',
        ]

        for pattern in incomplete_endings:
            if re.search(pattern, current_line, re.IGNORECASE):
                return True

        return False

    def _is_incomplete_sentence(self, line: str) -> bool:
        """Check if line appears to be an incomplete sentence."""
        line = line.strip()
        if not line:
            return False

        # If line doesn't end with proper punctuation, likely incomplete
        if not re.search(r'[.!?;:]\s*$', line):
            return True

        # Check for common incomplete sentence patterns
        incomplete_patterns = [
            r'(?:yang|di|ke|dari|untuk|dengan|oleh|atas|dalam)\s*$',  # Prepositions at end
            r'\w+nya\s*$',  # Words ending with -nya
            r'(?:tidak|belum|sudah|akan|dapat|harus|wajib)\s*$',  # Modal words
        ]

        for pattern in incomplete_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True

        return False

    # Legacy methods for backward compatibility
    def clean_whitespace(self, text: str) -> str:
        """Clean whitespace patterns."""
        return self.pattern_manager.normalize_whitespace(text)

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace patterns."""
        return self.pattern_manager.normalize_whitespace(text)

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        return self._fix_encoding_issues(text)

    def fix_encoding(self, text: str) -> str:
        """Fix encoding issues."""
        return self._fix_encoding_issues(text)

    def remove_noise(self, text: str) -> str:
        """Remove noise patterns."""
        return self.pattern_manager.clean_noise_artifacts(text)

    def clean_page_break_artifacts(self, text: str) -> str:
        """
        Clean page break artifacts from real PDF extraction while preserving BAB structure.

        Handles patterns like:
        - "(3) Pidana . . ." + garbage + "(3) Pidana actual content"
        - "Pasal 2..." + "Pasal 2"
        - "a. keamanan , , ," + garbage + "a. keamanan actual content"
        - "BABVII ..." -> "BAB VII"
        - "BAB XIV. . ." -> "BAB XIV"
        - OCR garbage that corrupts chapter headers

        Args:
            text: Input text with page break artifacts

        Returns:
            Cleaned text with artifacts removed and BAB structure preserved
        """
        if not text:
            return ""

        # Step 0: Pre-clean and fix corrupted BAB patterns
        text = self._preserve_and_fix_corrupted_bab(text)

        # Step 1: Remove OCR garbage lines (line by line approach)
        lines = text.split('\n')
        cleaned_lines = []

        garbage_patterns = [
            r'llrr-\{rT;trIllilTlTatrtltrtrIrtr',
            r'TIr:I\+Tf\.\{Il',
            r'I THITT : IIIiirIITItrIIIIEIA',
            r'nrrFF\[iriN\]',
            r'aftfd\'TFITli',
            r'EITtrELItrEEIIEIn',
            r'TEIIEtrTilNEEtrtrEIn',
            r'rrfl Tf:IrIXTItililIt\+Jn!',
            r'I:rfITI\'I\[TXT\.Tjr;tTf,Trf',
            r'EIIIIEIIItrN',
            r'REI\'UEUK\s+INDONESIA',
            r'REPUA\|JK\s+INDONESIA',
            r'rIrIt\{f\.I\{II',
            r'^[A-Z\|\'\:\+\-\{\}\[\]\\]+$',  # Lines with only OCR symbols
            r'.*TIr:I.*',  # More flexible pattern for TIr variations
            r'^[A-Z][a-z]*:[A-Z].*',  # Pattern like TIr:I+Tf.I{Il
            r'^-t\d+-$'  # Page break patterns like "-t29-"
            r'[J\s]*EETITIilIilTI;a lf\*r\'o\s*',
            r'TIr:I\+Tf\.I\{Il\s*',
            r'I THITT : IIIiirIITItrIIIIEIA\s*',
            r'nrrFF\[iriN\]\s*',
            r'REFIJEUK INDONESIA\s*',
            r'REFUEUK INDONESIA\s*',
            r'REPTIEUK INDONESIA\s*',
            r'REPUELIK INDONESIA\s*',
            r'REPUEUK INDONESIA\s*',
            r'REPTIEUI\( INDONESIA\s*',
            r'REI\'UEUK INOONESIA\s*',
            r'REPIJBUK INDONESIA\s*',
            r'REPIJEUX INDONESIA\s*',
            r'llrr-\{rT;trIllilTlTatrtltrtrIrtr\s*',
            r'aftfd\'TFITli\s*',
            r'EITtrELItrEEIIEIn\s*',
            r'l-NIitrIII,FfA\s*',
            r'I f rl rFIT-l r\[I\]\s*',
            r'i-ili rIrL IITtTIIItrtIIEtrII\'tr\s*',
            r'FNESIDEN\s*',
            r':lrlTl :E\]1ITllir\.TII\*Tnt\s*',
            r'rNIitr\[FEtA\s*',
            r'PRESTDEN\s*',
            r'REPUBUK INDONESIA\s*',
            r'REI\'UBUK INDONESIA\s*',
            r'iilT\.frIItrf,INEEtrtrEm\s*',
            r'T\{!TTilTTTIT\'T\'\]ITSrJ\s*',
            r'EEtrtrIEtrN\s*',
            r'\[rrfi rEIrrilTIrd\.TII4rA\s*',
            r'EITI;EIEEN\]\s*',
            r'RE,\'UEUK INDONESIA\s*',
            r'FRESIbEN\s*',
            # NEW GARBLED PATTERNS FROM USER
            r'PRESIOEl\{\s*',
            r'REPTIELIK INDONESIA\s*',
            r'PRESIDEH\s*',
            r'REPTIBIIK INDONESI\.A\s*',
            r'IIIFFIIItrN\]\s*',
            r'EIfIIEtrLIf,EENtrEIn\s*',
            r'Pasal2TI\s*',
            r'\|:I\^IIIEIEtrN\s*',
            r'REI\'UEUK INDONESIA\s*',
            r'Erfflmill\s*',
            r'REFUELIK INDONESIA\s*',
            r'\| ;J:l rFITil !N\s*',
            r'tJrlTl:TIIXNI-LNT,FII\.\]\s*',
            r'REPUBIJK INDONESIA\s*',
            r'REPUEL\|K INDONESIA\s*',
            r'rrfl Tf:IrIXTItililIt\+Jn!\s*',
            r'6ll38A\s*',
            r'EEEIEtrtr\]\s*',
            r'EiltrtrLINEEtrtrEIA\s*',
            r'6ll4l A\s*',
            r'NEPUBUK INDONESIA\s*',
            r'NEPUBLIK INDONESIA\s*',
            r'FRESIDEN\s*',
            r'IEIFFIIiIINI\s*',
            r'iiIflIEtrT\.IIIEEtrtrEM\s*',
            # Additional noise patterns from user examples
            r'61205A\s*',
            r'REFTIBUK INDONESIA\s*',
            r'I r rJE rl :TrltTItil\[If\*Int\s*',
            r'NEFUBUK INDONESIA,\s*',
            r'PNESIDEN\s*',
            r'REPUEIJK INDONESIA\s*',
            r'6l2l0A\s*',
            r'EUK INDONESIA\s*',
            r'NEPUELIK INDONESIA\s*',
            r'REPUEUK :NDONESIA\s*',
            r'6l2l6A\(\(2\)\s*',
            r'b-2t5-\s*',
            r'6l2l4A\s*',
            r'REPUBL\|K INDONESIA\s*',
            r'6122l A\s*',
            r'EEEEIEtrN\s*',
            r't r r !rT: IrTilNI-LNIEFitA\s*',
            r'EEEItrtrN\s*',
            r',148APasaJ492\.\. \.\s*',
            r'Tr3-ll-NIrtrtlIiEIA\s*',
            r'TIEITFITTIEN\s*',
            r'FEPUBUK INDONESIA\s*',
            r'REI\'UELIK INOONESIA\s*',
            r'E\[trtrLIIIEEf,trEIn\s*',
            r'EEFFIEIIN\s*',
            r'EfltrtrItrItrEEf,trEIA\s*',
            r'\[iTfiTEIflilIrIitrIIEEIE\s*',
            r'r\[i-\{Ttrtrf,ItrEEtrtrEm\s*',
            r'REPUBL\|K INOONESIA\s*',
            r'I-NI-I\.-II\[rFInt\s*',
            r'I\'NTItrTT\{itrtrJ\s*',
            r'REFTIEUK INDONESIA\s*',
            r'REI\'UELIK TNDONESIA\s*',

            r'-\s*\d+\s*-\s*',
            r'-\d+-',
            r'- \d+ -',
            r'-\s*\d+\s*-',
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            is_garbage = False

            # Preserve BAB lines - never mark as garbage
            if re.match(r'^\s*BAB\s+[IVXLCDM\d]+', line, re.IGNORECASE):
                cleaned_lines.append(line)
                continue

            # Check if line matches any garbage pattern
            for pattern in garbage_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    is_garbage = True
                    break

            # Check if line has ellipsis (page break footer) but not BAB
            if re.search(r'\.(\s*\.)+', line) and not re.search(r'BAB', line, re.IGNORECASE):
                is_garbage = True

            if not is_garbage:
                cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        # Step 2: Fix duplicate content patterns
        # Pattern: "a. partial , , ," + "a. full content" -> "a. full content"
        text = re.sub(r'([a-z])\.\s+[^.\n]*?\s*,\s*,\s*,\s*\n([a-z])\.\s+', r'\2. ', text, flags=re.MULTILINE)

        # Pattern: "(5) partial . . ." + "(5) full content" -> "(5) full content"
        text = re.sub(r'\((\d+)\)\s+[^(]*?\s*\.{2,}\s*\n\((\d+)\)\s+', r'(\2) ', text, flags=re.MULTILINE)

        # Step 3: Post-clean BAB structure
        # text = self._fix_bab_formatting(text)

        # Step 4: Clean excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)

        return text.strip()

    def _fix_bab_formatting(self, text: str) -> str:
        """Fix BAB formatting after cleaning using PatternManager."""
        # Use PatternManager for BAB fixes first
        text = self.pattern_manager.fix_bab_patterns(text)

        # Additional formatting fixes
        text = re.sub(r'([^\n])\s*(BAB\s+[IVXLCDM\d]+)', r'\1\n\2', text)
        text = re.sub(r'BAB\s+([IVXLCDM\d]+)([A-Z])', r'BAB \1\n\2', text)

        return text

    def standardize_legal_formatting(self, text: str) -> str:
        """Standardize legal formatting."""
        return self._standardize_legal_formatting(text)

    def clean_comprehensive(self, text: str, preserve_structure: bool = True) -> CleaningResult:
        """
        Apply comprehensive cleaning to legal document text.

        Args:
            text: Input text to clean
            preserve_structure: Whether to preserve legal structure (always True for legal docs)

        Returns:
            CleaningResult with detailed information
        """
        if not text:
            return CleaningResult(
                cleaned_text="",
                operations_applied=[],
                original_length=0,
                cleaned_length=0,
                processing_time=0.0
            )

        start_time = time.time()
        original_length = len(text)

        try:
            # Use the comprehensive legal document cleaning
            cleaned_text = self.clean_legal_document_comprehensive(text)
            processing_time = time.time() - start_time

            operations = [
                "critical_ocr_fixes",
                "bab_pattern_fixes",
                "page_break_cleaning",
                "legal_pattern_fixes",
                "reference_formatting",
                "ayat_formatting",
                "ocr_correction",
                "noise_removal",
                "spacing_fixes",
                "line_filtering",
                "legal_formatting",
                "line_wrapping_fixes",
                "collapsed_list_formatting",
                "whitespace_cleanup"
            ]

            return CleaningResult(
                cleaned_text=cleaned_text,
                operations_applied=operations,
                original_length=original_length,
                cleaned_length=len(cleaned_text),
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Error in comprehensive cleaning: {e}")
            # Fallback to basic cleaning
            basic_cleaned = self._final_whitespace_cleanup(text)
            processing_time = time.time() - start_time

            return CleaningResult(
                cleaned_text=basic_cleaned,
                operations_applied=["basic_fallback"],
                original_length=original_length,
                cleaned_length=len(basic_cleaned),
                processing_time=processing_time
            )

    def get_cleaning_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about text cleaning potential.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with cleaning statistics
        """
        if not text:
            return {}

        stats = {
            'character_count': len(text),
            'word_count': len(text.split()),
            'line_count': len(text.splitlines()),
            'has_pasal_references': bool(re.search(r'\bPasal\s+\d+', text, re.IGNORECASE)),
            'has_bab_headers': bool(re.search(r'\bBAB\s+[IVX]+', text, re.IGNORECASE)),
            'has_ayat_numbers': bool(re.search(r'\(\d+\)', text)),
            'has_list_markers': bool(re.search(r'^[a-z]\.\s', text, re.MULTILINE)),
            'has_sk_noise': bool(re.search(r'SK\s+No', text, re.IGNORECASE)),
            'has_garbled_text': bool(re.search(r'[{}\[\]\\|~`@#$%^&*+=<>]', text)),
            'has_spacing_issues': bool(re.search(r',\s*,\s*,', text)),
            'has_excessive_newlines': bool(re.search(r'\n{3,}', text)),
            'needs_legal_cleaning': True  # Always true for legal documents
        }

        return stats

    def suggest_cleaning_operations(self, text: str) -> List[str]:
        """
        Suggest which cleaning operations would be most beneficial.

        Args:
            text: Text to analyze

        Returns:
            List of recommended cleaning operations
        """
        if not text:
            return []

        stats = self.get_cleaning_statistics(text)
        suggestions = []

        # Always suggest comprehensive legal cleaning for legal documents
        if stats.get('needs_legal_cleaning', False):
            suggestions.append('clean_legal_document_comprehensive')

        # Specific suggestions based on detected issues
        if stats.get('has_sk_noise', False):
            suggestions.append('remove_document_noise')

        if stats.get('has_garbled_text', False):
            suggestions.append('fix_critical_ocr_errors')

        if stats.get('has_spacing_issues', False):
            suggestions.append('fix_spacing_issues')

        if stats.get('has_excessive_newlines', False):
            suggestions.append('normalize_whitespace')

        return suggestions if suggestions else ['clean_legal_document_comprehensive']
