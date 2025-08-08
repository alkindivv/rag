#!/usr/bin/env python3
"""
Unified PDF Extractor - Simple, Powerful, Optimal
Combines PyMuPDF, PDFPlumber, PyPDF with ranking + fallback
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Unified extraction result"""
    success: bool
    text: str
    method: str
    confidence: float
    processing_time: float
    page_count: int
    error: str = ""
    metadata: Dict[str, Any] = None


class UnifiedPDFExtractor:
    """Unified PDF extractor with ranking and fallback"""

    def __init__(self):
        self.methods = []
        self._init_extractors()

    def _init_extractors(self):
        """Initialize extractors in priority order"""
        # 1. PyMuPDF (fastest, best for legal docs)
        try:
            import fitz
            self.methods.append(('pymupdf', self._extract_pymupdf))
            logger.info("PyMuPDF initialized")
        except ImportError:
            logger.warning("PyMuPDF not available")

        # 2. PDFPlumber (best for tables/layout)
        try:
            import pdfplumber
            self.methods.append(('pdfplumber', self._extract_pdfplumber))
            logger.info("PDFPlumber initialized")
        except ImportError:
            logger.warning("PDFPlumber not available")

        # 3. PyPDF (pure Python fallback)
        try:
            import pypdf
            self.methods.append(('pypdf', self._extract_pypdf))
            logger.info("PyPDF initialized")
        except ImportError:
            logger.warning("PyPDF not available")

        if not self.methods:
            raise RuntimeError("No PDF extractors available")

    def extract_text(self, file_path: str) -> ExtractionResult:
        """Extract text with ranking and fallback"""
        start_time = time.time()

        if not Path(file_path).exists():
            return ExtractionResult(False, "", "none", 0.0, 0.0, 0, f"File not found: {file_path}")

        best_result = None
        best_score = 0.0

        for method_name, extractor_func in self.methods:
            try:
                result = extractor_func(file_path)
                score = self._calculate_score(result)

                logger.info(f"{method_name}: {len(result.text)} chars, score: {score:.2f}")

                if score > best_score:
                    best_score = score
                    best_result = result

                # If score is good enough, use it
                if score > 80.0:
                    break

            except Exception as e:
                logger.warning(f"{method_name} failed: {e}")
                continue

        if best_result:
            best_result.processing_time = time.time() - start_time
            return best_result

        return ExtractionResult(False, "", "all_failed", 0.0, time.time() - start_time, 0, "All extractors failed")

    def _extract_pymupdf(self, file_path: str) -> ExtractionResult:
        """PyMuPDF extraction"""
        import fitz
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        confidence = min(95.0, len(text) / 100)  # Simple confidence based on text length
        return ExtractionResult(True, text, "pymupdf", confidence, 0.0, len(doc), "", {"fast": True})

    def _extract_pdfplumber(self, file_path: str) -> ExtractionResult:
        """PDFPlumber extraction"""
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            confidence = min(90.0, len(text) / 100)
            return ExtractionResult(True, text, "pdfplumber", confidence, 0.0, len(pdf.pages), "", {"layout_aware": True})

    def _extract_pypdf(self, file_path: str) -> ExtractionResult:
        """PyPDF extraction"""
        import pypdf
        text = ""
        with open(file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()

        confidence = min(85.0, len(text) / 100)
        return ExtractionResult(True, text, "pypdf", confidence, 0.0, len(reader.pages), "", {"pure_python": True})

    def _calculate_score(self, result: ExtractionResult) -> float:
        """Calculate quality score for ranking"""
        if not result.success or not result.text:
            return 0.0

        # Base score from confidence
        score = result.confidence

        # Boost for text length (longer = better for legal docs)
        if len(result.text) > 5000:
            score += 10
        elif len(result.text) > 1000:
            score += 5

        # Method-specific bonuses
        if result.method == "pymupdf":
            score += 5  # Speed bonus
        elif result.method == "pdfplumber":
            score += 3  # Layout bonus

        return min(100.0, score)
