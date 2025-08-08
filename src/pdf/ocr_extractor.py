"""
OCR Text Extractor
Specialized PDF text extraction using OCR (Optical Character Recognition)

This module provides OCR-based text extraction for scanned or image-based PDFs,
specifically optimized for Indonesian legal documents with fallback to English.

Author: Refactored Architecture
Purpose: Single responsibility OCR extraction using Tesseract
"""

import io
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
import tempfile
import os

try:
    import pytesseract
    from PIL import Image
    import fitz  # PyMuPDF for PDF to image conversion
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None
    Image = None
    fitz = None


@dataclass
class ExtractionResult:
    """Result of PDF text extraction."""
    success: bool
    text: str
    method: str
    page_count: int
    confidence: float
    processing_time: float
    file_size: int
    metadata: Dict[str, Any]
    layout_preserved: bool = False
    error: str = ""
    tables: List[Any] = None
    precision_mode: bool = False


@dataclass
class OCRConfig:
    """Configuration for OCR extraction."""
    language: str = "ind+eng"  # Indonesian + English
    dpi: int = 300
    psm: int = 6  # Page segmentation mode (6 = uniform block of text)
    oem: int = 3  # OCR engine mode (3 = default, based on available)
    confidence_threshold: float = 60.0  # Minimum confidence for text
    preprocess_image: bool = True
    enhance_contrast: bool = True
    denoise: bool = True
    timeout_seconds: int = 300  # 5 minutes max per PDF


class OCRExtractor:
    """
    OCR-based PDF text extractor using Tesseract.

    Specialized for Indonesian legal documents with intelligent preprocessing
    and confidence scoring. Falls back gracefully when OCR is unavailable.
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        """
        Initialize OCR extractor.

        Args:
            config: OCR configuration settings
        """
        self.config = config or OCRConfig()
        self.logger = logging.getLogger(__name__)

        # Validate Tesseract availability
        if not TESSERACT_AVAILABLE:
            self.logger.warning(
                "OCR dependencies not available. Install: pip install pytesseract pillow pymupdf"
            )

        # Configure Tesseract if available
        if TESSERACT_AVAILABLE and pytesseract:
            self._configure_tesseract()

    def _configure_tesseract(self) -> None:
        """Configure Tesseract OCR settings."""
        try:
            # Try to get Tesseract version to validate installation
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract version: {version}")

            # Set Indonesian language data path if available
            try:
                # Test if Indonesian language is available
                test_config = f'--psm {self.config.psm} --oem {self.config.oem} -l {self.config.language}'
                pytesseract.image_to_string(
                    Image.new('RGB', (100, 50), color='white'),
                    config=test_config
                )
                self.logger.info(f"OCR configured for languages: {self.config.language}")
            except Exception as e:
                self.logger.warning(f"Language {self.config.language} not available, falling back to eng: {e}")
                self.config.language = "eng"

        except Exception as e:
            self.logger.error(f"Failed to configure Tesseract: {e}")
            raise RuntimeError(f"Tesseract configuration failed: {e}")

    def extract_text(self, file_path: str) -> ExtractionResult:
        """
        Extract text from PDF using OCR.

        Args:
            file_path: Path to PDF file

        Returns:
            ExtractionResult with extracted text and metadata
        """
        start_time = time.time()

        # Validate inputs
        if not self._validate_inputs(file_path):
            return ExtractionResult(
                success=False,
                text="",
                method="ocr",
                page_count=0,
                confidence=0.0,
                processing_time=0.0,
                file_size=0,
                metadata={},
                error="Invalid input parameters"
            )

        if not TESSERACT_AVAILABLE:
            return ExtractionResult(
                success=False,
                text="",
                method="ocr",
                page_count=0,
                confidence=0.0,
                processing_time=0.0,
                file_size=0,
                metadata={},
                error="OCR dependencies not available. Install pytesseract, pillow, and pymupdf."
            )

        try:
            # Get file info
            file_size = Path(file_path).stat().st_size

            # Convert PDF to images and extract text
            extracted_text, page_count, confidence, metadata = self._extract_with_ocr(file_path)

            # Return raw OCR text - cleaning will be done by DocumentProcessingPipeline
            cleaned_text = extracted_text.strip() if extracted_text else ""

            processing_time = time.time() - start_time

            return ExtractionResult(
                success=bool(cleaned_text),
                text=cleaned_text,
                method="ocr",
                page_count=page_count,
                confidence=confidence,
                processing_time=processing_time,
                file_size=file_size,
                metadata=metadata,
                layout_preserved=True,  # OCR preserves visual layout
                error="" if cleaned_text else "No text extracted from PDF"
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"OCR extraction failed: {str(e)}"
            self.logger.error(error_msg)

            return ExtractionResult(
                success=False,
                text="",
                method="ocr",
                page_count=0,
                confidence=0.0,
                processing_time=processing_time,
                file_size=Path(file_path).stat().st_size if Path(file_path).exists() else 0,
                metadata={},
                error=error_msg
            )

    def _validate_inputs(self, file_path: str) -> bool:
        """Validate input parameters."""
        try:
            path = Path(file_path)

            if not path.exists():
                self.logger.error(f"File not found: {file_path}")
                return False

            if not path.is_file():
                self.logger.error(f"Not a file: {file_path}")
                return False

            if path.suffix.lower() != '.pdf':
                self.logger.error(f"Not a PDF file: {file_path}")
                return False

            # Check file size (max 50MB for OCR)
            max_size = 50 * 1024 * 1024  # 50MB
            if path.stat().st_size > max_size:
                self.logger.error(f"File too large for OCR: {path.stat().st_size} bytes")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False

    def _extract_with_ocr(self, file_path: str) -> Tuple[str, int, float, Dict[str, Any]]:
        """
        Extract text using OCR on PDF pages converted to images.

        Returns:
            Tuple of (extracted_text, page_count, confidence, metadata)
        """
        try:
            # Open PDF with PyMuPDF
            pdf_doc = fitz.open(file_path)
            page_count = len(pdf_doc)

            if page_count == 0:
                return "", 0, 0.0, {}

            # Extract text from each page
            all_text = []
            all_confidences = []
            processing_metadata = {
                'pages_processed': 0,
                'pages_failed': 0,
                'average_confidence': 0.0,
                'ocr_config': {
                    'language': self.config.language,
                    'dpi': self.config.dpi,
                    'psm': self.config.psm,
                    'oem': self.config.oem
                }
            }

            for page_num in range(page_count):
                try:
                    # Convert page to image
                    page = pdf_doc[page_num]
                    image_data = self._convert_page_to_image(page)

                    if image_data:
                        # Extract text with confidence
                        page_text, page_confidence = self._extract_text_from_image(image_data)

                        if page_text and page_confidence >= self.config.confidence_threshold:
                            all_text.append(page_text)
                            all_confidences.append(page_confidence)
                            processing_metadata['pages_processed'] += 1
                        else:
                            processing_metadata['pages_failed'] += 1
                            self.logger.warning(
                                f"Page {page_num + 1} failed OCR: confidence {page_confidence:.1f}%"
                            )
                    else:
                        processing_metadata['pages_failed'] += 1

                except Exception as e:
                    self.logger.warning(f"Failed to process page {page_num + 1}: {e}")
                    processing_metadata['pages_failed'] += 1
                    continue

            pdf_doc.close()

            # Combine results
            combined_text = '\n\n'.join(all_text)
            average_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            processing_metadata['average_confidence'] = average_confidence

            return combined_text, page_count, average_confidence, processing_metadata

        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return "", 0, 0.0, {'error': str(e)}

    def _convert_page_to_image(self, page) -> Optional[Image.Image]:
        """
        Convert PDF page to PIL Image.

        Args:
            page: PyMuPDF page object

        Returns:
            PIL Image or None if conversion fails
        """
        try:
            # Render page to image with specified DPI
            mat = fitz.Matrix(self.config.dpi / 72, self.config.dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            image = Image.open(io.BytesIO(img_data))

            # Preprocess image if enabled
            if self.config.preprocess_image:
                image = self._preprocess_image(image)

            return image

        except Exception as e:
            self.logger.error(f"Failed to convert page to image: {e}")
            return None

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy.

        Args:
            image: PIL Image to preprocess

        Returns:
            Preprocessed PIL Image
        """
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')

            # Enhance contrast if enabled
            if self.config.enhance_contrast:
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)  # Increase contrast by 50%

            # Apply denoising if enabled
            if self.config.denoise:
                from PIL import ImageFilter
                image = image.filter(ImageFilter.MedianFilter(size=3))

            return image

        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {e}")
            return image  # Return original image if preprocessing fails

    def _extract_text_from_image(self, image: Image.Image) -> Tuple[str, float]:
        """
        Extract text from image using Tesseract.

        Args:
            image: PIL Image to process

        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            # Configure Tesseract
            custom_config = f'--psm {self.config.psm} --oem {self.config.oem} -l {self.config.language}'

            # Extract text
            extracted_text = pytesseract.image_to_string(image, config=custom_config)

            # Get confidence data
            try:
                data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            except Exception:
                avg_confidence = 50.0  # Default confidence if data extraction fails

            return extracted_text.strip(), avg_confidence

        except Exception as e:
            self.logger.error(f"Text extraction from image failed: {e}")
            return "", 0.0

    def supports_file(self, file_path: str) -> bool:
        """
        Check if OCR extractor can process the given file.

        Args:
            file_path: Path to file

        Returns:
            True if file can be processed
        """
        if not TESSERACT_AVAILABLE:
            return False

        try:
            path = Path(file_path)
            return (
                path.exists() and
                path.is_file() and
                path.suffix.lower() == '.pdf' and
                path.stat().st_size <= 50 * 1024 * 1024  # 50MB limit
            )
        except Exception:
            return False

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ['.pdf'] if TESSERACT_AVAILABLE else []

    def get_extraction_info(self) -> Dict[str, Any]:
        """Get information about the extraction capabilities."""
        return {
            'name': 'OCR Extractor',
            'description': 'OCR-based text extraction using Tesseract',
            'supported_formats': self.get_supported_formats(),
            'features': [
                'Scanned PDF support',
                'Indonesian language support',
                'Confidence scoring',
                'Image preprocessing',
                'Layout preservation'
            ],
            'dependencies': ['pytesseract', 'pillow', 'pymupdf'],
            'available': TESSERACT_AVAILABLE,
            'config': {
                'language': self.config.language,
                'dpi': self.config.dpi,
                'confidence_threshold': self.config.confidence_threshold
            } if TESSERACT_AVAILABLE else {}
        }
