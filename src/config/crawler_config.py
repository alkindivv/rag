#!/usr/bin/env python3
"""
CrawlerConfig - Centralized configuration for all crawler components
All hardcoded values, mappings, and configurations centralized here
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List


class CrawlerConfig:
    """Centralized configuration for all crawler operations - Single source of truth"""

    # ========================================
    # BPK WEBSITE CONFIGURATION
    # ========================================
    BASE_URL = "https://peraturan.bpk.go.id"
    SEARCH_URL = f"{BASE_URL}/Search"

    # ========================================
    # REGULATION TYPE MAPPING (Single source of truth)
    # ========================================
    REGULATION_TYPE_MAP = {
        'uud': '7',
        'ketetapan_mpr': '39',
        'uu': '8',
        'uudarurat': '36',
        'perpu': '9',
        'pp': '10',
        'perpres': '11',
        'keppres': '12',
        'inpres': '13',
        'peraturan_mendagri': '40',
        'peraturan_menkeu': '42',
        'peraturan_menpolhukam': '43',
        'peraturan_menlu': '45',
        'peraturan_menhukham': '46',
        'peraturan_menhan': '47',
        'peraturan_menhub': '48',
        'pergub': '20',
        'perbup': '23',
        'peraturan_walikota': '30'
    }

    # ========================================
    # METADATA FIELDS CONFIGURATION (Single source of truth)
    # ========================================
    # Raw field names for extraction from web pages
    METADATA_FIELDS = {
        'tipe_dokumen': ['Tipe Dokumen', 'Document Type'],
        'title': ['Judul', 'Title'],
        'teu': ['T.E.U.', 'TEU'],
        'number': ['Nomor', 'Number'],
        'year': ['Tahun', 'Year'],
        'form': ['Bentuk', 'Form'],
        'form_short': ['Bentuk Singkat', 'Short Form'],
        'place_enacted': ['Tempat Penetapan', 'Place of Enactment'],
        'date_enacted': ['Tanggal Penetapan', 'Date of Enactment'],
        'date_promulgated': ['Tanggal Pengundangan', 'Date of Promulgation'],
        'date_effective': ['Tanggal Berlaku', 'Effective Date'],
        'source': ['Sumber', 'Source'],
        'subject': ['Subjek', 'Subject'],
        'status': ['Status', 'Status'],
        'language': ['Bahasa', 'Language'],
        'location': ['Lokasi', 'Location'],
        'field': ['Bidang', 'Field'],
    }

    # ========================================
    # HTTP REQUEST CONFIGURATION
    # ========================================
    HTTP_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'DNT': '1'
    }

    PDF_HEADERS = {
        'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }

    # ========================================
    # CONNECTION AND RETRY CONFIGURATION
    # ========================================
    CONNECTION_LIMITS = {
        'limit': 10,
        'limit_per_host': 2,
        'ttl_dns_cache': 300,
        'use_dns_cache': True,
    }

    TIMEOUT_CONFIG = {
        'total': 60,
        'connect': 30,
        'sock_read': 60
    }

    DOWNLOAD_TIMEOUT_CONFIG = {
        'total': 300,
        'connect': 30,
        'sock_read': 60
    }

    RETRY_CONFIG = {
        'max_retries': 3,
        'retry_delay': 2.0,
        'download_retries': 5,
        'download_delay': 3.0
    }

    # ========================================
    # FILE PROCESSING CONFIGURATION
    # ========================================
    FILENAME_CONFIG = {
        'max_length': 255,
        'invalid_chars_pattern': r'[<>:"/\\|?*]',
        'whitespace_pattern': r'\s+',
        'underscore_pattern': r'_+'
    }

    DOWNLOAD_CONFIG = {
        'chunk_size': 65536,  # 64KB chunks
        'sleep_between_requests': 1.0,
        'sleep_between_pages': 1.0
    }

    # ========================================
    # REGULATION TYPE MAPPING FOR FILENAMES
    # ========================================
    FILENAME_TYPE_MAPPING = {
        'uud': 'undang_undang_dasar',
        'ketetapan_mpr': 'ketetapan_mpr',
        'uu': 'undang_undang',
        'uudarurat': 'undang_undang_darurat',
        'perpu': 'peraturan_pemerintah_pengganti_undang_undang',
        'pp': 'peraturan_pemerintah',
        'perpres': 'peraturan_presiden',
        'keppres': 'keputusan_presiden',
        'inpres': 'instruksi_presiden',
        'peraturan_mendagri': 'peraturan_menteri_dalam_negeri',
        'peraturan_menkeu': 'peraturan_menteri_keuangan',
        'pergub': 'peraturan_gubernur',
        'perbup': 'peraturan_bupati',
        'peraturan_walikota': 'peraturan_walikota'
    }

    # ========================================
    # FORM SHORT MAPPINGS
    # ========================================
    FORM_SHORT_MAPPINGS = {
        'undang-undang': 'UU',
        'peraturan pemerintah': 'PP',
        'peraturan presiden': 'Perpres',
        'peraturan menteri': 'Permen',
        'peraturan daerah': 'Perda',
        'peraturan walikota': 'Perwali',
        'peraturan bupati': 'Perbup',
        'peraturan gubernur': 'Pergub',
    }

    # ========================================
    # RELATIONSHIP PATTERNS CONFIGURATION
    # ========================================
    RELATIONSHIP_PATTERNS = [
        # Specific patterns first (longer/more specific)
        ('mencabut sebagian', 'revokes_partially'),
        ('dicabut sebagian', 'revoked_partially_by'),
        ('mengubah sebagian', 'amends_partially'),
        ('diubah sebagian', 'amended_partially_by'),
        ('dicabut dengan', 'revoked_by'),
        ('diubah dengan', 'amended_by'),
        ('mengatur lebih lanjut', 'further_regulates'),
        ('diatur lebih lanjut', 'further_regulated_by'),
        # General patterns last (shorter/less specific)
        ('mencabut', 'revokes'),
        ('mengubah', 'amends'),
        ('merubah', 'amends'),
        ('diubah', 'amended_by'),
        ('dicabut', 'revoked_by'),
        ('melaksanakan', 'implements'),
        ('dilaksanakan', 'implemented_by'),
        ('merujuk', 'refers_to'),
        ('dirujuk', 'referred_by'),
        ('membentuk', 'established_by'),
        ('menetapkan', 'establishes'),
        ('ditetapkan', 'established_by')
    ]

    # ========================================
    # TEXT PROCESSING PATTERNS
    # ========================================
    TEXT_PATTERNS = {
        'pasal_pattern': r'(?m)^\s*Pasal\s+\d+\s*(?:ayat\s+\(\d+\))?',
        'numeric_pattern': r'\d+',
        'year_pattern': r'(\d{4})',
        'regulation_keywords': ['eraturan', 'undang', 'eputusan', 'nstruksi', 'ermenko', 'erpres', 'anun'],
        'subject_delimiters': [',', ';', '|', ' - ', ' / ', '\n', '\r'],
        'excluded_subjects': ['tidak ada', '-', 'n/a']
    }

    # ========================================
    # SEARCH SELECTORS AND PATTERNS
    # ========================================
    SEARCH_SELECTORS = {
        'detail_links': r'/Details/\d+/',
        'regulation_pattern': r'(PP|UU|Perpres|Permen|Perda)\s*(?:No\.?\s*|Nomor\s*)\d+',
        'selectors': ['div.search-result', 'div.regulation-item', 'div.item', 'li', 'h3 a', 'h4 a'],
        'excluded_text_patterns': ['ABSTRAK:', 'CATATAN:', 'LAMPIRAN:']
    }

    # ========================================
    # PDF EXTRACTION PATTERNS
    # ========================================
    PDF_PATTERNS = {
        'download_patterns': [
            'DownloadUjiMateri',
            'downloadujimateri'
        ],
        'file_section_patterns': ['FILEFILEPERATURAN', 'filefileperaturan'],
        'uji_materi_patterns': ['UJI MATERI', 'uji materi'],
        'pdf_indicators': ['.pdf', 'download', 'unduh']
    }

    def __init__(self):
        """Initialize configuration - no instance variables, all class-level"""
        pass

    @classmethod
    def get_base_url(cls) -> str:
        """Get BPK base URL"""
        return cls.BASE_URL

    @classmethod
    def get_search_url(cls) -> str:
        """Get BPK search URL"""
        return cls.SEARCH_URL

    @classmethod
    def get_regulation_type_id(cls, regulation_type: str) -> str:
        """Get regulation type ID for BPK search"""
        return cls.REGULATION_TYPE_MAP.get(regulation_type.lower(), '')

    @classmethod
    def get_metadata_fields(cls) -> Dict[str, List[str]]:
        """Get metadata field mappings"""
        return cls.METADATA_FIELDS

    @classmethod
    def get_http_headers(cls) -> Dict[str, str]:
        """Get HTTP headers for requests"""
        return cls.HTTP_HEADERS.copy()

    @classmethod
    def get_pdf_headers(cls, base_url: str) -> Dict[str, str]:
        """Get PDF-specific headers"""
        headers = cls.get_http_headers()
        headers.update(cls.PDF_HEADERS)
        headers['Referer'] = base_url
        return headers

    @classmethod
    def get_connection_config(cls) -> Dict:
        """Get connection configuration"""
        return cls.CONNECTION_LIMITS.copy()

    @classmethod
    def get_timeout_config(cls) -> Dict:
        """Get timeout configuration"""
        return cls.TIMEOUT_CONFIG.copy()

    @classmethod
    def get_download_timeout_config(cls) -> Dict:
        """Get download timeout configuration"""
        return cls.DOWNLOAD_TIMEOUT_CONFIG.copy()

    @classmethod
    def get_retry_config(cls) -> Dict:
        """Get retry configuration"""
        return cls.RETRY_CONFIG.copy()

    @classmethod
    def get_relationship_patterns(cls) -> List[tuple]:
        """Get relationship extraction patterns"""
        return cls.RELATIONSHIP_PATTERNS.copy()

    @classmethod
    def setup_logging(cls, log_level: str = "INFO") -> logging.Logger:
        """Setup logging configuration"""
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)

        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        logger = logging.getLogger()
        logger.setLevel(numeric_level)
        logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    @classmethod
    def create_directories(cls, directories: List[Path]) -> None:
        """Create directories if they don't exist"""
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logging.getLogger(__name__).debug(f"Created directory: {directory}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to create directory {directory}: {str(e)}")
                raise

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename to be safe for filesystem"""
        config = cls.FILENAME_CONFIG
        sanitized = re.sub(config['invalid_chars_pattern'], '_', filename)
        sanitized = re.sub(config['whitespace_pattern'], ' ', sanitized)
        sanitized = sanitized.strip(' .')

        if len(sanitized) > config['max_length']:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:config['max_length'] - len(ext)] + ext

        return sanitized

    @classmethod
    def clean_filename_component(cls, text: str) -> str:
        """Clean text for use in filename"""
        if not text:
            return 'unknown'

        text = str(text).lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '_', text)
        text = re.sub(cls.FILENAME_CONFIG['underscore_pattern'], '_', text)
        text = text.strip('_')

        return text or 'unknown'

    @classmethod
    def get_regulation_type_from_command(cls, command_arg: str) -> str:
        """Map command line argument to specific regulation type for file naming"""
        return cls.FILENAME_TYPE_MAPPING.get(command_arg.lower(), 'peraturan_perundang_undangan')

    @classmethod
    def detect_regulation_type_from_metadata(cls, regulation: Dict) -> str:
        """Detect regulation type from metadata fields for file naming"""
        title = regulation.get('doc_title', regulation.get('title', '')).lower()
        form = regulation.get('doc_form', regulation.get('form', '')).lower()

        # Check title patterns
        title_patterns = {
            'undang_undang': ['undang-undang', 'uu'],
            'peraturan_pemerintah': ['peraturan pemerintah', 'pp'],
            'peraturan_presiden': ['peraturan presiden', 'perpres'],
            'peraturan_menteri': ['peraturan menteri', 'permen'],
            'peraturan_daerah': ['peraturan daerah', 'perda']
        }

        for reg_type, patterns in title_patterns.items():
            if any(pattern in title for pattern in patterns):
                return reg_type

        # Check form patterns
        form_patterns = {
            'undang_undang': 'undang-undang',
            'peraturan_pemerintah': 'peraturan pemerintah',
            'peraturan_presiden': 'peraturan presiden',
            'peraturan_menteri': 'peraturan menteri',
            'peraturan_daerah': 'peraturan daerah'
        }

        for reg_type, pattern in form_patterns.items():
            if pattern in form:
                return reg_type

        return 'peraturan_perundang_undangan'

    @classmethod
    def generate_filename(cls, regulation: Dict, regulation_type: str = "", extension: str = "") -> str:
        """Generate filename based on regulation metadata"""
        if regulation_type:
            reg_type = cls.get_regulation_type_from_command(regulation_type)
        else:
            reg_type = cls.detect_regulation_type_from_metadata(regulation)

        number = regulation.get('doc_number', regulation.get('number', 'unknown'))
        year = regulation.get('doc_year', regulation.get('year', 'unknown'))

        safe_number = cls.clean_filename_component(str(number))
        safe_year = cls.clean_filename_component(str(year))

        return f"{reg_type}_{safe_number}_{safe_year}{extension}"

    @classmethod
    def generate_pdf_filename(cls, regulation: Dict, regulation_type: str = "") -> str:
        """Generate PDF filename based on regulation metadata"""
        return cls.generate_filename(regulation, regulation_type, ".pdf")

    @classmethod
    def generate_json_filename(cls, regulation: Dict, regulation_type: str = "") -> str:
        """Generate JSON filename based on regulation metadata"""
        return cls.generate_filename(regulation, regulation_type, ".json")

    @classmethod
    def generate_text_filename(cls, regulation: Dict, regulation_type: str = "") -> str:
        """Generate text filename based on regulation metadata"""
        return cls.generate_filename(regulation, regulation_type, ".txt")

    @classmethod
    def extract_form_short(cls, form: str) -> str:
        """Extract short form from regulation form"""
        if not form:
            return 'Unknown'

        form_lower = form.lower()
        for key, short in cls.FORM_SHORT_MAPPINGS.items():
            if key in form_lower:
                return short

        # Check for parentheses pattern
        match = re.search(r'\(([^)]+)\)', form)
        if match:
            return match.group(1)

        return form

    @classmethod
    def get_text_patterns(cls) -> Dict:
        """Get text processing patterns"""
        return cls.TEXT_PATTERNS.copy()

    @classmethod
    def get_search_config(cls) -> Dict:
        """Get search configuration"""
        return cls.SEARCH_SELECTORS.copy()

    @classmethod
    def get_pdf_config(cls) -> Dict:
        """Get PDF extraction configuration"""
        return cls.PDF_PATTERNS.copy()

    @classmethod
    def get_download_config(cls) -> Dict:
        """Get download configuration"""
        return cls.DOWNLOAD_CONFIG.copy()
