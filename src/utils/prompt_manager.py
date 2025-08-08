"""
Prompt Manager
Centralized management of AI prompts and templates for legal document processing

This module provides a unified interface for managing AI prompts, supporting
template variables, localization, and Indonesian legal document optimization.

Author: Refactored Architecture
Purpose: Single responsibility prompt template management
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json


class PromptType(Enum):
    """Types of AI prompts available."""
    SUMMARIZE_BRIEF = "summarize_brief"
    SUMMARIZE_DETAILED = "summarize_detailed"
    SUMMARIZE_EXECUTIVE = "summarize_executive"
    SUMMARIZE_TECHNICAL = "summarize_technical"
    SUMMARIZE_LEGAL = "summarize_legal"

    ANALYZE_STRUCTURE = "analyze_structure"
    EXTRACT_DEFINITIONS = "extract_definitions"
    FIND_SANCTIONS = "find_sanctions"
    ANALYZE_CONCEPTS = "analyze_concepts"

    QUESTION_ANSWER = "question_answer"
    EXPLAIN_CONCEPT = "explain_concept"

    COMPARE_DOCUMENTS = "compare_documents"
    COMPARE_SIMILARITY = "compare_similarity"

    LEGAL_ANALYSIS = "legal_analysis"
    DOCUMENT_INTELLIGENCE = "document_intelligence"

    EXTRACT_LEGAL_CONCEPTS = "extract_legal_concepts"
    MAP_RELATIONSHIPS = "map_relationships"


class Language(Enum):
    """Supported languages."""
    INDONESIAN = "id"
    ENGLISH = "en"


@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata."""
    type: PromptType
    language: Language
    template: str
    variables: List[str]
    description: str
    examples: List[str] = None
    max_length: int = 32000
    legal_optimized: bool = True

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


class PromptManager:
    """
    Centralized prompt manager for AI services.

    Manages prompt templates, supports variable substitution, and provides
    Indonesian legal document optimization.
    """

    def __init__(self, language: Language = Language.INDONESIAN):
        """
        Initialize prompt manager.

        Args:
            language: Default language for prompts
        """
        self.language = language
        self.logger = logging.getLogger(__name__)
        self.templates: Dict[PromptType, PromptTemplate] = {}

        # Initialize with default templates
        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Load default prompt templates."""

        # Summarization prompts
        self.templates[PromptType.SUMMARIZE_BRIEF] = PromptTemplate(
            type=PromptType.SUMMARIZE_BRIEF,
            language=self.language,
            template="""
Anda adalah asisten AI yang ahli dalam menganalisis dokumen hukum Indonesia.
Buatlah ringkasan SINGKAT dari dokumen berikut:

DOKUMEN:
{text}

INSTRUKSI:
1. Ringkasan maksimal 200 kata
2. Fokus pada poin-poin UTAMA saja
3. Gunakan bahasa Indonesia yang jelas
4. Sertakan tujuan utama dan ketentuan penting

RINGKASAN SINGKAT:
""",
            variables=["text"],
            description="Brief summary for quick overview",
            max_length=200
        )

        self.templates[PromptType.SUMMARIZE_DETAILED] = PromptTemplate(
            type=PromptType.SUMMARIZE_DETAILED,
            language=self.language,
            template="""
Anda adalah asisten AI yang ahli dalam menganalisis dokumen hukum Indonesia.
Buatlah ringkasan TERPERINCI dari dokumen berikut:

DOKUMEN:
{text}

INSTRUKSI:
1. Berikan ringkasan yang mencakup:
   - Tujuan utama dokumen
   - Ruang lingkup pengaturan
   - Ketentuan penting dan detail implementasi
   - Sanksi atau konsekuensi (jika ada)
   - Hubungan dengan peraturan lain

2. Gunakan bahasa Indonesia yang jelas dan formal
3. Fokus pada aspek hukum yang paling relevan
4. Maksimal 800 kata
5. Struktur dengan sub-bagian yang jelas

RINGKASAN TERPERINCI:
""",
            variables=["text"],
            description="Detailed summary with comprehensive coverage",
            max_length=800
        )

        self.templates[PromptType.SUMMARIZE_EXECUTIVE] = PromptTemplate(
            type=PromptType.SUMMARIZE_EXECUTIVE,
            language=self.language,
            template="""
Anda adalah asisten AI yang ahli dalam menganalisis dokumen hukum Indonesia.
Buatlah ringkasan EKSEKUTIF untuk pimpinan dari dokumen berikut:

DOKUMEN:
{text}

INSTRUKSI:
1. Format ringkasan eksekutif:
   - Ringkasan Utama (2-3 kalimat)
   - Dampak Bisnis/Organisasi
   - Tindakan yang Diperlukan
   - Risiko Hukum (jika ada)
   - Rekomendasi Implementasi

2. Gunakan bahasa yang mudah dipahami pimpinan
3. Fokus pada implikasi praktis dan strategis
4. Maksimal 400 kata

RINGKASAN EKSEKUTIF:
""",
            variables=["text"],
            description="Executive summary for leadership",
            max_length=400
        )

        # Analysis prompts
        self.templates[PromptType.ANALYZE_STRUCTURE] = PromptTemplate(
            type=PromptType.ANALYZE_STRUCTURE,
            language=self.language,
            template="""
Anda adalah asisten AI yang ahli dalam menganalisis struktur dokumen hukum Indonesia.
Analisis struktur hierarki dari dokumen berikut:

DOKUMEN:
{text}

INSTRUKSI:
1. Identifikasi struktur hierarki:
   - BAB (Chapters)
   - BAGIAN (Sections)
   - PARAGRAF (Paragraphs)
   - PASAL (Articles)
   - AYAT (Verses)
   - Huruf (Letters: a, b, c, ...)
   - Angka (Numbers: 1, 2, 3, ...)

2. Untuk setiap elemen struktur, berikan:
   - Nomor/identifier
   - Judul (jika ada)
   - Ringkasan isi
   - Tingkat hierarki

3. Identifikasi pola struktur dan konsistensi
4. Gunakan format JSON yang jelas

ANALISIS STRUKTUR:
""",
            variables=["text"],
            description="Structural analysis of legal documents"
        )

        self.templates[PromptType.EXTRACT_DEFINITIONS] = PromptTemplate(
            type=PromptType.EXTRACT_DEFINITIONS,
            language=self.language,
            template="""
Anda adalah asisten AI yang ahli dalam menganalisis dokumen hukum Indonesia.
Ekstrak semua definisi dan pengertian dari dokumen berikut:

DOKUMEN:
{text}

INSTRUKSI:
1. Cari semua definisi dengan pola:
   - "yang dimaksud dengan..."
   - "dalam [dokumen] ini yang dimaksud..."
   - "pengertian ... adalah..."
   - "... adalah..."

2. Untuk setiap definisi, berikan:
   - Istilah yang didefinisikan
   - Definisi lengkap
   - Konteks penggunaan
   - Kategori (subjek, objek, prosedur, dll)

3. Format dalam struktur JSON
4. Urutkan berdasarkan urutan kemunculan

EKSTRAKSI DEFINISI:
""",
            variables=["text"],
            description="Extract legal definitions and terms"
        )

        self.templates[PromptType.FIND_SANCTIONS] = PromptTemplate(
            type=PromptType.FIND_SANCTIONS,
            language=self.language,
            template="""
Anda adalah asisten AI yang ahli dalam menganalisis dokumen hukum Indonesia.
Identifikasi semua sanksi dan konsekuensi hukum dari dokumen berikut:

DOKUMEN:
{text}

INSTRUKSI:
1. Cari semua bentuk sanksi:
   - Pidana (penjara, kurungan, denda)
   - Perdata (ganti rugi, pembatalan)
   - Administratif (pencabutan izin, peringatan)
   - Disiplin (pemecatan, penurunan pangkat)

2. Untuk setiap sanksi, berikan:
   - Jenis sanksi
   - Besaran/durasi (jika disebutkan)
   - Kondisi/syarat penerapan
   - Dasar hukum/pasal
   - Kategori pelanggaran

3. Analisis tingkat keseriusan sanksi
4. Format dalam struktur yang jelas

IDENTIFIKASI SANKSI:
""",
            variables=["text"],
            description="Identify sanctions and legal consequences"
        )

        # Q&A prompts
        self.templates[PromptType.QUESTION_ANSWER] = PromptTemplate(
            type=PromptType.QUESTION_ANSWER,
            language=self.language,
            template="""
Anda adalah asisten AI yang ahli dalam hukum Indonesia.
Jawab pertanyaan berikut berdasarkan konteks dokumen yang diberikan:

KONTEKS DOKUMEN:
{context}

PERTANYAAN:
{question}

INSTRUKSI:
1. Berikan jawaban yang akurat berdasarkan konteks
2. Jika informasi tidak cukup, nyatakan dengan jelas
3. Sertakan rujukan pasal/bagian yang relevan
4. Gunakan bahasa Indonesia yang jelas
5. Berikan contoh praktis jika memungkinkan
6. Maksimal 500 kata

JAWABAN:
""",
            variables=["context", "question"],
            description="Answer questions based on legal context"
        )

        # Comparison prompts
        self.templates[PromptType.COMPARE_DOCUMENTS] = PromptTemplate(
            type=PromptType.COMPARE_DOCUMENTS,
            language=self.language,
            template="""
Anda adalah asisten AI yang ahli dalam menganalisis dokumen hukum Indonesia.
Bandingkan dua dokumen berikut dan identifikasi persamaan serta perbedaannya:

DOKUMEN 1:
{document1}

DOKUMEN 2:
{document2}

INSTRUKSI:
1. Analisis perbandingan meliputi:
   - Tujuan dan ruang lingkup
   - Struktur dan organisasi
   - Ketentuan utama
   - Sanksi dan konsekuensi
   - Implementasi dan prosedur

2. Identifikasi:
   - PERSAMAAN: Aspek yang sama atau serupa
   - PERBEDAAN: Aspek yang berbeda signifikan
   - KONFLIK: Ketentuan yang bertentangan
   - KOMPLEMENTER: Aspek yang saling melengkapi

3. Berikan skor similaritas (0-100%)
4. Rekomendasi harmonisasi jika diperlukan

ANALISIS PERBANDINGAN:
""",
            variables=["document1", "document2"],
            description="Compare two legal documents"
        )

        # Legal analysis prompts
        self.templates[PromptType.LEGAL_ANALYSIS] = PromptTemplate(
            type=PromptType.LEGAL_ANALYSIS,
            language=self.language,
            template="""
Anda adalah asisten AI yang ahli dalam hukum Indonesia.
Lakukan analisis hukum komprehensif terhadap pertanyaan berikut:

PERTANYAAN HUKUM:
{query}

KONTEKS DOKUMEN:
{context}

INSTRUKSI:
1. Berikan analisis yang mencakup:
   - Ringkasan isu hukum
   - Ketentuan yang relevan
   - Interpretasi dan penjelasan
   - Implikasi praktis
   - Risiko hukum potensial
   - Rekomendasi tindakan

2. Sertakan rujukan pasal dan ketentuan spesifik
3. Gunakan pendekatan analitis yang sistematis
4. Pertimbangkan aspek implementasi praktis
5. Maksimal 1000 kata

ANALISIS HUKUM:
""",
            variables=["query", "context"],
            description="Comprehensive legal analysis"
        )

        # Concept extraction prompts
        self.templates[PromptType.EXTRACT_LEGAL_CONCEPTS] = PromptTemplate(
            type=PromptType.EXTRACT_LEGAL_CONCEPTS,
            language=self.language,
            template="""
Anda adalah asisten AI yang ahli dalam menganalisis konsep hukum Indonesia.
Ekstrak dan analisis konsep-konsep hukum dari dokumen berikut:

DOKUMEN:
{text}

INSTRUKSI:
1. Identifikasi konsep hukum utama:
   - Subjek hukum (orang, badan hukum)
   - Objek hukum (benda, hak)
   - Hubungan hukum
   - Kewajiban dan hak
   - Prosedur dan mekanisme

2. Untuk setiap konsep, berikan:
   - Nama konsep
   - Kategori konsep
   - Definisi/penjelasan
   - Tingkat kepentingan (1-10)
   - Hubungan dengan konsep lain
   - Contoh konteks penggunaan

3. Buat peta hubungan antar konsep
4. Format dalam struktur JSON

EKSTRAKSI KONSEP HUKUM:
""",
            variables=["text"],
            description="Extract and analyze legal concepts"
        )

    def get_prompt(self, prompt_type: PromptType, **variables) -> str:
        """
        Get formatted prompt with variable substitution.

        Args:
            prompt_type: Type of prompt to retrieve
            **variables: Variables to substitute in template

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If prompt type not found or required variables missing
        """
        if prompt_type not in self.templates:
            raise ValueError(f"Prompt type {prompt_type} not found")

        template = self.templates[prompt_type]

        # Check required variables
        missing_vars = set(template.variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        # Substitute variables
        try:
            formatted_prompt = template.template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Template formatting error: {e}")

        # Validate length
        if len(formatted_prompt) > template.max_length:
            self.logger.warning(
                f"Prompt length ({len(formatted_prompt)}) exceeds maximum ({template.max_length})"
            )

        return formatted_prompt

    def add_template(self, template: PromptTemplate) -> None:
        """
        Add custom prompt template.

        Args:
            template: PromptTemplate to add
        """
        self.templates[template.type] = template
        self.logger.info(f"Added template: {template.type}")

    def update_template(self, prompt_type: PromptType, template_string: str,
                       variables: Optional[List[str]] = None) -> None:
        """
        Update existing template.

        Args:
            prompt_type: Type of prompt to update
            template_string: New template string
            variables: List of required variables (auto-detected if None)
        """
        if prompt_type not in self.templates:
            raise ValueError(f"Prompt type {prompt_type} not found")

        # Auto-detect variables if not provided
        if variables is None:
            variables = self._extract_variables(template_string)

        original_template = self.templates[prompt_type]
        self.templates[prompt_type] = PromptTemplate(
            type=prompt_type,
            language=original_template.language,
            template=template_string,
            variables=variables,
            description=original_template.description,
            max_length=original_template.max_length
        )

        self.logger.info(f"Updated template: {prompt_type}")

    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template string."""
        pattern = r'\{(\w+)\}'
        variables = re.findall(pattern, template)
        return list(set(variables))

    def get_available_prompts(self) -> List[PromptType]:
        """Get list of available prompt types."""
        return list(self.templates.keys())

    def validate_template(self, prompt_type: PromptType, **test_variables) -> bool:
        """
        Validate template with test variables.

        Args:
            prompt_type: Type of prompt to validate
            **test_variables: Test variables for validation

        Returns:
            True if template is valid
        """
        try:
            self.get_prompt(prompt_type, **test_variables)
            return True
        except Exception as e:
            self.logger.error(f"Template validation failed: {e}")
            return False

    def get_template_info(self, prompt_type: PromptType) -> Dict[str, Any]:
        """
        Get information about a template.

        Args:
            prompt_type: Type of prompt

        Returns:
            Dictionary with template information
        """
        if prompt_type not in self.templates:
            raise ValueError(f"Prompt type {prompt_type} not found")

        template = self.templates[prompt_type]
        return {
            "type": template.type.value,
            "language": template.language.value,
            "description": template.description,
            "variables": template.variables,
            "max_length": template.max_length,
            "legal_optimized": template.legal_optimized,
            "examples_count": len(template.examples)
        }

    def optimize_for_legal_domain(self, enable: bool = True) -> None:
        """
        Enable/disable legal domain optimization for all templates.

        Args:
            enable: Whether to enable legal optimization
        """
        for template in self.templates.values():
            template.legal_optimized = enable

        self.logger.info(f"Legal domain optimization: {'enabled' if enable else 'disabled'}")

    def export_templates(self, file_path: str) -> None:
        """
        Export templates to JSON file.

        Args:
            file_path: Path to save templates
        """
        templates_data = {}
        for prompt_type, template in self.templates.items():
            templates_data[prompt_type.value] = {
                "language": template.language.value,
                "template": template.template,
                "variables": template.variables,
                "description": template.description,
                "max_length": template.max_length,
                "legal_optimized": template.legal_optimized
            }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(templates_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Templates exported to: {file_path}")

    def import_templates(self, file_path: str) -> None:
        """
        Import templates from JSON file.

        Args:
            file_path: Path to template file
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Template file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            templates_data = json.load(f)

        for prompt_type_str, template_data in templates_data.items():
            try:
                prompt_type = PromptType(prompt_type_str)
                language = Language(template_data["language"])

                template = PromptTemplate(
                    type=prompt_type,
                    language=language,
                    template=template_data["template"],
                    variables=template_data["variables"],
                    description=template_data["description"],
                    max_length=template_data.get("max_length", 32000),
                    legal_optimized=template_data.get("legal_optimized", True)
                )

                self.templates[prompt_type] = template

            except (ValueError, KeyError) as e:
                self.logger.warning(f"Failed to import template {prompt_type_str}: {e}")

        self.logger.info(f"Templates imported from: {file_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get prompt manager statistics."""
        total_templates = len(self.templates)
        legal_optimized = sum(1 for t in self.templates.values() if t.legal_optimized)

        language_counts = {}
        for template in self.templates.values():
            lang = template.language.value
            language_counts[lang] = language_counts.get(lang, 0) + 1

        return {
            "total_templates": total_templates,
            "legal_optimized_templates": legal_optimized,
            "language_distribution": language_counts,
            "default_language": self.language.value,
            "average_template_length": sum(len(t.template) for t in self.templates.values()) // total_templates if total_templates > 0 else 0
        }

    def __repr__(self) -> str:
        """String representation of prompt manager."""
        return f"PromptManager(language={self.language.value}, templates={len(self.templates)})"
