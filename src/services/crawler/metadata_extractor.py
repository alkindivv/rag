#!/usr/bin/env python3
"""
MetadataExtractor - Pure HTML metadata and relationship extraction
Uses centralized configuration - no hardcoded values
"""

import asyncio
import logging
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from bs4 import BeautifulSoup
from src.config.crawler_config import CrawlerConfig
from src.config.ai_config import AIConfig
import google.generativeai as genai


class RelationshipReference(BaseModel):
    """Pydantic model for relationship reference structure"""
    regulation_reference: str = Field(..., description="Reference to the regulation")
    reference_link: Optional[str] = Field(None, description="Link to the referenced regulation")

class UjiMateriDecision(BaseModel):
    """Clear UJI MATERI decision structure with explicit provision format"""
    decision_number: str = Field(..., description="Court decision number")
    pdf_url: Optional[str] = Field(None, description="PDF URL for the decision")
    decision_content: str = Field(..., description="Raw decision content")
    provision_references: Optional[List[str]] = Field(None, description="Clear provision format: Pasal_[nomor]/Ayat_[nomor]/[UU/PP]_[nomor]_[tahun]")
    decision_summary: Optional[str] = Field(None, description="Summary of the court decision")
    decision_type: Optional[str] = Field(None, description="Type of decision (bertentangan/tidak bertentangan)")
    legal_basis: Optional[str] = Field(None, description="Legal basis for the decision")
    binding_status: Optional[str] = Field(None, description="Binding status of the decision")
    conditions: Optional[str] = Field(None, description="Special conditions or requirements")
    interpretation: Optional[str] = Field(None, description="Legal interpretation or meaning")

class RelationshipData(BaseModel):
    """Pydantic model for relationship data structure"""
    relationship_type: str = Field(..., description="Type of relationship (mengubah, dicabut, dll)")
    references: List[RelationshipReference] = Field(default_factory=list, description="List of regulation references")

class MetadataExtractor:
    """Extract metadata from BPK legal documents using pure HTML structure analysis"""

    def __init__(self):
        """Initialize with centralized configuration"""
        self.config = CrawlerConfig
        self.metadata_fields = self.config.get_metadata_fields()
        self.text_patterns = self.config.get_text_patterns()

        # Initialize AI config for UJI MATERI parsing
        self.ai_config = AIConfig()
        gemini_api_key = self._read_gemini_api_key()
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.ai_model = genai.GenerativeModel(self.ai_config.model.model_name)
            logging.info("AI parsing enabled with Gemini API")
        else:
            self.ai_model = None
            logging.warning("No GEMINI_API_KEY found, AI parsing disabled")

    def _read_gemini_api_key(self) -> Optional[str]:
        """Read GEMINI_API_KEY directly from .env file"""
        try:
            env_file = Path(".env")
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            return line.split("=", 1)[1].strip().strip('"').strip("'")
            return None
        except Exception as e:
            logging.debug(f"Error reading .env file: {e}")
            return None

    def extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from BPK document page using pure HTML structure analysis"""
        metadata = {}

        try:
            self._extract_from_metadata_section(soup, metadata)
            self._extract_from_row_layout(soup, metadata)
            self._extract_from_table_structure(soup, metadata)
            self._extract_from_definition_list(soup, metadata)
            self._post_process_metadata(metadata)

            if not metadata.get('title'):
                self._extract_title_from_page(soup, metadata)

            # Extract UJI MATERI information
            uji_materi_data = self.extract_uji_materi(soup)
            if uji_materi_data:
                metadata['uji_materi'] = uji_materi_data

            logging.info(f"Extracted metadata fields: {list(metadata.keys())}")

        except Exception as e:
            logging.error(f"Error in metadata extraction: {str(e)}")

        return metadata

    def _extract_from_metadata_section(self, soup: BeautifulSoup, metadata: Dict):
        """Extract metadata from dedicated metadata section"""
        try:
            metadata_headers = soup.find_all(text=re.compile(r'METADATA\s+PERATURAN', re.IGNORECASE))

            for header in metadata_headers:
                container = header.parent
                while container and container.name not in ['div', 'section', 'article']:
                    container = container.parent

                if container:
                    self._extract_from_container(container, metadata)

        except Exception as e:
            logging.debug(f"Error extracting from metadata section: {str(e)}")

    def _extract_from_row_layout(self, soup: BeautifulSoup, metadata: Dict):
        """Extract metadata from Bootstrap row layout structure"""
        try:
            rows = soup.find_all('div', class_=re.compile(r'row'))

            for row in rows:
                cols = row.find_all('div', class_=re.compile(r'col'))

                if len(cols) >= 2:
                    label_col = cols[0]
                    value_col = cols[1]

                    label_text = label_col.get_text(strip=True)
                    value_text = value_col.get_text(strip=True)

                    field_name = self._match_metadata_field(label_text)
                    if field_name and value_text:
                        metadata[field_name] = value_text

        except Exception as e:
            logging.debug(f"Error extracting from row layout: {str(e)}")

    def _extract_from_table_structure(self, soup: BeautifulSoup, metadata: Dict):
        """Extract metadata from table structure"""
        try:
            tables = soup.find_all('table')

            for table in tables:
                rows = table.find_all('tr')

                for row in rows:
                    cells = row.find_all(['td', 'th'])

                    if len(cells) >= 2:
                        label_text = cells[0].get_text(strip=True)
                        value_text = cells[1].get_text(strip=True)

                        field_name = self._match_metadata_field(label_text)
                        if field_name and value_text:
                            metadata[field_name] = value_text

        except Exception as e:
            logging.debug(f"Error extracting from table structure: {str(e)}")

    def _extract_from_definition_list(self, soup: BeautifulSoup, metadata: Dict):
        """Extract metadata from definition list (dl, dt, dd) structure"""
        try:
            definition_lists = soup.find_all('dl')

            for dl in definition_lists:
                terms = dl.find_all('dt')
                definitions = dl.find_all('dd')

                if len(terms) == len(definitions):
                    for term, definition in zip(terms, definitions):
                        label_text = term.get_text(strip=True)
                        value_text = definition.get_text(strip=True)

                        field_name = self._match_metadata_field(label_text)
                        if field_name and value_text:
                            metadata[field_name] = value_text

        except Exception as e:
            logging.debug(f"Error extracting from definition list: {str(e)}")

    def _extract_from_container(self, container, metadata: Dict):
        """Extract metadata from a generic container element"""
        try:
            text_content = container.get_text()
            lines = text_content.split('\n')

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                field_name = self._match_metadata_field(line)
                if field_name and i + 1 < len(lines):
                    value = lines[i + 1].strip()
                    if value and value != line:
                        metadata[field_name] = value

        except Exception as e:
            logging.debug(f"Error extracting from container: {str(e)}")

    def _match_metadata_field(self, text: str) -> Optional[str]:
        """Match text to known metadata field names and return doc_ prefixed names"""
        text_clean = text.strip().lower()

        # Direct mapping to doc_ prefixed field names
        field_mapping = {
            'tipe_dokumen': 'doc_type',
            'title': 'doc_title',
            'teu': 'doc_teu',
            'number': 'doc_number',
            'year': 'doc_year',
            'form': 'doc_form',
            'form_short': 'doc_form_short',
            'place_enacted': 'doc_place_enacted',
            'date_enacted': 'doc_date_enacted',
            'date_promulgated': 'doc_date_promulgated',
            'date_effective': 'doc_date_effective',
            'source': 'doc_source',
            'subject': 'doc_subject',
            'status': 'doc_status',
            'language': 'doc_language',
            'location': 'doc_location',
            'field': 'doc_field',
        }

        for field_name, field_labels in self.metadata_fields.items():
            for label in field_labels:
                if text_clean == label.lower() or text_clean.startswith(label.lower()):
                    return field_mapping.get(field_name, field_name)

        return None

    def _post_process_metadata(self, metadata: Dict):
        """Post-process extracted metadata to clean and normalize values"""
        try:
            if 'doc_subject' in metadata:
                subjects = self._process_subjects(metadata['doc_subject'])
                if subjects:
                    metadata['doc_subject'] = subjects
                else:
                    metadata['doc_subject_raw'] = metadata['doc_subject']

            for field in ['doc_number', 'doc_year']:
                if field in metadata:
                    numeric_match = re.search(self.text_patterns['numeric_pattern'], metadata[field])
                    if numeric_match:
                        metadata[field] = numeric_match.group()

            for field in ['doc_date_enacted', 'doc_date_promulgated', 'doc_date_effective']:
                if field in metadata:
                    metadata[field] = self._clean_date(metadata[field])

            # Handle legacy tipe_dokumen field
            if 'tipe_dokumen' in metadata:
                metadata['doc_type'] = metadata.pop('tipe_dokumen')

        except Exception as e:
            logging.debug(f"Error in post-processing: {str(e)}")

    def _process_subjects(self, subject_text: str) -> List[str]:
        """Process subject text into a list of individual subjects"""
        subjects = []

        if not subject_text or subject_text.lower() in self.text_patterns['excluded_subjects']:
            return subjects

        delimiters = self.text_patterns['subject_delimiters']
        parts = [subject_text]

        for delimiter in delimiters:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(delimiter))
            parts = new_parts

        for part in parts:
            subject = part.strip()
            if subject and len(subject) > 1:
                subject = re.sub(r'^(subjek|subject)[\s:]*', '', subject, flags=re.IGNORECASE)
                subject = re.sub(r'[\s:]*$', '', subject)

                if subject and len(subject) > 1:
                    subjects.append(subject.upper())

        return subjects

    def _clean_date(self, date_text: str) -> str:
        """Clean and normalize date text"""
        if not date_text:
            return date_text

        date_text = re.sub(r'\s+', ' ', date_text.strip())
        return date_text

    def _extract_title_from_page(self, soup: BeautifulSoup, metadata: Dict):
        """Extract title from page structure if not found in metadata"""
        try:
            for tag in ['h1', 'h2', 'h3']:
                title_elem = soup.find(tag)
                if title_elem:
                    title_text = title_elem.get_text(strip=True)
                    if title_text and len(title_text) > 10:
                        metadata['title'] = title_text
                        break

            if not metadata.get('title'):
                meta_title = soup.find('meta', {'name': 'title'})
                if meta_title and meta_title.get('content'):
                    metadata['title'] = meta_title.get('content')

            if not metadata.get('title'):
                page_title = soup.find('title')
                if page_title:
                    title_text = page_title.get_text(strip=True)
                    if title_text and 'BPK' not in title_text:
                        metadata['title'] = title_text

        except Exception as e:
            logging.debug(f"Error extracting title from page: {str(e)}")

    def extract_relationships(self, soup: BeautifulSoup) -> Dict[str, List[Dict]]:
        """Extract relationship information using pure semantic HTML extraction with Pydantic models"""
        relationships = {}

        try:
            # Find STATUS PERATURAN section
            status_headers = soup.find_all('h4', class_='text-primary')

            valid_status_headers = []
            for header in status_headers:
                header_text = header.get_text(strip=True)
                if 'STATUS' in header_text.upper() and 'PERATURAN' in header_text.upper():
                    valid_status_headers.append(header)

            if not valid_status_headers:
                status_headers = soup.find_all(text=re.compile(r'STATUS.*PERATURAN', re.IGNORECASE))
            else:
                status_headers = valid_status_headers

            for header in status_headers:
                if hasattr(header, 'parent'):
                    container = header.parent
                    while container and container.name not in ['div'] or not container.get('class') or 'card' not in ' '.join(container.get('class', [])):
                        container = container.parent
                        if container and container.name in ['body', 'html']:
                            break

                    if not container or container.name in ['body', 'html']:
                        container = header.parent
                else:
                    container = header

                if container:
                    self._extract_relationship_sections_pure(container, relationships)

        except Exception as e:
            logging.error(f"Error extracting relationships: {str(e)}")

        # Convert to structured format using Pydantic models
        structured_relationships = {}
        for rel_type, refs in relationships.items():
            if isinstance(refs, list) and refs:
                # Convert RelationshipReference objects to dict format
                ref_dicts = []
                for ref in refs:
                    if isinstance(ref, RelationshipReference):
                        ref_dicts.append(ref.dict())
                    elif isinstance(ref, dict):
                        ref_dicts.append(ref)

                structured_relationships[rel_type.lower().replace(' ', '_')] = ref_dicts

        return structured_relationships

    def extract_uji_materi(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract UJI MATERI (judicial review) information using pure HTML structure analysis"""
        uji_materi_decisions = []

        try:
            # Find UJI MATERI section
            h4_tags = soup.find_all('h4', class_='text-primary')
            uji_materi_header = None

            for h4 in h4_tags:
                h4_text = h4.get_text(strip=True).upper()
                if 'UJI MATERI' in h4_text:
                    uji_materi_header = h4
                    break

            if not uji_materi_header:
                logging.debug("No UJI MATERI section found")
                return uji_materi_decisions

            # Find the container that holds the UJI MATERI content
            container = uji_materi_header.parent
            while container and container.name not in ['div'] or not container.get('class') or 'card' not in ' '.join(container.get('class', [])):
                container = container.parent
                if container and container.name in ['body', 'html']:
                    break

            if not container or container.name in ['body', 'html']:
                container = uji_materi_header.find_next('div', class_='container')

            if container:
                uji_materi_decisions = self._extract_uji_materi_decisions(container)

            logging.info(f"Extracted {len(uji_materi_decisions)} UJI MATERI decisions")

        except Exception as e:
            logging.error(f"Error extracting UJI MATERI: {str(e)}")

        return uji_materi_decisions

    def _extract_uji_materi_decisions(self, container) -> List[Dict[str, Any]]:
        """Extract individual UJI MATERI decisions from container"""
        decisions = []

        try:
            logging.debug(f"Starting UJI MATERI extraction from container: {container.name if container else None}")

            # Find all decision header rows (contains PUTUSAN Nomor) - fix BeautifulSoup class selector
            decision_headers = container.find_all('div', class_='col-12 fw-semibold bg-light-primary p-4')
            logging.debug(f"Found {len(decision_headers)} decision headers with exact class match")

            # Fallback: if no exact match, find divs that contain PUTUSAN text with required classes
            if not decision_headers:
                logging.debug("No exact class match, trying fallback logic")
                all_divs = container.find_all('div')
                logging.debug(f"Total divs in container: {len(all_divs)}")

                for div in all_divs:
                    div_classes = div.get('class', [])
                    required_classes = {'col-12', 'fw-semibold', 'bg-light-primary', 'p-4'}
                    if required_classes.issubset(set(div_classes)):
                        div_text = div.get_text(strip=True)
                        logging.debug(f"Found div with required classes, text: {div_text[:50]}...")
                        if 'PUTUSAN' in div_text.upper():
                            logging.debug(f"Adding PUTUSAN div to headers")
                            decision_headers.append(div)

            logging.debug(f"Total decision headers found: {len(decision_headers)}")

            for i, header in enumerate(decision_headers):
                try:
                    header_text = header.get_text(strip=True)
                    logging.debug(f"Processing header {i+1}: {header_text[:100]}...")

                    # Skip if not a decision header
                    if 'PUTUSAN' not in header_text.upper() or 'NOMOR' not in header_text.upper():
                        logging.debug(f"Skipping header {i+1}: doesn't contain PUTUSAN or NOMOR")
                        continue

                    logging.debug(f"Processing valid decision header {i+1}")
                    decision = {}

                    # Extract decision number and download link
                    putusan_link = header.find('a', href=True)
                    if putusan_link:
                        decision_number = putusan_link.get_text(strip=True)
                        pdf_url = putusan_link.get('href')

                        # Normalize PDF URL
                        if pdf_url and not pdf_url.startswith('http'):
                            pdf_url = f"https://peraturan.bpk.go.id{pdf_url}"

                        decision['decision_number'] = decision_number
                        decision['pdf_url'] = pdf_url
                    else:
                        # Extract decision number from text if no link
                        import re
                        number_match = re.search(r'PUTUSAN\s+Nomor\s+([^\s]+)', header_text, re.IGNORECASE)
                        if number_match:
                            decision['decision_number'] = number_match.group(1)
                        decision['pdf_url'] = None

                    # Find the corresponding content within the same row - improved logic
                    parent_row = header.find_parent('div', class_='row')
                    decision_content = None
                    logging.debug(f"Parent row found: {parent_row is not None}")

                    if parent_row:
                        # Find all col-12 divs in this row
                        col_divs = parent_row.find_all('div', class_='col-12')
                        logging.debug(f"Found {len(col_divs)} col-12 divs in parent row")

                        for j, col_div in enumerate(col_divs):
                            # Skip the header div, get the content div
                            if col_div != header:
                                div_classes = col_div.get('class', [])
                                logging.debug(f"Col-div {j}: classes={div_classes}")
                                # Skip divs with bg-light-primary (those are headers)
                                if 'bg-light-primary' not in div_classes:
                                    content_text = col_div.get_text(strip=True)
                                    logging.debug(f"Content text length: {len(content_text)}")
                                    if content_text and len(content_text) > 20:  # Meaningful content
                                        logging.debug(f"Found meaningful content: {content_text[:100]}...")
                                        decision_content = content_text
                                        break

                    # Fallback: look for next sibling div with content
                    if not decision_content:
                        logging.debug("No content found in parent row, trying sibling approach")
                        next_sibling = header.find_next_sibling('div')
                        sibling_count = 0
                        while next_sibling and sibling_count < 5:  # Limit search
                            sibling_count += 1
                            if 'col-12' in next_sibling.get('class', []):
                                sibling_classes = next_sibling.get('class', [])
                                logging.debug(f"Sibling {sibling_count}: classes={sibling_classes}")
                                if 'bg-light-primary' not in sibling_classes:
                                    content_text = next_sibling.get_text(strip=True)
                                    logging.debug(f"Sibling content length: {len(content_text)}")
                                    if content_text and len(content_text) > 20:
                                        logging.debug(f"Found content in sibling: {content_text[:100]}...")
                                        decision_content = content_text
                                        break
                            next_sibling = next_sibling.find_next_sibling('div')

                    if decision_content:
                        decision['decision_content'] = decision_content
                        logging.debug(f"Added decision content, length: {len(decision_content)}")

                        # Parse content using AI if available
                        parsed_content = self._parse_uji_materi_content_with_ai(decision_content)
                        if parsed_content:
                            # Handle both dict and list responses from AI
                            if isinstance(parsed_content, dict):
                                logging.debug(f"AI parsing successful, keys: {list(parsed_content.keys())}")
                                decision.update(parsed_content)
                            elif isinstance(parsed_content, list) and len(parsed_content) > 0:
                                # If AI returns list, take first item if it's a dict
                                if isinstance(parsed_content[0], dict):
                                    logging.debug(f"AI returned list, using first item with keys: {list(parsed_content[0].keys())}")
                                    decision.update(parsed_content[0])
                                else:
                                    logging.debug("AI returned list but first item is not dict")
                            else:
                                logging.debug(f"AI returned unexpected format: {type(parsed_content)}")
                        else:
                            logging.debug("AI parsing failed or unavailable")

                    # Only add if we have meaningful content and convert to Pydantic model
                    logging.debug(f"Decision validation - number: {bool(decision.get('decision_number'))}, content: {bool(decision.get('decision_content'))}")
                    if decision.get('decision_number') and decision.get('decision_content'):
                        try:
                            uji_materi_model = UjiMateriDecision(**decision)
                            decisions.append(uji_materi_model.dict())
                            logging.debug(f"Successfully added UjiMateriDecision model")
                        except Exception as e:
                            logging.debug(f"Error creating UjiMateriDecision model: {e}")
                            logging.debug(f"Adding raw decision data instead")
                            decisions.append(decision)
                    else:
                        logging.debug(f"Decision rejected - missing required fields")

                except Exception as e:
                    logging.error(f"Error processing UJI MATERI decision {i+1}: {str(e)}")
                    import traceback
                    logging.debug(f"Full traceback: {traceback.format_exc()}")
                    continue

        except Exception as e:
            logging.error(f"Error extracting UJI MATERI decisions: {str(e)}")
            import traceback
            logging.debug(f"Full traceback: {traceback.format_exc()}")

        logging.debug(f"Final UJI MATERI decisions count: {len(decisions)}")
        return decisions

    def _parse_uji_materi_content_with_ai(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse UJI MATERI content using AI to extract structured information"""
        if not self.ai_model:
            return None

        try:
            prompt = f"""
Ekstrak informasi dari konten Uji Materi berikut:

"{content}"

Berikan output JSON:
{{
    "provision_references": ["Format: Pasal_[nomor]/Ayat_[nomor]/[UU/PP]_[nomor]_[tahun] - contoh: Pasal_197/Ayat_1/UU_8_1981, Pasal_263/Ayat_1/UU_8_1981"],
    "decision_summary": "ringkasan singkat putusan mahkamah konstitusi",
    "decision_type": "bertentangan/tidak bertentangan/bersyarat",
    "legal_basis": "UUD NRI Tahun 1945",
    "binding_status": "tidak mempunyai kekuatan hukum mengikat/berlaku",
    "conditions": "syarat khusus jika ada",
    "interpretation": "penafsiran MK jika ada"
}}

Ekstrak SEMUA pasal/ayat yang disebutkan dalam format provision_references. Berikan hanya JSON valid."""

            response = self.ai_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.ai_config.model.temperature,
                    max_output_tokens=self.ai_config.model.max_tokens
                )
            )

            if response.text:
                # Clean response and parse JSON
                clean_response = response.text.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response.replace('```json', '').replace('```', '').strip()
                elif clean_response.startswith('```'):
                    clean_response = clean_response.replace('```', '').strip()

                # Handle potential JSON array or object
                try:
                    parsed_data = json.loads(clean_response)
                    logging.debug(f"Raw AI response type: {type(parsed_data)}")

                    # Ensure we return a dict, not a list
                    if isinstance(parsed_data, list) and len(parsed_data) > 0:
                        if isinstance(parsed_data[0], dict):
                            parsed_data = parsed_data[0]
                            logging.info(f"AI parsed UJI MATERI content successfully (converted from list)")
                        else:
                            logging.warning("AI returned list but first item is not dict")
                            return None
                    elif isinstance(parsed_data, dict):
                        logging.info(f"AI parsed UJI MATERI content successfully")
                    else:
                        logging.warning(f"AI returned unexpected data type: {type(parsed_data)}")
                        return None

                    # Simple data type fixes for practical use
                    if parsed_data:
                        # Fix data types for Pydantic model compatibility
                        string_fields = ['conditions', 'interpretation', 'decision_type', 'legal_basis', 'binding_status', 'decision_summary']
                        for field in string_fields:
                            if field in parsed_data:
                                if isinstance(parsed_data[field], list):
                                    parsed_data[field] = ', '.join(str(item) for item in parsed_data[field]) if parsed_data[field] else ""
                                elif parsed_data[field] is None:
                                    parsed_data[field] = ""

                        # Ensure provision_references is a list
                        if 'provision_references' in parsed_data:
                            if not isinstance(parsed_data['provision_references'], list):
                                if isinstance(parsed_data['provision_references'], str) and parsed_data['provision_references']:
                                    parsed_data['provision_references'] = [parsed_data['provision_references']]
                                else:
                                    parsed_data['provision_references'] = []

                    return parsed_data

                except json.JSONDecodeError as je:
                    logging.debug(f"JSON decode error: {je}")
                    logging.debug(f"Raw response: {clean_response[:200]}...")
                    return None

        except Exception as e:
            logging.debug(f"AI parsing failed: {str(e)}")
            return None

        return None



    #     """Extract regulation information from content text"""
    #     import re

    #     regulation_info = {
    #         'type': 'UU',
    #         'number': '',
    #         'year': '',
    #         'title': ''
    #     }

    #     try:
    #         # Extract UU/PP/etc pattern
    #         type_pattern = r'(Undang-Undang|Peraturan Pemerintah|UU|PP)\s+(?:Nomor\s+)?(\d+)\s+Tahun\s+(\d{4})'
    #         match = re.search(type_pattern, content, re.IGNORECASE)

    #         if match:
    #             reg_type_full = match.group(1).lower()
    #             regulation_info['number'] = match.group(2)
    #             regulation_info['year'] = match.group(3)

    #             # Map to short type
    #             if 'undang-undang' in reg_type_full or reg_type_full == 'uu':
    #                 regulation_info['type'] = 'UU'
    #             elif 'peraturan pemerintah' in reg_type_full or reg_type_full == 'pp':
    #                 regulation_info['type'] = 'PP'
    #             else:
    #                 regulation_info['type'] = 'UU'  # Default

    #         # Extract title if present
    #         title_pattern = r'tentang\s+([^(]+)'
    #         title_match = re.search(title_pattern, content, re.IGNORECASE)
    #         if title_match:
    #             regulation_info['title'] = title_match.group(1).strip()

    #     except Exception as e:
    #         logging.debug(f"Error extracting regulation info: {e}")

    #     return regulation_info

    def _extract_relationship_sections_pure(self, container, relationships: Dict):
        """Extract relationship sections using pure semantic HTML extraction with Pydantic models"""
        try:
            # Find all relationship header divs
            relationship_divs = container.find_all('div', class_=['col-12', 'fw-semibold', 'bg-light-primary', 'p-4'])

            processed_sections = set()

            for rel_div in relationship_divs:
                div_id = id(rel_div)
                if div_id in processed_sections:
                    continue

                # Extract the relationship label exactly as it appears
                rel_text = rel_div.get_text(strip=True)

                # Clean up the label but preserve the Indonesian term
                if ':' in rel_text:
                    label = rel_text.replace(':', '').strip()
                else:
                    label = rel_text.strip()

                if label:
                    # Find the content that follows this relationship header
                    content = self._find_relationship_content_pure(rel_div)

                    if content:
                        relationships[label] = content
                        processed_sections.add(div_id)

        except Exception as e:
            logging.error(f"Error extracting relationship sections: {str(e)}")





    def _clean_regulation_reference(self, text: str) -> str:
        """Clean regulation reference text to extract key information"""
        if not text:
            return ""

        text = text.strip()
        text = re.sub(r'^\d+\.?\s*', '', text)

        if re.match(r'^[a-z]\.?\s+', text, re.IGNORECASE):
            text_without_letter = re.sub(r'^[a-z]\.?\s*', '', text, flags=re.IGNORECASE)

            regulation_keywords = self.text_patterns['regulation_keywords']
            is_broken_regulation = any(text_without_letter.lower().startswith(keyword) for keyword in regulation_keywords)

            if not is_broken_regulation:
                text = text_without_letter

        text = re.sub(r'(\w)tentang(\w)', r'\1 tentang \2', text)
        text = re.sub(r'tentang\s+', 'tentang ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _extract_text_references(self, container) -> List[str]:
        """Extract regulation references from plain text when no links are available"""
        references = []

        try:
            text_content = container.get_text()

            patterns = [
                r'(UU\s+No\.?\s*\d+\s+Tahun\s+\d{4})',
                r'(PP\s+No\.?\s*\d+\s+Tahun\s+\d{4})',
                r'(Peraturan\s+\w+\s+No\.?\s*\d+\s+Tahun\s+\d{4})',
                r'(Undang-[Uu]ndang\s+Nomor\s+\d+\s+Tahun\s+\d{4})',
                r'(Peraturan\s+Pemerintah\s+Nomor\s+\d+\s+Tahun\s+\d{4})'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                for match in matches:
                    clean_match = match.strip()
                    if clean_match and clean_match not in references:
                        references.append(clean_match)

        except Exception as e:
            logging.debug(f"Error extracting text references: {str(e)}")

        return references

    def _find_relationship_content_pure(self, rel_div) -> List[RelationshipReference]:
        """Find content that follows a relationship header div using pure semantic extraction"""
        content = []

        try:
            # Look for the next row div that contains the actual content
            current = rel_div.parent
            if current:
                next_row = current.find_next_sibling('div', class_='row')
                if next_row:
                    content = self._extract_content_from_container(next_row)

        except Exception as e:
            logging.error(f"Error finding relationship content: {str(e)}")

        return content

    def _extract_content_from_container(self, container) -> List[RelationshipReference]:
        """Extract content with structured format using Pydantic models"""
        content = []

        try:
            # Extract from list items
            # Extract from list items first (most common structure)
            list_items = container.find_all('li')
            for li in list_items:
                li_text = li.get_text(separator=' ', strip=True)
                if not li_text:
                    continue

                # Check if has link
                link = li.find('a', href=True)
                if link:
                    href = link.get('href')
                    if href.startswith('/'):
                        href = f"https://peraturan.bpk.go.id{href}"

                    # Create structured object using Pydantic model
                    regulation_ref = RelationshipReference(
                        regulation_reference=li_text,
                        reference_link=href
                    )
                    content.append(regulation_ref)
                else:
                    # No link - plain text (like Staatsblad entries)
                    regulation_ref = RelationshipReference(
                        regulation_reference=li_text,
                        reference_link=None
                    )
                    content.append(regulation_ref)

        except Exception as e:
            logging.error(f"Error extracting content: {str(e)}")

        return content
