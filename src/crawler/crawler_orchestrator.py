#!/usr/bin/env python3
"""
CrawlerOrchestrator - High-level coordination and processing
Uses centralized configuration - no hardcoded values or duplications
"""

import asyncio
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any

# Remove PDF processing - crawler should only handle web scraping and downloads
# PDF processing will be handled by separate services

from .web_scraper import WebScraper
from .file_downloader import FileDownloader
from src.config.crawler_config import CrawlerConfig



class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime and date objects"""
    def default(self, o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return super().default(o)


class CrawlerOrchestrator:
    """Main crawler class that orchestrates the crawling process using centralized configuration"""

    def __init__(self, output_dir: str = "./data/json", pdf_dir: str = "./data/pdfs", regulation_type: str = ""):
        """Initialize with centralized configuration"""
        self.config = CrawlerConfig
        self.output_dir = Path(output_dir)
        self.pdf_dir = Path(pdf_dir)
        self.regulation_type = regulation_type
        # Removed PDF processor - crawler should only handle web scraping and downloads
        self.file_downloader = FileDownloader()

        self.config.create_directories([self.output_dir, self.pdf_dir])

    async def crawl_regulations(self,
                               query: str = "",
                               regulation_type: str = "",
                               year: str = "",
                               limit: int = 10) -> List[Dict]:
        """Crawl regulations from BPK website"""
        logging.info(f"Starting BPK crawl with query='{query}', type='{regulation_type}', year='{year}', limit={limit}")

        crawler = WebScraper()
        return await self._process_regulations_with_crawler(
            crawler, query, regulation_type, year, limit
        )

    async def _process_regulations_with_crawler(self, crawler, query: str, regulation_type: str, year: str, limit: int) -> List[Dict]:
        """Process regulations with the given crawler instance with smart skipping"""
        try:
            processed_regulations = []
            search_limit = limit * 3
            max_attempts = 5
            attempt = 0

            while len(processed_regulations) < limit and attempt < max_attempts:
                attempt += 1
                logging.info(f"Search attempt {attempt}: Looking for {search_limit} regulations to find {limit - len(processed_regulations)} new ones")

                regulations = await crawler.search_regulations(
                    query=query,
                    regulation_type=regulation_type,
                    year=year,
                    limit=search_limit
                )

                logging.info(f"Found {len(regulations)} regulations in attempt {attempt}")

                if not regulations:
                    logging.info("No more regulations found")
                    break

                for i, regulation in enumerate(regulations, 1):
                    if len(processed_regulations) >= limit:
                        break

                    logging.info(f"Processing regulation {i}/{len(regulations)}: {regulation.get('title', 'Unknown')}")

                    try:
                        detailed_info = await crawler.get_regulation_details(regulation['detail_url'])
                        regulation.update(detailed_info)

                        regulation = self._clean_regulation_data(regulation)

                        json_filename = self.config.generate_json_filename(regulation)
                        json_path = self.output_dir / json_filename

                        if json_path.exists():
                            logging.info(f"Regulation already exists, skipping: {regulation.get('title', 'Unknown')}")
                            continue

                        if regulation.get('pdf_url'):
                            if hasattr(crawler, 'download_pdf'):
                                filename = self.config.generate_pdf_filename(regulation)
                                pdf_path_str = await crawler.download_pdf(regulation['pdf_url'], str(self.pdf_dir), filename)
                                pdf_path = Path(pdf_path_str) if pdf_path_str else None
                            else:
                                pdf_path = await self.download_pdf(regulation['pdf_url'], regulation)

                            regulation['pdf_path'] = str(pdf_path) if pdf_path else None

                            if pdf_path and pdf_path.exists():
                                # Crawler only handles download - PDF processing will be done separately
                                regulation['doc_processing_status'] = 'pdf_downloaded'
                                logging.info(f"PDF downloaded successfully: {pdf_path}")
                            else:
                                regulation['doc_processing_status'] = 'pdf_download_failed'
                                logging.warning(f"Failed to download PDF")

                        text_path = self.save_text_file(regulation)
                        if text_path:
                            regulation['text_path'] = str(text_path)
                            logging.info(f"Text file saved successfully: {text_path}")
                        else:
                            logging.info(f"No content available for text file: {regulation.get('doc_title', 'Unknown')}")

                        await self.save_regulation_json(regulation)

                        processed_regulations.append(regulation)
                        logging.info(f"Successfully processed new regulation: {regulation.get('doc_title', 'Unknown')}")

                    except Exception as e:
                        logging.error(f"Error processing regulation {regulation.get('doc_title', 'Unknown')}: {str(e)}")
                        try:
                            await self.save_regulation_json(regulation)
                            processed_regulations.append(regulation)
                        except Exception as save_error:
                            logging.error(f"Error saving failed regulation: {str(save_error)}")
                        continue

                if len(processed_regulations) < limit:
                    search_limit = min(search_limit * 2, 100)
                    logging.info(f"Still need {limit - len(processed_regulations)} more regulations, increasing search limit to {search_limit}")

            logging.info(f"Successfully processed {len(processed_regulations)} new regulations (target was {limit})")
            return processed_regulations

        except Exception as e:
            logging.error(f"Error in crawl_regulations: {str(e)}")
            raise

    async def download_pdf(self, pdf_url: str, regulation: Dict) -> Optional[Path]:
        """Download PDF file from URL"""
        return await self.file_downloader.download_pdf_simple(pdf_url, regulation, self.pdf_dir)

    async def save_regulation_json(self, regulation: Dict, update_existing: bool = False):
        """Save regulation data as JSON file - handles both creation and updates"""
        try:
            filename = self.config.generate_json_filename(regulation)
            json_path = self.output_dir / filename

            if json_path.exists() and not update_existing:
                logging.info(f"JSON file already exists, skipping save: {json_path}")
                return

            # Handle updates to existing files
            if json_path.exists() and update_existing:
                # Read existing data
                with open(json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

                # Update with new content (preserve existing data, add new fields)
                existing_data.update({
                    # 'content': regulation.get('content'),
                    'document_tree': regulation.get('document_tree'),
                    'pdf_extraction_metadata': regulation.get('pdf_extraction_metadata'),
                    'pasal_metadata': regulation.get('pasal_metadata'),
                    'structure_flags': regulation.get('structure_flags'),
                    'structure_analysis': regulation.get('structure_analysis'),
                    'chapters': regulation.get('chapters'),
                    'articles': regulation.get('articles'),
                    'processing_status': 'pdf_processed' if regulation.get('content') else existing_data.get('processing_status', 'pdf_downloaded'),
                    'last_updated': datetime.now().isoformat()
                })
                regulation = existing_data
                logging.info(f"Updated existing JSON file: {json_path}")
            else:
                regulation['last_updated'] = datetime.now().isoformat()
                logging.info(f"Successfully processed crawler metadata: {regulation.get('title', 'Unknown')}")

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(regulation, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)

            logging.debug(f"Saved regulation JSON: {json_path}")

        except Exception as e:
            logging.error(f"Error saving regulation JSON: {str(e)}")

    def save_text_file(self, regulation: Dict) -> Optional[Path]:
        """Save cleaned regulation text to a text file"""
        try:
            filename = self.config.generate_text_filename(regulation)
            text_path = self.output_dir / filename

            if text_path.exists():
                logging.info(f"Text file already exists: {text_path}")
                return text_path

            # Use standardized content field with fallbacks
            text = regulation.get('doc_content') or regulation.get('content') or regulation.get('full_text') or regulation.get('text', '')
            if not text:
                logging.info("No content available to save as text file")
                return None

            # Create a basic text file with available metadata if no content
            if not text.strip():
                text = f"REGULATION METADATA (Crawled by BPK Crawler)\n"
                text += f"======================================================\n\n"
                text += f"Title: {regulation.get('doc_title', 'Unknown')}\n"
                text += f"Type: {regulation.get('doc_type', 'Unknown')}\n"
                text += f"Number: {regulation.get('doc_number', 'Unknown')}\n"
                text += f"Year: {regulation.get('doc_year', 'Unknown')}\n"
                text += f"Status: {regulation.get('doc_status', 'Unknown')}\n"
                text += f"Language: {regulation.get('doc_language', 'Bahasa Indonesia')}\n\n"
                text += f"URLs:\n"
                text += f"- Detail URL: {regulation.get('detail_url', 'N/A')}\n"
                text += f"- PDF URL: {regulation.get('pdf_url', 'N/A')}\n\n"
                text += f"Processing Status:\n"
                text += f"- PDF Downloaded: {bool(regulation.get('pdf_path'))}\n"
                text += f"- Requires PDF Processing: {not bool(regulation.get('doc_content'))}\n\n"
                text += f"Note: Full document content requires PDF processing by separate services.\n"

            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)

            logging.info(f"Saved text file: {text_path}")
            return text_path

        except Exception as e:
            logging.error(f"Error saving text file: {str(e)}")
            return None

    def _clean_regulation_data(self, regulation: Dict) -> Dict:
        """Clean and structure regulation data for consistent output"""
        try:
            clean_reg = {}

            # Core document metadata with doc_ prefix
            clean_reg['doc_source'] = 'BPK'
            clean_reg['doc_id'] = self._generate_doc_id(regulation)
            clean_reg['doc_type'] = regulation.get('doc_type', 'Unknown')
            clean_reg['doc_title'] = regulation.get('doc_title', 'Unknown')
            clean_reg['doc_teu'] = regulation.get('doc_teu', None)
            clean_reg['doc_number'] = regulation.get('doc_number', 'Unknown')
            clean_reg['doc_form'] = regulation.get('doc_form', 'Unknown')
            clean_reg['doc_form_short'] = regulation.get('doc_form_short', self.config.extract_form_short(clean_reg['doc_form']))
            clean_reg['doc_year'] = regulation.get('doc_year', 'Unknown')
            clean_reg['doc_place_enacted'] = regulation.get('doc_place_enacted', None)
            clean_reg['doc_date_enacted'] = self._normalize_date_to_iso(regulation.get('doc_date_enacted'))
            clean_reg['doc_date_promulgated'] = self._normalize_date_to_iso(regulation.get('doc_date_promulgated'))
            clean_reg['doc_date_effective'] = self._normalize_date_to_iso(regulation.get('doc_date_effective'))
            clean_reg['doc_subject'] = regulation.get('doc_subject', [])
            clean_reg['doc_status'] = regulation.get('doc_status', 'Unknown')
            clean_reg['doc_language'] = regulation.get('doc_language', 'Bahasa Indonesia')
            clean_reg['doc_location'] = regulation.get('doc_location', None)
            clean_reg['doc_field'] = regulation.get('doc_field', None)
            clean_reg['doc_content'] = regulation.get('doc_content', None)
            clean_reg['doc_processing_status'] = regulation.get('doc_processing_status', None)

            # Extract all relationships using centralized structure with Pydantic models
            # Keep relationships organized under 'relationships' key for better RAG structure
            relationships = regulation.get('relationships', {})
            clean_reg['relationships'] = relationships

            # Non-document metadata (URLs, paths, etc.)
            clean_reg['detail_url'] = regulation.get('detail_url', regulation.get('source_url'))
            clean_reg['source_url'] = regulation.get('source_url', regulation.get('detail_url'))
            clean_reg['pdf_url'] = regulation.get('pdf_url', None)
            clean_reg['uji_materi_pdf_url'] = regulation.get('uji_materi_pdf_url', None)
            clean_reg['uji_materi'] = regulation.get('uji_materi', [])
            clean_reg['pdf_path'] = regulation.get('pdf_path', None)
            clean_reg['text_path'] = regulation.get('text_path', None)
            clean_reg['document_tree'] = regulation.get('document_tree', None)

            clean_reg['last_updated'] = datetime.now().isoformat()

            return clean_reg

        except Exception as e:
            logging.error(f"Error cleaning regulation data: {str(e)}")
            return regulation

    def _normalize_relationship_field_name(self, label: str) -> str:
        """Convert Indonesian relationship labels to normalized field names"""
        if not label:
            return "unknown_relationship"

        # Convert to lowercase and remove common punctuation
        normalized = label.lower().strip()
        normalized = normalized.replace(':', '').replace('dengan', 'dengan').strip()

        # Replace spaces with underscores and ensure valid field name
        normalized = normalized.replace(' ', '_').replace('-', '_')

        # Remove any remaining special characters except underscores
        import re
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)

        # Ensure it starts with a letter
        if normalized and not normalized[0].isalpha():
            normalized = 'rel_' + normalized

        return normalized or "relationship"

    def _normalize_date_to_iso(self, date_text: str) -> Optional[str]:
        """Convert Indonesian date format to ISO format (YYYY-MM-DD)"""
        if not date_text or not isinstance(date_text, str):
            return None

        try:
            import re
            from datetime import datetime

            # Indonesian month mapping
            month_map = {
                'januari': '01', 'februari': '02', 'maret': '03', 'april': '04',
                'mei': '05', 'juni': '06', 'juli': '07', 'agustus': '08',
                'september': '09', 'oktober': '10', 'november': '11', 'desember': '12'
            }

            # Clean the date text
            date_clean = date_text.lower().strip()

            # Pattern: "14 Maret 1981" or "14 maret 1981"
            pattern = r'(\d{1,2})\s+(\w+)\s+(\d{4})'
            match = re.search(pattern, date_clean)

            if match:
                day = match.group(1).zfill(2)
                month_name = match.group(2)
                year = match.group(3)

                if month_name in month_map:
                    month = month_map[month_name]
                    iso_date = f"{year}-{month}-{day}"

                    # Validate the date
                    datetime.strptime(iso_date, '%Y-%m-%d')
                    return iso_date

            # Pattern: "YYYY-MM-DD" (already ISO)
            iso_pattern = r'(\d{4})-(\d{2})-(\d{2})'
            if re.match(iso_pattern, date_clean):
                return date_clean

            logging.debug(f"Could not parse date: {date_text}")
            return None

        except Exception as e:
            logging.debug(f"Error normalizing date '{date_text}': {str(e)}")
            return None

    def _generate_doc_id(self, regulation: Dict) -> str:
        """Generate deterministic doc_id in format: {type_short}-{year}-{number}"""
        try:
            # Get regulation type, year, and number
            reg_type = regulation.get('doc_type', '')
            number = regulation.get('doc_number', 'UNKNOWN')
            year = regulation.get('doc_year', 'UNKNOWN')

            # Extract type abbreviation
            type_short = self._extract_type_abbreviation(reg_type)

            # Clean number (remove any non-alphanumeric except dash)
            import re
            clean_number = re.sub(r'[^a-zA-Z0-9-]', '', str(number))

            # Generate doc_id
            doc_id = f"{type_short}-{year}-{clean_number}"

            logging.debug(f"Generated doc_id: {doc_id} for {regulation.get('title', 'Unknown')}")
            return doc_id

        except Exception as e:
            logging.debug(f"Error generating doc_id: {str(e)}")
            return f"UNKNOWN-{regulation.get('year', 'UNKNOWN')}-{regulation.get('number', 'UNKNOWN')}"

    def _extract_type_abbreviation(self, reg_type: str) -> str:
        """Extract type abbreviation from regulation type"""
        if not reg_type:
            return "UNKNOWN"

        # Common Indonesian regulation type mappings
        type_mappings = {
            'undang-undang': 'UU',
            'undang undang': 'UU',
            'uu': 'UU',
            'peraturan pemerintah': 'PP',
            'pp': 'PP',
            'peraturan otoritas jasa keuangan': 'POJK',
            'surat edaran otoritas jasa keuangan': 'SEOJK',
            'surat edaran': 'SE',
            'peraturan presiden': 'PERPRES',
            'perpres': 'PERPRES',
            'peraturan menteri': 'PERMEN',
            'permen': 'PERMEN',
            'keputusan presiden': 'KEPPRES',
            'keppres': 'KEPPRES',
            'keputusan menteri': 'KEPMEN',
            'kepmen': 'KEPMEN',
            'peraturan daerah': 'PERDA',
            'perda': 'PERDA',
            'peraturan pemerintah pengganti undang-undang': 'PERPU',
            'perpu': 'PERPU',
            'keputusan': 'KEP',
            'peraturan': 'PER'
        }

        reg_type_lower = reg_type.lower().strip()

        # Try exact matches first
        if reg_type_lower in type_mappings:
            return type_mappings[reg_type_lower]

        # Try partial matches
        for key, value in type_mappings.items():
            if key in reg_type_lower:
                return value

        # Extract first letters as fallback
        words = reg_type_lower.split()
        if words:
            abbreviation = ''.join(word[0].upper() for word in words if word)
            return abbreviation[:10]  # Max 10 characters

        return "UNKNOWN"
