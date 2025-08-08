#!/usr/bin/env python3
"""
WebScraper - Web scraping, search, and data extraction
Uses centralized configuration - no hardcoded values or duplications
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup

from .metadata_extractor import MetadataExtractor
from src.config.crawler_config import CrawlerConfig


class WebScraper:
    """Crawler for BPK legal documents website using centralized configuration"""

    def __init__(self):
        """Initialize with centralized configuration"""
        self.config = CrawlerConfig
        self.session = None
        self.metadata_extractor = MetadataExtractor()

    @staticmethod
    def clean_regulation_text(text: str) -> str:
        """Clean regulation text by splitting into Pasal blocks and removing internal line breaks"""
        config = CrawlerConfig
        text_patterns = config.get_text_patterns()
        pasal_pattern = re.compile(text_patterns['pasal_pattern'])

        parts = []
        current_part = []

        lines = text.split('\n')

        for line in lines:
            stripped_line = line.strip()

            if pasal_pattern.match(line):
                if current_part:
                    cleaned_part = ' '.join(current_part).strip()
                    cleaned_part = re.sub(r'\s+', ' ', cleaned_part)
                    parts.append(cleaned_part)
                    current_part = []

            if stripped_line:
                current_part.append(stripped_line)

        if current_part:
            cleaned_part = ' '.join(current_part).strip()
            cleaned_part = re.sub(r'\s+', ' ', cleaned_part)
            parts.append(cleaned_part)

        return '\n\n'.join(parts)

    async def __aenter__(self):
        """Async context manager entry"""
        connection_config = self.config.get_connection_config()
        timeout_config = self.config.get_timeout_config()

        connector = aiohttp.TCPConnector(**connection_config)

        self.session = aiohttp.ClientSession(
            headers=self.config.get_http_headers(),
            timeout=aiohttp.ClientTimeout(**timeout_config),
            connector=connector
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def search_regulations(self,
                               query: str = "",
                               regulation_type: str = "",
                               year: str = "",
                               limit: int = 10) -> List[Dict]:
        """Search for regulations on BPK website"""
        logging.info(f"Searching BPK with query='{query}', type='{regulation_type}', year='{year}', limit={limit}")

        timeout_config = self.config.get_timeout_config()
        async with aiohttp.ClientSession(
            headers=self.config.get_http_headers(),
            timeout=aiohttp.ClientTimeout(total=timeout_config['connect'])
        ) as session:
            self.session = session
            return await self._search_regulations_internal(query, regulation_type, year, limit)

    async def _search_regulations_internal(self, query: str, regulation_type: str, year: str, limit: int) -> List[Dict]:
        """Internal search method with retry mechanism"""
        try:
            params = {
                'keywords': query,
                'tentang': '',
                'nomor': '',
                'jenis': self.config.get_regulation_type_id(regulation_type),
                'tahun': year,
                'page': 1
            }

            all_regulations = []
            page = 1
            download_config = self.config.get_download_config()

            while len(all_regulations) < limit:
                params['page'] = page
                logging.debug(f"Fetching BPK page {page} with params: {params}")

                if page > 1:
                    await asyncio.sleep(download_config['sleep_between_pages'])

                page_regulations = await self._fetch_page_with_retry(params, page)
                if not page_regulations:
                    logging.info("No more regulations found on this page")
                    break

                all_regulations.extend(page_regulations)

                if len(all_regulations) >= limit:
                    break

                page += 1

            limited_results = all_regulations[:limit]
            logging.info(f"Found {len(limited_results)} regulations total")

            return limited_results

        except Exception as e:
            logging.error(f"Error in BPK search: {str(e)}")
            return []

    async def _fetch_page_with_retry(self, params: dict, page: int) -> List[Dict]:
        """Fetch a single page with retry mechanism"""
        retry_config = self.config.get_retry_config()
        max_retries = retry_config['max_retries']
        retry_delay = retry_config['retry_delay']

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(retry_delay * attempt)
                    logging.info(f"Retrying BPK page {page} attempt {attempt + 1}/{max_retries}")

                async with self.session.get(self.config.get_search_url(), params=params) as response:
                    if response.status != 200:
                        logging.error(f"BPK search failed: HTTP {response.status} for page {page}")
                        if attempt == max_retries - 1:
                            return []
                        continue

                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    page_regulations = []
                    search_config = self.config.get_search_config()

                    detail_links = soup.find_all('a', href=re.compile(search_config['detail_links'], re.IGNORECASE))
                    if detail_links:
                        logging.debug(f"Found {len(detail_links)} regulation detail links")
                        regulation_items = detail_links
                    else:
                        all_rows = soup.find_all('tr')
                        filtered_rows = []
                        excluded_patterns = search_config['excluded_text_patterns']

                        for row in all_rows:
                            text = row.get_text(strip=True)
                            if not any(text.startswith(pattern) for pattern in excluded_patterns) and len(text.strip()) >= 10:
                                if (re.search(search_config['regulation_pattern'], text, re.IGNORECASE) or
                                    row.find('a', href=re.compile(r'Details', re.IGNORECASE))):
                                    filtered_rows.append(row)

                        if filtered_rows:
                            logging.debug(f"Found {len(filtered_rows)} filtered table rows with regulation content")
                            regulation_items = filtered_rows
                        else:
                            selectors = search_config['selectors']

                            for selector in selectors:
                                items = soup.select(selector)
                                if items:
                                    logging.debug(f"Found {len(items)} items with selector '{selector}'")
                                    regulation_items = items
                                    break
                            else:
                                regulation_items = []

                    for item in regulation_items:
                        try:
                            regulation = self._extract_regulation_from_item(item)
                            if regulation:
                                page_regulations.append(regulation)
                        except Exception as e:
                            logging.error(f"Error extracting regulation from item: {str(e)}")
                            continue

                    logging.debug(f"Found {len(page_regulations)} regulations on page {page}")
                    return page_regulations

            except Exception as e:
                logging.error(f"Error fetching BPK page {page} (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    return []
                continue

        return []

    def _extract_regulation_from_item(self, item) -> Optional[Dict]:
        """Extract regulation info from a single search result item"""
        try:
            item_text = item.get_text(strip=True)[:100]
            logging.debug(f"Processing item: {item_text}...")

            if item.name == 'a' and item.get('href'):
                title_link = item
            else:
                title_link = item.find('a', href=True)

                if not title_link:
                    title_link = item.find('a', href=re.compile(r'Details', re.IGNORECASE))

            if not title_link:
                logging.debug(f"No title link found in item: {item_text[:50]}")
                return None

            title = title_link.get_text(strip=True)
            href = title_link['href']

            if href.startswith('/'):
                detail_url = urljoin(self.config.get_base_url(), href)
            elif href.startswith('http'):
                detail_url = href
            else:
                detail_url = urljoin(self.config.get_base_url(), href)

            logging.debug(f"Extracted regulation: {title} -> {detail_url}")

            regulation = {
                'doc_title': title,
                'detail_url': detail_url,
                'source_url': detail_url,
            }

            self._parse_title_info(regulation)

            text_content = item.get_text()
            self._extract_metadata_from_text(regulation, text_content)

            return regulation

        except Exception as e:
            logging.debug(f"Error extracting regulation from item: {str(e)}")
            return None

    def _parse_title_info(self, regulation: Dict):
        """Parse regulation type, number, and year from title"""
        title = regulation.get('doc_title', '')

        patterns = [
            r'(UU|Undang-Undang)\s*(?:No\.?\s*|Nomor\s*)(\d+)\s*Tahun\s*(\d{4})',
            r'(PP|Peraturan\s+Pemerintah)\s*(?:No\.?\s*|Nomor\s*)(\d+)\s*Tahun\s*(\d{4})',
            r'(Permen|Peraturan\s+Menteri)\s*(?:No\.?\s*|Nomor\s*)(\d+)\s*Tahun\s*(\d{4})',
            r'(Perpres|Peraturan\s+Presiden)\s*(?:No\.?\s*|Nomor\s*)(\d+)\s*Tahun\s*(\d{4})',
            r'(Perda|Peraturan\s+Daerah)\s*(?:No\.?\s*|Nomor\s*)(\d+)\s*Tahun\s*(\d{4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                regulation['doc_type'] = match.group(1)
                regulation['doc_number'] = match.group(2)
                regulation['doc_year'] = match.group(3)
                break

    def _extract_metadata_from_text(self, regulation: Dict, text: str):
        """Extract metadata from text content"""
        text_patterns = self.config.get_text_patterns()

        if 'doc_year' not in regulation:
            year_match = re.search(text_patterns['year_pattern'], text)
            if year_match:
                regulation['doc_year'] = year_match.group(1)

        if 'berlaku' in text.lower():
            regulation['doc_status'] = 'Berlaku'
        elif 'dicabut' in text.lower():
            regulation['doc_status'] = 'Dicabut'

    async def get_regulation_details(self, detail_url: str) -> Dict:
        """Get detailed information about a regulation from its detail page"""
        logging.debug(f"Fetching details from: {detail_url}")

        timeout_config = self.config.get_timeout_config()
        async with aiohttp.ClientSession(
            headers=self.config.get_http_headers(),
            timeout=aiohttp.ClientTimeout(total=timeout_config['connect'])
        ) as session:
            self.session = session
            return await self._get_regulation_details_internal(detail_url)

    async def _get_regulation_details_internal(self, detail_url: str) -> Dict:
        """Internal method to get regulation details"""
        try:
            async with self.session.get(detail_url) as response:
                if response.status != 200:
                    logging.error(f"Failed to fetch details from {detail_url}: HTTP {response.status}")
                    return {}

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                details = {}

                self._extract_metadata_section(soup, details)
                self._extract_pdf_link(soup, details)

                try:
                    relationships = self.metadata_extractor.extract_relationships(soup)
                    if relationships:
                        # Store relationships in a dedicated section for dynamic processing
                        details['relationships'] = relationships
                        logging.debug(f"Extracted relationships for {detail_url}: {list(relationships.keys())}")
                    else:
                        logging.debug(f"No relationships found for {detail_url}")
                except Exception as e:
                    logging.error(f"Error extracting relationships from {detail_url}: {str(e)}")

                await asyncio.sleep(0.5)

                # Check if any relationships were found (dynamic approach)
                relationship_count = len(details.get('relationships', {}))
                if relationship_count > 0:
                    logging.debug(f"Successfully extracted {relationship_count} relationship types for {detail_url}")
                else:
                    logging.debug(f"No relationships extracted for {detail_url}")

                return details

        except Exception as e:
            logging.error(f"Error fetching regulation details from {detail_url}: {str(e)}")
            return {}

    def _extract_regulation_text(self, soup: BeautifulSoup) -> str:
        """Extract and clean regulation text from the detail page"""
        try:
            content = soup.find('div', {'class': 'content'})
            if not content:
                content = soup.find('div', {'class': 'container'})

            if content:
                text = content.get_text(separator='\n', strip=True)
                return self.clean_regulation_text(text)

            return ""
        except Exception as e:
            logging.error(f"Error extracting regulation text: {str(e)}")
            return ""

    def _extract_metadata_section(self, soup: BeautifulSoup, details: Dict):
        """Extract comprehensive metadata from the detail page using pure metadata extraction"""
        try:
            extracted_metadata = self.metadata_extractor.extract_metadata(soup)
            details.update(extracted_metadata)

            # Relationships are handled in _get_regulation_details_internal
            # No need to duplicate extraction here

            details['text'] = self._extract_regulation_text(soup)

            pdf_link = self._extract_pdf_link(soup, details)
            if pdf_link:
                details['pdf_url'] = pdf_link

            # Extract legacy UJI MATERI PDF URL for backward compatibility
            self._extract_uji_materi_pdf_legacy(soup, details)

            logging.info(f"Pure metadata extraction completed. Fields: {list(details.keys())}")

        except Exception as e:
            logging.error(f"Error extracting metadata: {str(e)}")

    def _extract_pdf_link(self, soup: BeautifulSoup, details: Dict):
        """Extract PDF download link - consolidated logic to avoid redundancy"""
        try:
            pdf_config = self.config.get_pdf_config()
            all_links = soup.find_all('a', href=True)

            # Identify UJI MATERI links to exclude them
            uji_materi_links = set()
            h4_tags = soup.find_all('h4')
            for h4 in h4_tags:
                if 'UJI MATERI' in h4.get_text(strip=True).upper():
                    uji_materi_section = h4.find_parent()
                    if uji_materi_section:
                        uji_materi_section_links = uji_materi_section.find_all('a', href=True)
                        for link in uji_materi_section_links:
                            href = link['href']
                            if any(pattern in href for pattern in pdf_config['download_patterns']):
                                uji_materi_links.add(href)

            # Priority 1: Look for PDF in FILE-FILE PERATURAN section
            file_section = None
            for h4 in h4_tags:
                h4_text = h4.get_text(strip=True).upper()
                if 'FILE' in h4_text and 'PERATURAN' in h4_text:
                    file_section = h4.find_next_sibling() or h4.find_parent()
                    break

            pdf_url = None
            extraction_source = ""

            if file_section:
                section_links = file_section.find_all('a', href=True)
                for link in section_links:
                    href = link['href']
                    text = link.get_text(strip=True).lower()

                    if href in uji_materi_links or any(pattern in href for pattern in pdf_config['download_patterns']):
                        continue

                    if self._is_valid_pdf_link(href, text, pdf_config):
                        pdf_url = self._normalize_pdf_url(href)
                        extraction_source = "FILE-FILE PERATURAN section"
                        break

            # Priority 2: General search if not found in FILE section
            if not pdf_url:
                for link in all_links:
                    href = link['href']
                    text = link.get_text(strip=True).lower()

                    if href in uji_materi_links or any(pattern in href for pattern in pdf_config['download_patterns']):
                        continue

                    if self._is_valid_pdf_link(href, text, pdf_config):
                        pdf_url = self._normalize_pdf_url(href)
                        extraction_source = "general search"
                        break

            # Priority 3: BPK download pattern as fallback
            if not pdf_url:
                download_links = soup.find_all('a', href=re.compile(r'^/Download/\d+/.*\.pdf$', re.IGNORECASE))
                for link in download_links:
                    href = link['href']
                    if href not in uji_materi_links and not any(pattern in href for pattern in pdf_config['download_patterns']):
                        pdf_url = self._normalize_pdf_url(href)
                        extraction_source = "BPK download pattern"
                        break

            # Set the PDF URL and log once
            if pdf_url:
                details['pdf_url'] = pdf_url
                logging.info(f"Found PDF URL via {extraction_source}: {pdf_url}")
            else:
                logging.warning("No valid PDF URL found")

        except Exception as e:
            logging.error(f"Error extracting PDF link: {str(e)}")

    def _is_valid_pdf_link(self, href: str, text: str, pdf_config: Dict) -> bool:
        """Check if a link is a valid PDF download link"""
        return ((any(indicator in href.lower() for indicator in pdf_config['pdf_indicators'])) or
                (any(indicator in text for indicator in pdf_config['pdf_indicators'])) or
                (href.startswith('/Download/') and '.pdf' in href))

    def _normalize_pdf_url(self, href: str) -> str:
        """Normalize PDF URL to full URL"""
        if not href.startswith('http'):
            return urljoin(self.config.get_base_url(), href)
        return href

    def _extract_uji_materi_pdf_legacy(self, soup: BeautifulSoup, details: Dict):
        """Extract PDF link from UJI MATERI section for backward compatibility"""
        try:
            pdf_config = self.config.get_pdf_config()
            all_links = soup.find_all('a', href=True)

            for link in all_links:
                href = link['href']
                text = link.get_text(strip=True)

                if (any(pattern in href for pattern in pdf_config['download_patterns']) or
                    ('PUU' in text and '.pdf' in href)):

                    pdf_url = href
                    if not pdf_url.startswith('http'):
                        pdf_url = urljoin(self.config.get_base_url(), pdf_url)

                    details['uji_materi_pdf_url'] = pdf_url
                    logging.info(f"Found UJI MATERI PDF URL: {pdf_url}")
                    return

            h4_tags = soup.find_all('h4')
            for h4 in h4_tags:
                h4_text = h4.get_text(strip=True)
                if any(pattern in h4_text for pattern in pdf_config['uji_materi_patterns']):
                    parent = h4.parent
                    if parent:
                        links = parent.find_all('a', href=True)
                        for link in links:
                            href = link['href']
                            if any(pattern in href for pattern in pdf_config['download_patterns']) or '.pdf' in href:
                                pdf_url = href
                                if not pdf_url.startswith('http'):
                                    pdf_url = urljoin(self.config.get_base_url(), pdf_url)

                                details['uji_materi_pdf_url'] = pdf_url
                                logging.info(f"Found UJI MATERI PDF URL: {pdf_url}")
                                return
                    break

            if 'uji_materi_pdf_url' not in details:
                details['uji_materi_pdf_url'] = None
                logging.debug("No UJI MATERI PDF found, field set to null")

        except Exception as e:
            logging.error(f"Error extracting UJI MATERI PDF link: {str(e)}")
            if 'uji_materi_pdf_url' not in details:
                details['uji_materi_pdf_url'] = None
