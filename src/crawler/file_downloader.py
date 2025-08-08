#!/usr/bin/env python3
"""
FileDownloader - PDF downloading and file management
Uses centralized configuration - no hardcoded values
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import aiohttp

from src.config.crawler_config import CrawlerConfig


class FileDownloader:
    """Handle PDF downloading and file management with retry mechanism"""

    def __init__(self):
        """Initialize with centralized configuration"""
        self.config = CrawlerConfig

    async def download_pdf(self, pdf_url: str, pdf_dir: str, filename: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[str]:
        """Download PDF file with duplicate checking and retry mechanism"""
        try:
            os.makedirs(pdf_dir, exist_ok=True)

            file_path = os.path.join(pdf_dir, filename)

            if os.path.exists(file_path):
                logging.info(f"BPK PDF file already exists, skipping download: {file_path}")
                return file_path

            retry_config = self.config.get_retry_config()
            download_config = self.config.get_download_config()
            max_retries = retry_config['download_retries']
            retry_delay = retry_config['download_delay']

            session_to_use = session
            close_session_after = False

            if not session_to_use or session_to_use.closed:
                connection_config = self.config.get_connection_config()
                download_timeout_config = self.config.get_download_timeout_config()

                connector = aiohttp.TCPConnector(**connection_config)
                session_to_use = aiohttp.ClientSession(
                    headers=self.config.get_http_headers(),
                    timeout=aiohttp.ClientTimeout(**download_timeout_config),
                    connector=connector
                )
                close_session_after = True

            try:
                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            await asyncio.sleep(retry_delay * attempt)
                            logging.info(f"Retrying BPK PDF download attempt {attempt + 1}/{max_retries}")

                        logging.info(f"Downloading BPK PDF: {pdf_url}")

                        await asyncio.sleep(download_config['sleep_between_requests'])

                        pdf_headers = self.config.get_pdf_headers(self.config.get_base_url())

                        async with session_to_use.get(pdf_url, headers=pdf_headers) as response:
                            if response.status == 200:
                                # Get content length for progress tracking
                                content_length = response.headers.get('Content-Length')
                                if content_length:
                                    total_size = int(content_length)
                                    logging.info(f"PDF size: {total_size / (1024*1024):.1f} MB")
                                else:
                                    total_size = 0

                                # Optimized streaming download
                                downloaded_size = 0
                                chunk_size = download_config['chunk_size']
                                pdf_chunks = []

                                try:
                                    async for chunk in response.content.iter_chunked(chunk_size):
                                        pdf_chunks.append(chunk)
                                        downloaded_size += len(chunk)

                                    # Single progress log at completion
                                    if total_size > 0:
                                        logging.info(f"Downloaded {downloaded_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
                                    else:
                                        logging.info(f"Downloaded {downloaded_size / (1024*1024):.1f} MB")

                                except Exception as chunk_error:
                                    logging.error(f"Error during streaming download: {chunk_error}")
                                    raise

                                # Combine chunks
                                content = b''.join(pdf_chunks)

                                if content.startswith(b'%PDF'):
                                    with open(file_path, 'wb') as f:
                                        f.write(content)

                                    file_size = len(content) / (1024 * 1024)  # MB
                                    logging.info(f"Downloaded BPK PDF successfully: {file_path} ({file_size:.1f} MB)")
                                    return file_path
                                else:
                                    logging.error(f"Downloaded content is not a valid PDF: {pdf_url}")
                                    return None
                            else:
                                logging.error(f"Failed to download BPK PDF: HTTP {response.status}")
                                if attempt == max_retries - 1:
                                    return None
                                continue

                    except Exception as e:
                        logging.error(f"Error downloading BPK PDF (attempt {attempt + 1}): {str(e)}")
                        if attempt == max_retries - 1:
                            return None
                        continue

                return None

            finally:
                if close_session_after and session_to_use:
                    await session_to_use.close()

        except Exception as e:
            logging.error(f"Error in BPK PDF download: {str(e)}")
            return None

    async def download_pdf_simple(self, pdf_url: str, regulation: dict, pdf_dir: Path) -> Optional[Path]:
        """Download PDF file from URL - simplified version"""
        try:
            filename = self.config.generate_pdf_filename(regulation)
            pdf_path = pdf_dir / filename

            if pdf_path.exists():
                logging.info(f"PDF already exists: {pdf_path}")
                return pdf_path

            logging.info(f"Downloading PDF: {pdf_url}")

            async with aiohttp.ClientSession(headers=self.config.get_http_headers()) as session:
                async with session.get(pdf_url) as response:
                    if response.status == 200:
                        content = await response.read()

                        with open(pdf_path, 'wb') as f:
                            f.write(content)

                        logging.info(f"PDF downloaded successfully: {pdf_path}")
                        return pdf_path
                    else:
                        logging.error(f"Failed to download PDF: HTTP {response.status}")
                        return None

        except Exception as e:
            logging.error(f"Error downloading PDF from {pdf_url}: {str(e)}")
            return None
