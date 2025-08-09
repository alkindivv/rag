#!/usr/bin/env python3
"""
FileDownloader - PDF downloading and file management
Uses centralized configuration - no hardcoded values
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional

import aiohttp

from src.config.crawler_config import CrawlerConfig
from .utils import retry_async


logger = logging.getLogger(__name__)


class FileDownloader:
    """Handle PDF downloading and file management with retry mechanism."""

    def __init__(self) -> None:
        """Initialize with centralized configuration."""
        self.config = CrawlerConfig

    async def download_pdf(
        self,
        pdf_url: str,
        pdf_dir: str,
        filename: str,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Optional[str]:
        """Download PDF file with duplicate checking and retry mechanism."""
        try:
            os.makedirs(pdf_dir, exist_ok=True)
            file_path = os.path.join(pdf_dir, filename)

            if os.path.exists(file_path):
                logger.info("BPK PDF file already exists, skipping download: %s", file_path)
                return file_path

            retry_cfg = self.config.get_retry_config()
            download_cfg = self.config.get_download_config()

            session_to_use = session
            close_session = False
            if not session_to_use or session_to_use.closed:
                conn_cfg = self.config.get_connection_config()
                timeout_cfg = self.config.get_download_timeout_config()
                connector = aiohttp.TCPConnector(**conn_cfg)
                session_to_use = aiohttp.ClientSession(
                    headers=self.config.get_http_headers(),
                    timeout=aiohttp.ClientTimeout(**timeout_cfg),
                    connector=connector,
                )
                close_session = True

            async def _attempt() -> str:
                logger.info("Downloading BPK PDF: %s", pdf_url)
                await asyncio.sleep(download_cfg["sleep_between_requests"])
                pdf_headers = self.config.get_pdf_headers(self.config.get_base_url())
                async with session_to_use.get(pdf_url, headers=pdf_headers) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}")
                    content_length = response.headers.get("Content-Length")
                    total_size = int(content_length) if content_length else 0
                    if total_size:
                        logger.info("PDF size: %.1f MB", total_size / (1024 * 1024))

                    downloaded_size = 0
                    chunk_size = download_cfg["chunk_size"]
                    pdf_chunks: List[bytes] = []
                    async for chunk in response.content.iter_chunked(chunk_size):
                        pdf_chunks.append(chunk)
                        downloaded_size += len(chunk)

                    if total_size:
                        logger.info(
                            "Downloaded %.1f MB / %.1f MB",
                            downloaded_size / (1024 * 1024),
                            total_size / (1024 * 1024),
                        )
                    else:
                        logger.info("Downloaded %.1f MB", downloaded_size / (1024 * 1024))

                    content = b"".join(pdf_chunks)
                    if not content.startswith(b"%PDF"):
                        raise RuntimeError("Not a valid PDF")

                    with open(file_path, "wb") as f:
                        f.write(content)
                    file_size = len(content) / (1024 * 1024)
                    logger.info(
                        "Downloaded BPK PDF successfully: %s (%.1f MB)",
                        file_path,
                        file_size,
                    )
                    return file_path

            try:
                return await retry_async(
                    _attempt,
                    retries=retry_cfg["download_retries"],
                    delay=retry_cfg["download_delay"],
                    logger=logger,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Error downloading BPK PDF: %s", exc)
                return None
            finally:
                if close_session and session_to_use:
                    await session_to_use.close()

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error in BPK PDF download: %s", exc)
            return None

    async def download_pdf_simple(
        self,
        pdf_url: str,
        regulation: dict,
        pdf_dir: Path,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Optional[Path]:
        """Download PDF file from URL - simplified version."""
        try:
            filename = self.config.generate_pdf_filename(regulation)
            pdf_path = pdf_dir / filename

            if pdf_path.exists():
                logger.info("PDF already exists: %s", pdf_path)
                return pdf_path

            logger.info("Downloading PDF: %s", pdf_url)

            session_to_use = session
            close_session = False
            if not session_to_use or session_to_use.closed:
                session_to_use = aiohttp.ClientSession(headers=self.config.get_http_headers())
                close_session = True

            try:
                async with session_to_use.get(pdf_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        with open(pdf_path, "wb") as f:
                            f.write(content)
                        logger.info("PDF downloaded successfully: %s", pdf_path)
                        return pdf_path
                    logger.error("Failed to download PDF: HTTP %s", response.status)
                    return None
            finally:
                if close_session:
                    await session_to_use.close()

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error downloading PDF from %s: %s", pdf_url, exc)
            return None
