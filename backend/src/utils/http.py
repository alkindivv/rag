from __future__ import annotations
"""HTTP client wrapper with retry and dependency injection."""

import time
from typing import Any, Dict, Optional

import httpx

from src.utils.logging import get_logger


logger = get_logger(__name__)


class HttpClient:
    """Wrapper around httpx with simple retry/backoff."""

    def __init__(self, client: Optional[httpx.Client] = None) -> None:
        self._client = client or httpx.Client(timeout=10)

    def post_json(
        self, url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None, retries: int = 5
    ) -> Dict[str, Any]:
        """POST JSON with retry and explicit JSON error handling."""

        for attempt in range(1, retries + 1):
            try:
                resp = self._client.post(url, json=data, headers=headers)
                resp.raise_for_status()
                try:
                    return resp.json()
                except ValueError as exc:  # pragma: no cover - simple
                    raise RuntimeError("invalid json response") from exc
            except Exception as exc:  # pragma: no cover - simple
                if attempt == retries:
                    logger.error("http_post_failed", url=url)
                    raise
                sleep = 2 ** (attempt - 1)
                time.sleep(sleep)
        raise RuntimeError("unreachable")
