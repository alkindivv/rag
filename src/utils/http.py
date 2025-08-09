"""Simplified HTTP client with smart retry logic and proper error handling."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx

from .logging import get_logger

logger = get_logger(__name__)


class HttpError(Exception):
    """Base HTTP error."""
    pass


class AuthError(HttpError):
    """Authentication/authorization error (4xx)."""
    pass


class ServerError(HttpError):
    """Server error (5xx)."""
    pass


class NetworkError(HttpError):
    """Network/timeout error."""
    pass


class HttpClient:
    """
    Simplified HTTP client with smart retry logic.

    Key features:
    - Only retries network/timeout errors (not auth errors)
    - Automatic Accept: application/json headers
    - Simple exponential backoff
    - Clear error classification
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        client: Optional[httpx.Client] = None,
    ):
        """
        Initialize HTTP client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for network errors only
            client: Optional httpx.Client for dependency injection
        """
        self.timeout = timeout
        self.max_retries = max_retries

        # Use provided client or create new one
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )

        # Track if we own the client (for cleanup)
        self._owns_client = client is None

    def __enter__(self) -> HttpClient:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()

    def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client and self._client:
            self._client.close()

    def _should_retry(self, error: Exception) -> bool:
        """
        Determine if an error should be retried.

        Only retry network/timeout errors, not auth (4xx) or client errors.
        """
        if isinstance(error, httpx.TimeoutException):
            return True
        if isinstance(error, (httpx.ConnectError, httpx.NetworkError)):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            # Only retry 5xx server errors, not 4xx client errors
            return error.response.status_code >= 500
        return False

    def _classify_error(self, error: Exception) -> HttpError:
        """Classify httpx errors into our error types."""
        if isinstance(error, httpx.TimeoutException):
            return NetworkError(f"Request timed out after {self.timeout}s")
        if isinstance(error, (httpx.ConnectError, httpx.NetworkError)):
            return NetworkError(f"Network error: {error}")
        if isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code
            if 400 <= status < 500:
                return AuthError(f"Client error '{status} {error.response.reason_phrase}' for url '{error.request.url}'")
            elif status >= 500:
                return ServerError(f"Server error '{status} {error.response.reason_phrase}' for url '{error.request.url}'")
        return HttpError(f"HTTP error: {error}")

    def _make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request with smart retry logic.

        Args:
            method: HTTP method
            url: Request URL
            headers: Optional headers
            **kwargs: Additional request parameters

        Returns:
            httpx.Response object

        Raises:
            HttpError: Classified HTTP error
        """
        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"HTTP {method} {url} (attempt {attempt + 1}/{self.max_retries + 1})")

                response = self._client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    **kwargs
                )

                duration_ms = (time.time() - start_time) * 1000

                # Log successful request (only debug level to reduce noise)
                logger.debug(
                    f"HTTP {method} {url} → {response.status_code} ({duration_ms:.1f}ms)"
                )

                # Check for HTTP errors
                response.raise_for_status()

                return response

            except Exception as e:
                last_error = e
                classified_error = self._classify_error(e)

                # Don't retry auth errors or client errors
                if not self._should_retry(e):
                    duration_ms = (time.time() - start_time) * 1000

                    # Only log auth errors at debug level to reduce noise
                    if isinstance(classified_error, AuthError):
                        logger.debug(f"HTTP {method} {url} → auth error: {classified_error}")
                    else:
                        logger.error(f"HTTP {method} {url} → {classified_error}")

                    raise classified_error

                # For retryable errors, wait before next attempt
                if attempt < self.max_retries:
                    wait_time = min(2 ** attempt, 10)  # Simple exponential backoff, max 10s
                    logger.debug(f"Retrying in {wait_time}s due to: {classified_error}")
                    time.sleep(wait_time)

        # All retries exhausted
        duration_ms = (time.time() - start_time) * 1000
        final_error = self._classify_error(last_error)
        logger.error(f"HTTP {method} {url} → failed after {self.max_retries + 1} attempts: {final_error}")
        raise final_error

    def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> httpx.Response:
        """Make GET request."""
        return self._make_request("GET", url, headers=headers, **kwargs)

    def post(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> httpx.Response:
        """Make POST request."""
        if json_data is not None:
            kwargs["json"] = json_data
        return self._make_request("POST", url, headers=headers, **kwargs)

    def get_json(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make GET request and return JSON response.

        Args:
            url: Request URL
            headers: Optional headers
            **kwargs: Additional request parameters

        Returns:
            Parsed JSON response

        Raises:
            HttpError: If request fails
            ValueError: If response is not valid JSON
        """
        # Ensure Accept: application/json header
        merged_headers = {"Accept": "application/json"}
        if headers:
            merged_headers.update(headers)

        response = self.get(url, headers=merged_headers, **kwargs)

        try:
            return response.json()
        except Exception as e:
            raise ValueError(f"Failed to decode JSON response: {e}")

    def post_json(
        self,
        url: str,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make POST request with JSON data and return JSON response.

        Args:
            url: Request URL
            data: JSON data to send
            headers: Optional headers
            **kwargs: Additional request parameters

        Returns:
            Parsed JSON response

        Raises:
            HttpError: If request fails
            ValueError: If response is not valid JSON
        """
        # Ensure Accept: application/json header
        merged_headers = {"Accept": "application/json"}
        if headers:
            merged_headers.update(headers)

        response = self.post(url, headers=merged_headers, json_data=data, **kwargs)

        try:
            return response.json()
        except Exception as e:
            raise ValueError(f"Failed to decode JSON response: {e}")


# Factory functions for easy instantiation
def create_http_client(
    timeout: float = 30.0,
    max_retries: int = 3,
) -> HttpClient:
    """Create HTTP client with default configuration."""
    return HttpClient(timeout=timeout, max_retries=max_retries)


def create_jina_client() -> HttpClient:
    """Create HTTP client optimized for Jina API calls."""
    return HttpClient(
        timeout=60.0,  # Jina can be slow
        max_retries=2,  # Less aggressive retries
    )
