"""HTTP client utility with retry/backoff and dependency injection support."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .logging import get_logger, log_api_request, log_error

logger = get_logger(__name__)


class HttpClient:
    """
    HTTP client with retry logic and structured logging.

    Supports both sync and async operations with automatic retries,
    exponential backoff, and comprehensive error handling.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        client: Optional[httpx.Client] = None,
    ):
        """
        Initialize HTTP client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
            client: Optional httpx.Client instance for dependency injection
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

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

    @retry(
        stop=stop_after_attempt(3),  # Will be overridden by instance config
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.HTTPStatusError,
        )),
        reraise=True,
    )
    def _make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method
            url: Request URL
            headers: Optional headers
            **kwargs: Additional request parameters

        Returns:
            httpx.Response object

        Raises:
            httpx.HTTPError: For HTTP-related errors
        """
        start_time = time.time()

        try:
            # Update retry configuration dynamically
            self._make_request.retry.stop = stop_after_attempt(self.max_retries)
            self._make_request.retry.wait = wait_exponential(
                multiplier=self.base_delay,
                min=self.base_delay,
                max=self.max_delay
            )

            logger.debug(f"Making {method} request to {url}")

            response = self._client.request(
                method=method,
                url=url,
                headers=headers,
                **kwargs
            )

            duration_ms = (time.time() - start_time) * 1000

            # Log successful request
            logger.info(
                "HTTP request completed",
                extra=log_api_request(
                    method=method,
                    url=url,
                    status_code=response.status_code,
                    duration_ms=duration_ms
                )
            )

            # Raise for HTTP error status codes
            response.raise_for_status()

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Log failed request
            logger.error(
                f"HTTP request failed: {e}",
                extra={
                    **log_api_request(
                        method=method,
                        url=url,
                        duration_ms=duration_ms
                    ),
                    **log_error(e)
                }
            )
            raise

    def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> httpx.Response:
        """Make GET request."""
        return self._make_request("GET", url, headers=headers, params=params, **kwargs)

    def post(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> httpx.Response:
        """Make POST request."""
        request_kwargs = kwargs.copy()

        if json_data is not None:
            request_kwargs["json"] = json_data
        elif data is not None:
            request_kwargs["data"] = data

        return self._make_request("POST", url, headers=headers, **request_kwargs)

    def put(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> httpx.Response:
        """Make PUT request."""
        request_kwargs = kwargs.copy()

        if json_data is not None:
            request_kwargs["json"] = json_data
        elif data is not None:
            request_kwargs["data"] = data

        return self._make_request("PUT", url, headers=headers, **request_kwargs)

    def delete(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> httpx.Response:
        """Make DELETE request."""
        return self._make_request("DELETE", url, headers=headers, **kwargs)

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
            json.JSONDecodeError: If response is not valid JSON
        """
        response = self.get(url, headers=headers, **kwargs)
        try:
            return response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise

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
            json.JSONDecodeError: If response is not valid JSON
        """
        response = self.post(url, headers=headers, json_data=data, **kwargs)
        try:
            return response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise

    def download_file(
        self,
        url: str,
        file_path: str,
        headers: Optional[Dict[str, str]] = None,
        chunk_size: int = 8192,
        **kwargs
    ) -> None:
        """
        Download file from URL.

        Args:
            url: File URL
            file_path: Local file path to save
            headers: Optional headers
            chunk_size: Download chunk size in bytes
            **kwargs: Additional request parameters
        """
        with self._client.stream("GET", url, headers=headers, **kwargs) as response:
            response.raise_for_status()

            with open(file_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    f.write(chunk)

        logger.info(f"Downloaded file: {url} -> {file_path}")


class AsyncHttpClient:
    """
    Async HTTP client with retry logic and structured logging.

    Similar to HttpClient but for async operations.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initialize async HTTP client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
            client: Optional httpx.AsyncClient instance for dependency injection
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        # Use provided client or create new one
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )

        # Track if we own the client (for cleanup)
        self._owns_client = client is None

    async def __aenter__(self) -> AsyncHttpClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        await self.close()

    async def close(self) -> None:
        """Close the async HTTP client if we own it."""
        if self._owns_client and self._client:
            await self._client.aclose()

    async def _make_request_async(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> httpx.Response:
        """
        Make async HTTP request with retry logic.

        Args:
            method: HTTP method
            url: Request URL
            headers: Optional headers
            **kwargs: Additional request parameters

        Returns:
            httpx.Response object
        """
        for attempt in range(self.max_retries + 1):
            start_time = time.time()

            try:
                logger.debug(f"Making async {method} request to {url} (attempt {attempt + 1})")

                response = await self._client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    **kwargs
                )

                duration_ms = (time.time() - start_time) * 1000

                # Log successful request
                logger.info(
                    "Async HTTP request completed",
                    extra=log_api_request(
                        method=method,
                        url=url,
                        status_code=response.status_code,
                        duration_ms=duration_ms,
                        attempt=attempt + 1
                    )
                )

                # Raise for HTTP error status codes
                response.raise_for_status()

                return response

            except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
                duration_ms = (time.time() - start_time) * 1000

                # Log failed attempt
                logger.warning(
                    f"Async HTTP request attempt {attempt + 1} failed: {e}",
                    extra={
                        **log_api_request(
                            method=method,
                            url=url,
                            duration_ms=duration_ms,
                            attempt=attempt + 1
                        ),
                        **log_error(e)
                    }
                )

                # If this is the last attempt, raise the exception
                if attempt == self.max_retries:
                    raise

                # Wait before retry with exponential backoff
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                await asyncio.sleep(delay)

            except Exception as e:
                # For non-retryable exceptions, fail immediately
                duration_ms = (time.time() - start_time) * 1000

                logger.error(
                    f"Async HTTP request failed with non-retryable error: {e}",
                    extra={
                        **log_api_request(
                            method=method,
                            url=url,
                            duration_ms=duration_ms,
                            attempt=attempt + 1
                        ),
                        **log_error(e)
                    }
                )
                raise

    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> httpx.Response:
        """Make async GET request."""
        return await self._make_request_async("GET", url, headers=headers, params=params, **kwargs)

    async def post(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> httpx.Response:
        """Make async POST request."""
        request_kwargs = kwargs.copy()

        if json_data is not None:
            request_kwargs["json"] = json_data
        elif data is not None:
            request_kwargs["data"] = data

        return await self._make_request_async("POST", url, headers=headers, **request_kwargs)

    async def get_json(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make async GET request and return JSON response.

        Args:
            url: Request URL
            headers: Optional headers
            **kwargs: Additional request parameters

        Returns:
            Parsed JSON response
        """
        response = await self.get(url, headers=headers, **kwargs)
        try:
            return response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise

    async def post_json(
        self,
        url: str,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make async POST request with JSON data and return JSON response.

        Args:
            url: Request URL
            data: JSON data to send
            headers: Optional headers
            **kwargs: Additional request parameters

        Returns:
            Parsed JSON response
        """
        response = await self.post(url, headers=headers, json_data=data, **kwargs)
        try:
            return response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise


def create_http_client(
    timeout: float = 30.0,
    max_retries: int = 3,
    **kwargs
) -> HttpClient:
    """
    Factory function to create HTTP client with default configuration.

    Args:
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        **kwargs: Additional client configuration

    Returns:
        Configured HttpClient instance
    """
    return HttpClient(timeout=timeout, max_retries=max_retries, **kwargs)


def create_async_http_client(
    timeout: float = 30.0,
    max_retries: int = 3,
    **kwargs
) -> AsyncHttpClient:
    """
    Factory function to create async HTTP client with default configuration.

    Args:
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        **kwargs: Additional client configuration

    Returns:
        Configured AsyncHttpClient instance
    """
    return AsyncHttpClient(timeout=timeout, max_retries=max_retries, **kwargs)
