import asyncio
import logging
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    retries: int,
    delay: float,
    logger: logging.Logger,
) -> T:
    """Execute an async operation with simple exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return await operation()
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            logger.error("retry_failed", exc_info=exc)
            if attempt < retries:
                await asyncio.sleep(delay * attempt)
    if last_exc:
        raise last_exc
    raise RuntimeError("retry_async: no operation executed")
