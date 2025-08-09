"""Structured logging utility for Legal RAG system."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, Optional

from ..config.settings import settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""

        # Base log data
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "getMessage"
            }
        }

        if extra_fields:
            log_data["extra"] = extra_fields

        return json.dumps(log_data, ensure_ascii=False, default=str)


def setup_logging(
    level: Optional[str] = None,
    structured: bool = True,
    include_console: bool = True
) -> None:
    """
    Set up application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Whether to use structured JSON logging
        include_console: Whether to include console output
    """
    log_level = level or settings.log_level

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root logger level
    root_logger.setLevel(numeric_level)

    if include_console:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            # Human-readable format for development
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)

        root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with consistent configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class RequestLogger:
    """Context manager for request-scoped logging."""

    def __init__(self, request_id: str, logger: logging.Logger):
        """
        Initialize request logger.

        Args:
            request_id: Unique request identifier
            logger: Base logger instance
        """
        self.request_id = request_id
        self.logger = logger
        self._old_factory = logging.getLogRecordFactory()

    def __enter__(self) -> logging.Logger:
        """Enter context and set up request-scoped logging."""
        def record_factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            record.request_id = self.request_id
            return record

        logging.setLogRecordFactory(record_factory)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore normal logging."""
        logging.setLogRecordFactory(self._old_factory)


def log_function_call(func_name: str, **kwargs) -> Dict[str, Any]:
    """
    Create structured log data for function calls.

    Args:
        func_name: Name of the function being called
        **kwargs: Function arguments to log

    Returns:
        Dictionary of log data
    """
    return {
        "event": "function_call",
        "function": func_name,
        "arguments": {k: v for k, v in kwargs.items() if not k.startswith('_')}
    }


def log_timing(operation: str, duration_ms: float, **context) -> Dict[str, Any]:
    """
    Create structured log data for timing information.

    Args:
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        **context: Additional context

    Returns:
        Dictionary of log data
    """
    return {
        "event": "timing",
        "operation": operation,
        "duration_ms": duration_ms,
        **context
    }


def log_error(error: Exception, **context) -> Dict[str, Any]:
    """
    Create structured log data for errors.

    Args:
        error: Exception instance
        **context: Additional context

    Returns:
        Dictionary of log data
    """
    return {
        "event": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        **context
    }


def log_api_request(
    method: str,
    url: str,
    status_code: Optional[int] = None,
    duration_ms: Optional[float] = None,
    **context
) -> Dict[str, Any]:
    """
    Create structured log data for API requests.

    Args:
        method: HTTP method
        url: Request URL
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        **context: Additional context

    Returns:
        Dictionary of log data
    """
    log_data = {
        "event": "api_request",
        "method": method,
        "url": url,
        **context
    }

    if status_code is not None:
        log_data["status_code"] = status_code

    if duration_ms is not None:
        log_data["duration_ms"] = duration_ms

    return log_data


# Initialize logging on import
setup_logging()
