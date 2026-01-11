"""Logging and monitoring configuration."""

import logging
from typing import Any

import structlog
from rich.logging import RichHandler

from papers_qa.config import get_settings


def configure_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    log_level = getattr(logging, settings.log_level)

    # Console handler with rich formatting
    console_handler = RichHandler(rich_tracebacks=True, markup=True)
    console_handler.setLevel(log_level)

    handlers = [console_handler]

    # File handler if specified
    if settings.log_file:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in handlers:
        root_logger.addHandler(handler)

    # Suppress verbose libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        structlog.typing.FilteringBoundLogger: Configured logger instance.
    """
    return structlog.get_logger(name)


class PerformanceTracker:
    """Track performance metrics for operations."""

    def __init__(self) -> None:
        """Initialize the performance tracker."""
        self.metrics: dict[str, Any] = {}
        self.logger = get_logger(__name__)

    def record(self, operation: str, duration_seconds: float, metadata: dict | None = None) -> None:
        """Record a performance metric.

        Args:
            operation: Name of the operation.
            duration_seconds: Duration in seconds.
            metadata: Optional metadata about the operation.
        """
        if operation not in self.metrics:
            self.metrics[operation] = []

        self.metrics[operation].append(
            {
                "duration": duration_seconds,
                "metadata": metadata or {},
            }
        )

        self.logger.info(
            "operation_completed",
            operation=operation,
            duration_seconds=round(duration_seconds, 3),
            metadata=metadata,
        )

    def get_summary(self, operation: str) -> dict[str, float] | None:
        """Get performance summary for an operation.

        Args:
            operation: Name of the operation.

        Returns:
            dict or None: Summary statistics if operation exists.
        """
        if operation not in self.metrics:
            return None

        records = self.metrics[operation]
        durations = [r["duration"] for r in records]

        return {
            "count": len(durations),
            "min": min(durations),
            "max": max(durations),
            "avg": sum(durations) / len(durations),
        }
