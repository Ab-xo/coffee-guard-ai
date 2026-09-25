"""Logging setup shared by the CLI, training and the API."""

from __future__ import annotations

import logging

_CONFIGURED = False


def setup_logging(level: str | int = "INFO") -> None:
    """Configure the root logger once. Uses rich formatting when available."""
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
        fmt = "%(message)s"
    except ImportError:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", handlers=[handler])
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
