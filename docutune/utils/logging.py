"""Application logging helpers.

Rules (see README / docs):
- never log HF tokens, API keys or credentials
- never log full resume text by default; log lengths and IDs instead
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int | str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # Third-party libraries are noisy at INFO.
    for noisy in ("urllib3", "httpx", "httpcore", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str, level: int | str | None = None) -> logging.Logger:
    configure_logging(level)
    return logging.getLogger(name)


def text_summary(text: str, limit: int = 60) -> str:
    """A short, non-identifying preview safe for logs."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
