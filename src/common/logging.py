"""
src/common/logging.py
Structured logging setup for the entire project.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: str | Path | None = None) -> logging.Logger:
    """Configure root logger with console + optional file handler."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "tegf_xau.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(fh)

    logging.basicConfig(level=numeric_level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)
    logger = logging.getLogger("tegf_xau")
    logger.setLevel(numeric_level)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"tegf_xau.{name}")
