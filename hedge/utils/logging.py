"""
Structured logging setup for the entire application.

Usage:
    from hedge.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started", extra={"run_id": run_id})
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from hedge.utils.config import get_settings

_INITIALISED = False


def _setup_root_logger() -> None:
    global _INITIALISED
    if _INITIALISED:
        return

    cfg = get_settings()
    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    log_dir = Path(cfg.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)

    # Rotating file handler.
    fh = RotatingFileHandler(
        log_dir / "hedge.log",
        maxBytes=cfg.logging.rotate_mb * 1_048_576,
        backupCount=cfg.logging.backup_count,
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)

    root = logging.getLogger("hedge")
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(fh)

    _INITIALISED = True


def get_logger(name: str) -> logging.Logger:
    _setup_root_logger()
    return logging.getLogger(name)
