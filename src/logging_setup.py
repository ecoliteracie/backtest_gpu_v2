# src/logging_setup.py
from __future__ import annotations
import logging, sys
from pathlib import Path

_FMT = "[%(asctime)s] %(levelname)s %(name)s - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

def get_logger(name: str, log_path: str):
    """
    Create a logger that writes to stdout and to the given file.
    Format: [timestamp] LEVEL name - message
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if re-created
    if logger.handlers:
        logger.handlers.clear()

    file_h = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_h.setLevel(logging.INFO)
    file_h.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))

    console_h = logging.StreamHandler(sys.stdout)
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))

    logger.addHandler(file_h)
    logger.addHandler(console_h)
    return logger
