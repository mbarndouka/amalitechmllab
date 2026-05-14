"""Centralized logging setup."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

logging = import_module("logging")


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        color = self.COLORS.get(record.levelno, "")
        record.levelname = f"{color}{original_levelname}{self.RESET}" if color else original_levelname
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def get_logger(name: str, level: int = logging.INFO, log_file: Path | None = None) -> logging.Logger:
    """
    Return a logger with a consistent format.

    Parameters
    ----------
    name:     Module name, typically ``__name__``.
    level:    Logging level (default INFO).
    log_file: Optional path to also write logs to a file.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    logger.setLevel(level)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(ColorFormatter(fmt, datefmt))
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(fh)
    return logger
