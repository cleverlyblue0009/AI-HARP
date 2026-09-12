"""Consistent logging. The mobility backend in use must always be obvious."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_DATEFMT = "%H:%M:%S"


def _configure(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger("aiharp")
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger under the ``aiharp`` namespace."""
    _configure(level)
    return logging.getLogger(f"aiharp.{name}")


def log_banner(logger: logging.Logger, title: str, lines: list[str]) -> None:
    """Log a boxed banner. Used to make backend selection impossible to miss."""
    width = max([len(title)] + [len(x) for x in lines]) + 4
    logger.info("+" + "-" * width + "+")
    logger.info("| %s%s |", title, " " * (width - len(title) - 2))
    if lines:
        logger.info("+" + "-" * width + "+")
    for line in lines:
        logger.info("| %s%s |", line, " " * (width - len(line) - 2))
    logger.info("+" + "-" * width + "+")
