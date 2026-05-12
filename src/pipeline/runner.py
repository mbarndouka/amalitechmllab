"""Pipeline runner — maps stage names to their run() functions and executes them."""
from __future__ import annotations

import time
from typing import Any, Callable

from features import exploration
from utils.logging import get_logger

logger = get_logger(__name__)

STAGES: dict[str, Callable[[dict[str, Any]], None]] = {
    "explore": exploration.run,
}


def run_stage(name: str, cfg: dict[str, Any]) -> None:
    logger.info(">>> Starting stage: %s", name)
    start = time.perf_counter()
    STAGES[name](cfg)
    elapsed = time.perf_counter() - start
    logger.info(">>> Stage '%s' completed in %.2fs", name, elapsed)


def run_all(cfg: dict[str, Any]) -> None:
    logger.info("Running all %d stage(s): %s", len(STAGES), list(STAGES.keys()))
    for name in STAGES:
        run_stage(name, cfg)
    logger.info("All stages complete.")
