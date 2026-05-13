"""Pipeline runner — maps stage names to their run() functions and executes them."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from features import cleaning, eda, engineering, exploration
from interpretation import importance, insights
from models import advanced, trainer
from tuning import optuna_tuner
from utils.logging import get_logger

logger = get_logger(__name__)


def _run_interpret(cfg: dict[str, Any]) -> None:
    """Combine importance extraction + business insights into one pipeline stage.

    WHY ONE STAGE: Both sub-tasks read from artifacts produced by 'advanced' stage
    (models/ and data/processed/). Grouping them means `run_all` runs them together
    without needing the user to call two stages separately.
    """
    importance.run(cfg)
    insights.run(cfg)


STAGES: dict[str, Callable[[dict[str, Any]], None]] = {
    "explore":   exploration.run,
    "clean":     cleaning.run,
    "engineer":  engineering.run,
    "eda":       eda.run,
    "train":     trainer.run,
    "advanced":  advanced.run,
    "tune":      optuna_tuner.run,
    "interpret": _run_interpret,
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
