#!/usr/bin/env python3
"""Single CLI entry point for the Flight Fare Prediction pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.config import load_config
from utils.logging import get_logger
from pipeline.runner import STAGES, run_all, run_stage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flight Fare Prediction Pipeline")
    p.add_argument(
        "--stage", "-s",
        choices=[*STAGES.keys(), "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    p.add_argument("--config", "-c", default=str(ROOT / "configs" / "config.toml"))
    p.add_argument("--log-file", "-l", default=None, help="Optional log file path")
    p.add_argument("--level", default="INFO", help="Logging level")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    level = getattr(logging, args.level.upper(), logging.INFO)

    # Set root logger level so all module-level loggers inherit it.
    logging.getLogger().setLevel(level)

    logger = get_logger(
        "pipeline",
        level=level,
        log_file=Path(args.log_file) if args.log_file else None,
    )

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        logger.exception("Failed to load config: %s", exc)
        return 2

    try:
        if args.stage == "all":
            run_all(cfg)
        else:
            run_stage(args.stage, cfg)
    except Exception as exc:
        logger.exception("Pipeline failed at stage '%s': %s", args.stage, exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
