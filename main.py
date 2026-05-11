#!/usr/bin/env python3
"""Simple main for smoke-testing the project.

This script loads the TOML config, sets up logging using the project's
centralized helper and prints a small config summary. It's intended as a
lightweight entrypoint for manual testing.
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse
import logging

# Make the `src` directory importable so we can import helpers from
# `src/utils` without requiring an installed package.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from utils.config import load_config, get_nested
from utils.logging import get_logger


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Simple smoke test for the project")
	p.add_argument("--config", "-c", default=str(ROOT / "configs" / "config.toml"),
				   help="Path to config TOML file")
	p.add_argument("--log-file", "-l", default=None, help="Optional log file path")
	p.add_argument("--level", default="INFO", help="Logging level (DEBUG, INFO, ...)")
	p.add_argument("--show", action="store_true", help="Print a short config summary and exit")
	return p.parse_args()


def main() -> int:
	args = parse_args()
	# resolve numeric level from string if provided
	level = args.level
	if isinstance(level, str):
		level = getattr(logging, level.upper(), logging.INFO)

	logger = get_logger("main", level=level, log_file=Path(args.log_file) if args.log_file else None)
	logger.info("Starting simple main smoke test")

	try:
		cfg = load_config(args.config)
	except Exception as exc:
		logger.exception("Failed to load config from %s: %s", args.config, exc)
		return 2

	logger.info("Loaded config from %s", args.config)

	if args.show:
		print("Config summary:\n")
		print(" data.raw_path          =", get_nested(cfg, "data", "raw_path"))
		print(" data.processed_path    =", get_nested(cfg, "data", "processed_path"))
		print(" features.target        =", get_nested(cfg, "features", "target"))
		print(" models.random_forest.n_estimators =", get_nested(cfg, "models", "random_forest", "n_estimators"))
		print(" evaluation.cv_folds    =", get_nested(cfg, "evaluation", "cv_folds"))
		print()

	# do a tiny smoke check for the raw data file
	raw_path = get_nested(cfg, "data", "raw_path")
	if raw_path:
		raw = Path(raw_path)
		if raw.exists():
			logger.info("Raw data exists at %s", raw)
		else:
			logger.warning("Raw data not found at %s", raw)

	logger.info("Smoke test complete")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

