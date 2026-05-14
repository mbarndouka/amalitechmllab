"""Configuration loading — reads config.toml via stdlib tomllib (Python 3.11+)."""

from __future__ import annotations

from pathlib import Path
from tomllib import load as load_toml
from typing import Any


def load_config(path: str | Path = "configs/config.toml") -> dict[str, Any]:
    """Load TOML config from *path* and return as a plain dict."""
    with open(path, "rb") as f:
        return load_toml(f)


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Get a nested value from *config* using *keys*."""
    node = config
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
    return node
