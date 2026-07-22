from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return config


def load_multi_config(config_dir: str | Path) -> dict[str, Any]:
    """Load and merge all YAML configs from a directory."""
    config_dir = Path(config_dir)
    merged = {}
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        merged.update(load_config(yaml_file))
    return merged


def save_json(data: dict, path: str | Path) -> None:
    """Save a dictionary as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved JSON to %s", path)


def load_json(path: str | Path) -> dict:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)
