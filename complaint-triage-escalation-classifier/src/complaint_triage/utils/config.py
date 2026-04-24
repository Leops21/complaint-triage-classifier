"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary to JSON."""
    import json

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
