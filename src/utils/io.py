"""I/O helpers for configs and experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def ensure_parent_dir(path: str | Path) -> Path:
    """Create a file's parent directory if needed and return the path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    """Save a mapping as formatted JSON."""
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file and return a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_csv(rows: pd.DataFrame | Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Save a DataFrame or iterable of dictionaries as CSV."""
    output_path = ensure_parent_dir(path)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
