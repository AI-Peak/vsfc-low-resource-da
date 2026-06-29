"""Result reporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    """Read a CSV file, returning an empty frame when it is missing."""
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)

def format_number(value: object, digits: int = 4) -> str:
    """Format a report cell with compact numeric output."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)

def markdown_table(rows: Iterable[Mapping[str, object]], columns: list[str]) -> str:
    """Render rows as a simple Markdown table without optional dependencies."""
    row_list = list(rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in row_list:
        values = [format_number(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)

def write_text(path: str | Path, content: str) -> Path:
    """Write text content and return the path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
