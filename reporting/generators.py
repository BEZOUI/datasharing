"""Automated reporting utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def _stringify(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _compute_column_widths(columns: Iterable[str], rows: Iterable[Iterable[str]]) -> List[int]:
    widths = [len(col) for col in columns]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    return widths


def _dataframe_to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "No records available."

    columns = [str(col) for col in table.columns]
    string_rows: List[List[str]] = []
    for _, row in table.iterrows():
        string_rows.append([_stringify(row[col]) for col in table.columns])

    widths = _compute_column_widths(columns, string_rows)

    def _format_row(values: Iterable[str]) -> str:
        cells = [f" {value.ljust(widths[idx])} " for idx, value in enumerate(values)]
        return "|" + "|".join(cells) + "|"

    header = _format_row(columns)
    separator_cells = ["-" * (width + 2) for width in widths]
    separator = "|" + "|".join(separator_cells) + "|"
    body = [_format_row(row) for row in string_rows]
    return "\n".join([header, separator, *body])


class MarkdownReporter:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def render(self, metrics: Dict[str, float], table: pd.DataFrame) -> Path:
        lines = ["# Experiment Summary", "", "## Aggregate Metrics"]
        for key, value in metrics.items():
            lines.append(f"- **{key}**: {value:.3f}")
        lines.append("\n## Detailed Results")
        lines.append(_dataframe_to_markdown(table))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("\n".join(lines), encoding="utf-8")
        return self.output_path
