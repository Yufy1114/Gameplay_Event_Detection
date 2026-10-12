"""Structured JSON and CSV event writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .types import GameplayEvent

CSV_FIELDS = [
    "timestamp",
    "frame_id",
    "event",
    "object_id",
    "object_type",
    "confidence",
    "bbox",
    "metadata",
]


def _prepare_path(path: str | Path) -> Path:
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Cannot create output directory for {output_path}") from exc
    if output_path.exists() and output_path.is_dir():
        raise OSError(f"Output path is a directory: {output_path}")
    return output_path


def write_events_json(events: list[GameplayEvent], path: str | Path) -> None:
    output_path = _prepare_path(path)
    try:
        output_path.write_text(
            json.dumps(
                [event.to_dict() for event in events],
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise OSError(f"Cannot write JSON output: {output_path}") from exc


def write_events_csv(events: list[GameplayEvent], path: str | Path) -> None:
    output_path = _prepare_path(path)
    try:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for event in events:
                row = event.to_dict()
                row["bbox"] = (
                    json.dumps(row.get("bbox"), allow_nan=False)
                    if row.get("bbox") is not None
                    else ""
                )
                row["metadata"] = (
                    json.dumps(row.get("metadata"), allow_nan=False) if row.get("metadata") else ""
                )
                writer.writerow(row)
    except OSError as exc:
        raise OSError(f"Cannot write CSV output: {output_path}") from exc
