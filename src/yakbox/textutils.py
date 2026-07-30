"""Shared batch input conventions for local and hosted synthesis."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from yakbox.errors import ValidationError


@dataclass(frozen=True, slots=True)
class BatchRow:
    index: int
    text: str
    row_id: str | None = None
    voice_uuid: str | None = None
    title: str | None = None
    output: str | None = None
    validation_error: str | None = None


def read_batch_rows(path: Path) -> tuple[BatchRow, ...]:
    suffix = path.suffix.casefold()
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise ValidationError(f"Cannot read batch input {path}: {error}") from error
    if suffix == ".txt":
        return tuple(
            BatchRow(index=line_number, text=line.strip())
            for line_number, line in enumerate(text.splitlines(), 1)
            if line.strip()
        )
    if suffix == ".csv":
        return _read_csv(text)
    if suffix == ".jsonl":
        return _read_jsonl(text)
    raise ValidationError("Batch input must use .txt, .csv, or .jsonl")


def _read_csv(text: str) -> tuple[BatchRow, ...]:
    lines = text.splitlines()
    if not lines:
        raise ValidationError("CSV batch input is empty")
    reader = csv.DictReader(lines)
    if reader.fieldnames is None or "text" not in reader.fieldnames:
        raise ValidationError("CSV batch input requires a text header")
    allowed = {"text", "id", "voice_uuid", "title", "output"}
    unknown = set(reader.fieldnames) - allowed
    if unknown:
        raise ValidationError(f"Unknown CSV columns: {', '.join(sorted(unknown))}")
    rows: list[BatchRow] = []
    for index, item in enumerate(reader, 2):
        if None in item:
            rows.append(
                BatchRow(index=index, text="", validation_error="extra columns")
            )
            continue
        rows.append(
            BatchRow(
                index=index,
                text=(item.get("text") or "").strip(),
                row_id=_none(item.get("id")),
                voice_uuid=_none(item.get("voice_uuid")),
                title=_none(item.get("title")),
                output=_none(item.get("output")),
            )
        )
    return tuple(rows)


def _read_jsonl(text: str) -> tuple[BatchRow, ...]:
    rows: list[BatchRow] = []
    allowed = {"text", "id", "voice_uuid", "title", "output"}
    for index, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            rows.append(
                BatchRow(
                    index=index,
                    text="",
                    validation_error=f"invalid JSON: {error.msg}",
                )
            )
            continue
        if not isinstance(item, dict):
            rows.append(
                BatchRow(
                    index=index, text="", validation_error="record must be an object"
                )
            )
            continue
        item = cast(dict[str, object], item)
        unknown = set(item) - allowed
        if unknown:
            rows.append(
                BatchRow(
                    index=index,
                    text="",
                    validation_error=f"unknown keys: {', '.join(sorted(unknown))}",
                )
            )
            continue
        rows.append(
            BatchRow(
                index=index,
                text=str(item.get("text", "")).strip(),
                row_id=_value(item.get("id")),
                voice_uuid=_value(item.get("voice_uuid")),
                title=_value(item.get("title")),
                output=_value(item.get("output")),
            )
        )
    return tuple(rows)


def _none(value: str | None) -> str | None:
    return value if value else None


def _value(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
