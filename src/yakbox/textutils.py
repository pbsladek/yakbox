"""Shared batch input conventions for local and hosted synthesis."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

from yakbox.errors import ValidationError
from yakbox.yaml_config import iter_yaml_documents


@dataclass(frozen=True, slots=True)
class BatchRow:
    """Normalized local or hosted batch-input row with validation state."""

    index: int
    text: str
    row_id: str | None = None
    voice_uuid: str | None = None
    title: str | None = None
    output: str | None = None
    validation_error: str | None = None


def read_batch_rows(path: Path) -> tuple[BatchRow, ...]:
    return tuple(iter_batch_rows(path))


def iter_batch_rows(path: Path) -> Iterator[BatchRow]:
    suffix = path.suffix.casefold()
    if suffix not in {".txt", ".csv", ".yaml", ".yml"}:
        raise ValidationError("Batch input must use .txt, .csv, .yaml, or .yml")
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            if suffix == ".txt":
                yield from (
                    BatchRow(index=line_number, text=line.strip())
                    for line_number, line in enumerate(stream, 1)
                    if line.strip()
                )
            elif suffix == ".csv":
                yield from _iter_csv(stream)
            else:
                yield from _iter_yaml(stream)
    except (OSError, UnicodeDecodeError) as error:
        raise ValidationError(f"Cannot read batch input {path}: {error}") from error


def _iter_csv(stream: TextIO) -> Iterator[BatchRow]:
    reader = csv.DictReader(stream)
    if reader.fieldnames is None or "text" not in reader.fieldnames:
        raise ValidationError("CSV batch input requires a text header")
    allowed = {"text", "id", "voice_uuid", "title", "output"}
    unknown = set(reader.fieldnames) - allowed
    if unknown:
        raise ValidationError(f"Unknown CSV columns: {', '.join(sorted(unknown))}")
    for index, item in enumerate(reader, 2):
        if None in item:
            yield BatchRow(index=index, text="", validation_error="extra columns")
            continue
        yield BatchRow(
            index=index,
            text=(item.get("text") or "").strip(),
            row_id=_none(item.get("id")),
            voice_uuid=_none(item.get("voice_uuid")),
            title=_none(item.get("title")),
            output=_none(item.get("output")),
        )


def _iter_yaml(stream: TextIO) -> Iterator[BatchRow]:
    index = 0
    for document in iter_yaml_documents(stream, description="YAML batch input"):
        values = document if isinstance(document, list) else [document]
        for value in values:
            index += 1
            yield _structured_batch_row(value, index)


def _structured_batch_row(value: object, index: int) -> BatchRow:
    if not isinstance(value, dict):
        return BatchRow(
            index=index,
            text="",
            validation_error="record must be a mapping",
        )
    item = cast(dict[object, object], value)
    allowed = {"text", "id", "voice_uuid", "title", "output"}
    unknown = set(item) - allowed
    if unknown:
        names = ", ".join(sorted((str(key) for key in unknown), key=str.casefold))
        return BatchRow(
            index=index,
            text="",
            validation_error=f"unknown keys: {names}",
        )
    return BatchRow(
        index=index,
        text=str(item.get("text", "")).strip(),
        row_id=_value(item.get("id")),
        voice_uuid=_value(item.get("voice_uuid")),
        title=_value(item.get("title")),
        output=_value(item.get("output")),
    )


def _none(value: str | None) -> str | None:
    return value if value else None


def _value(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
