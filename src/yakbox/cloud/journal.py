from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from yakbox._files import atomic_write_bytes
from yakbox.cloud.errors import BatchJournalError
from yakbox.contracts import schema_uri


@dataclass(slots=True)
class _Write:
    record: dict[str, object]
    acknowledgement: asyncio.Future[None]


class BatchJournalWriter:
    def __init__(self, path: Path, *, append: bool = False) -> None:
        self.path = path
        self.append = append
        self._queue: asyncio.Queue[_Write | None] = asyncio.Queue(maxsize=100)
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> BatchJournalWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self.append:
            raise BatchJournalError(
                f"Batch journal already exists; use --resume: {self.path}"
            )
        if self.append and self.path.exists():
            _require_appendable(self.path)
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._task is not None and self._task.done():
            await self._task
            return
        await self._queue.put(None)
        if self._task is not None:
            await self._task

    async def append_record(self, record: dict[str, object]) -> None:
        if self._task is None:
            raise BatchJournalError("Batch journal writer is not open")
        if self._task.done():
            await self._task
            raise BatchJournalError("Batch journal writer stopped unexpectedly")
        loop = asyncio.get_running_loop()
        acknowledgement: asyncio.Future[None] = loop.create_future()
        await self._queue.put(_Write(record, acknowledgement))
        await acknowledgement

    async def _run(self) -> None:
        mode = "ab" if self.append else "xb"
        try:
            with self.path.open(mode) as stream:
                while True:
                    item = await self._queue.get()
                    if item is None:
                        break
                    try:
                        payload = (
                            json.dumps(item.record, sort_keys=True, ensure_ascii=False)
                            + "\n"
                        ).encode()
                        await asyncio.to_thread(_durable_append, stream, payload)
                    except Exception as error:
                        item.acknowledgement.set_exception(error)
                        raise
                    else:
                        item.acknowledgement.set_result(None)
        except Exception as error:
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if item is not None and not item.acknowledgement.done():
                    item.acknowledgement.set_exception(error)
            raise BatchJournalError(f"Cannot write batch journal: {error}") from error


def read_journal(path: Path) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise BatchJournalError(f"Cannot read batch journal {path}: {error}") from error
    records: list[dict[str, object]] = []
    valid_bytes = 0
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if index == len(lines) - 1:
                diagnostic = path.with_suffix(".torn")
                atomic_write_bytes(diagnostic, line, overwrite=True)
                try:
                    with path.open("r+b") as stream:
                        stream.truncate(valid_bytes)
                        stream.flush()
                        os.fsync(stream.fileno())
                except OSError as error:
                    raise BatchJournalError(
                        f"Cannot recover torn batch journal {path}: {error}"
                    ) from error
                break
            raise BatchJournalError(f"Corrupt journal record {index + 1}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise BatchJournalError(f"Corrupt journal record {index + 1}") from error
        if (
            not isinstance(record, dict)
            or record.get("$schema") != schema_uri("batch-journal")
            or record.get("schema_version") != 1
        ):
            raise BatchJournalError(f"Unsupported journal record {index + 1}")
        records.append(record)
        valid_bytes += len(line)
    return tuple(records)


def _durable_append(stream: BinaryIO, payload: bytes) -> None:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())


def _require_appendable(path: Path) -> None:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            if size == 0:
                return
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                raise BatchJournalError(
                    f"Batch journal has a torn final record; recover it first: {path}"
                )
    except OSError as error:
        raise BatchJournalError(
            f"Cannot inspect batch journal {path}: {error}"
        ) from error
