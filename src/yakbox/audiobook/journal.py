from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from yakbox import __version__
from yakbox._files import atomic_write_bytes
from yakbox.contracts import schema_uri
from yakbox.errors import BuildError


@dataclass(frozen=True, slots=True)
class JournalEvent:
    schema: str
    schema_version: int
    yakbox_version: str
    timestamp: str
    run_id: str
    event: str
    node_id: str | None = None
    fingerprint: str | None = None
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    error: str | None = None
    usage: dict[str, object] | None = None


class RunJournal:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        event: str,
        *,
        node_id: str | None = None,
        fingerprint: str | None = None,
        artifact_path: str | None = None,
        artifact_sha256: str | None = None,
        error: str | None = None,
        usage: dict[str, object] | None = None,
    ) -> JournalEvent:
        if not event.strip():
            raise BuildError("Journal event name must not be empty")
        record = JournalEvent(
            schema=schema_uri("audiobook-journal"),
            schema_version=1,
            yakbox_version=__version__,
            timestamp=datetime.now(UTC).isoformat(),
            run_id=self.run_id,
            event=event,
            node_id=node_id,
            fingerprint=fingerprint,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            error=error,
            usage=usage,
        )
        payload = json.dumps(_event_dict(record), sort_keys=True)
        with self.path.open("ab") as stream:
            stream.write(payload.encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def events(self) -> tuple[dict[str, object], ...]:
        if not self.path.exists():
            return ()
        content = self.path.read_bytes()
        lines = content.splitlines(keepends=True)
        events: list[dict[str, object]] = []
        valid_bytes = 0
        recovered = False
        for index, line in enumerate(lines):
            if not line.endswith(b"\n"):
                if index != len(lines) - 1:
                    raise BuildError(f"Corrupt journal record {index + 1}")
                diagnostic = self.path.with_suffix(".torn")
                atomic_write_bytes(diagnostic, line, overwrite=True)
                with self.path.open("r+b") as stream:
                    stream.truncate(valid_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                recovered = True
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise BuildError(f"Corrupt journal record {index + 1}") from error
            _validate_event(value, self.run_id, index + 1)
            events.append(value)
            valid_bytes += len(line)
        if recovered:
            recovery = self.append(
                "journal_recovered",
                error="discarded and preserved a torn final journal record",
            )
            events.append(_event_dict(recovery))
        return tuple(events)


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


@contextmanager
def target_lock(state_root: Path, target: str) -> Iterator[None]:
    lock_dir = state_root / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{target}.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise BuildError(
            f"Target {target!r} is already locked; inspect {lock_path}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _event_dict(record: JournalEvent) -> dict[str, object]:
    return {
        "$schema" if field == "schema" else field: value
        for field, value in asdict(record).items()
        if value is not None
    }


def _validate_event(value: object, run_id: str, index: int) -> None:
    if not isinstance(value, dict):
        raise BuildError(f"Invalid journal record {index}")
    if (
        value.get("$schema") != schema_uri("audiobook-journal")
        or value.get("schema_version") != 1
    ):
        raise BuildError(f"Unsupported journal record {index}")
    if value.get("run_id") != run_id:
        raise BuildError(f"Journal run identity mismatch at record {index}")
    if not isinstance(value.get("event"), str) or not isinstance(
        value.get("timestamp"), str
    ):
        raise BuildError(f"Invalid journal record {index}")
