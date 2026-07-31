from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast
from uuid import uuid4

from yakbox import __version__
from yakbox._files import atomic_write_json, sha256_file
from yakbox.cloud.errors import (
    CloudError,
    HostedBudgetExceeded,
    ProviderError,
    ResumeMismatchError,
    RetryExhaustedError,
)
from yakbox.cloud.journal import BatchJournalWriter, read_journal
from yakbox.cloud.output import slugify, validate_output_name
from yakbox.cloud.usage import HostedUsageGate
from yakbox.contracts import runtime_metadata, schema_uri, utc_timestamp
from yakbox.errors import ArtifactError, ValidationError, stable_error_code
from yakbox.speech.guardrails import HostedWorkEstimate
from yakbox.speech.models import (
    AudioFormat,
    HostedUsageSnapshot,
    Precision,
    SpeechSynthesisRequest,
)
from yakbox.speech.services import TextToSpeechService
from yakbox.textutils import BatchRow


class BatchStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    NOT_RUN = "not_run"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Ordered outcome and integrity evidence for one hosted batch row."""

    index: int
    row_id: str | None
    status: BatchStatus
    path: Path | None
    bytes_written: int | None
    output_format: AudioFormat
    duration_seconds: float | None
    elapsed_seconds: float
    attempts: int
    request_id: str | None
    issues: tuple[str, ...]
    error_code: str | None
    error_message: str | None
    text_sha256: str
    request_sha256: str
    output_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        """Serialize one ordered batch result."""
        value = asdict(self)
        value["status"] = self.status.value
        value["path"] = str(self.path) if self.path else None
        value["output_format"] = self.output_format.value
        value["issues"] = list(self.issues)
        return value


@dataclass(frozen=True, slots=True)
class BatchReport:
    """Resumable hosted-batch outcome with per-row results and usage totals."""

    schema_version: int
    run_id: str
    started_at: str
    ended_at: str
    journal_path: Path
    results: tuple[BatchResult, ...]
    aborted: bool
    abort_reason: str | None
    usage: HostedUsageSnapshot | None = None
    preflight: HostedWorkEstimate | None = None

    @property
    def ok(self) -> int:
        """Return the count of successful or already-complete rows."""
        return sum(
            result.status in {BatchStatus.OK, BatchStatus.SKIPPED}
            for result in self.results
        )

    @property
    def failed(self) -> int:
        """Return the count of row-local errors."""
        return sum(result.status is BatchStatus.ERROR for result in self.results)

    def to_dict(self) -> dict[str, object]:
        """Serialize the versioned batch report and usage summary."""
        return {
            **runtime_metadata("batch-report", timestamp=self.ended_at),
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "journal_path": str(self.journal_path),
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "usage": _usage_dict(self.usage) if self.usage is not None else None,
            "preflight": (
                self.preflight.to_dict() if self.preflight is not None else None
            ),
            "summary": {
                "ok": self.ok,
                "failed": self.failed,
                "not_run": sum(
                    item.status is BatchStatus.NOT_RUN for item in self.results
                ),
            },
            "results": [item.to_dict() for item in self.results],
        }


type ProgressCallback = Callable[[BatchResult], None]
MAX_CONCURRENCY = 100
MAX_SYNTHESIS_CHARACTERS = 3_000


@dataclass(frozen=True, slots=True)
class _BatchSettings:
    default_voice: str | None
    project_uuid: str | None
    out_dir: Path
    concurrency: int
    output_format: AudioFormat
    use_hd: bool
    precision: Precision | None
    sample_rate: int | None
    apply_custom_pronunciations: bool
    overwrite: bool
    dry_run: bool
    progress: ProgressCallback | None
    journal_path: Path | None
    report_path: Path | None
    resume_path: Path | None
    write_report: bool
    usage_gate: HostedUsageGate | None
    preflight: HostedWorkEstimate | None

    def journal_options(self) -> dict[str, object]:
        return {
            "project_uuid": self.project_uuid,
            "format": self.output_format.value,
            "use_hd": self.use_hd,
            "precision": self.precision,
            "sample_rate": self.sample_rate,
            "apply_custom_pronunciations": self.apply_custom_pronunciations,
            "concurrency": self.concurrency,
        }


@dataclass(slots=True)
class _BatchRun:
    rows: _RowSpool
    service: TextToSpeechService
    settings: _BatchSettings
    run_id: str
    started_at: str
    journal: Path
    report: Path
    resumed: dict[int, BatchResult]
    append_journal: bool
    results: dict[int, BatchResult]
    stop: asyncio.Event
    abort_reason: str | None = None


@contextmanager
def _row_spool_stream() -> Iterator[tempfile.SpooledTemporaryFile[str]]:
    with tempfile.SpooledTemporaryFile(
        max_size=1024 * 1024,
        mode="w+t",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        yield stream


class _RowSpool:
    def __init__(self, rows: Iterable[BatchRow]) -> None:
        self._resources = ExitStack()
        self._stream = self._resources.enter_context(_row_spool_stream())
        digest = hashlib.sha256()
        self.count = 0
        for row in rows:
            self._stream.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            digest.update(str(row.index).encode())
            digest.update(b"\0")
            digest.update(row.text.encode())
            digest.update(b"\0")
            self.count += 1
        self.input_digest = digest.hexdigest()
        self._stream.flush()

    def rows(self) -> Iterator[BatchRow]:
        self._stream.seek(0)
        for line in self._stream:
            raw = json.loads(line)
            yield BatchRow(
                index=int(raw["index"]),
                text=str(raw["text"]),
                row_id=_string_or_none(raw.get("row_id")),
                voice_uuid=_string_or_none(raw.get("voice_uuid")),
                title=_string_or_none(raw.get("title")),
                output=_string_or_none(raw.get("output")),
                validation_error=_string_or_none(raw.get("validation_error")),
            )

    def close(self) -> None:
        self._resources.close()


async def run_cloud_batch(
    rows: Iterable[BatchRow],
    service: TextToSpeechService,
    *,
    default_voice: str | None,
    project_uuid: str | None,
    out_dir: Path,
    concurrency: int = 5,
    output_format: AudioFormat = AudioFormat.WAV,
    use_hd: bool = False,
    precision: Precision | None = None,
    sample_rate: int | None = None,
    apply_custom_pronunciations: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    progress: ProgressCallback | None = None,
    journal_path: Path | None = None,
    report_path: Path | None = None,
    resume_path: Path | None = None,
    write_report: bool = True,
    usage_gate: HostedUsageGate | None = None,
    preflight: HostedWorkEstimate | None = None,
) -> BatchReport:
    """Run a bounded, journaled, resumable hosted batch through one service."""
    spool = _RowSpool(rows)
    try:
        settings = _BatchSettings(
            default_voice=default_voice,
            project_uuid=project_uuid,
            out_dir=out_dir,
            concurrency=concurrency,
            output_format=output_format,
            use_hd=use_hd,
            precision=precision,
            sample_rate=sample_rate,
            apply_custom_pronunciations=apply_custom_pronunciations,
            overwrite=overwrite,
            dry_run=dry_run,
            progress=progress,
            journal_path=journal_path,
            report_path=report_path,
            resume_path=resume_path,
            write_report=write_report,
            usage_gate=usage_gate,
            preflight=preflight,
        )
        return await _run_cloud_batch_spooled(spool, service, settings)
    finally:
        spool.close()


async def _run_cloud_batch_spooled(
    rows: _RowSpool,
    service: TextToSpeechService,
    settings: _BatchSettings,
) -> BatchReport:
    if not 1 <= settings.concurrency <= MAX_CONCURRENCY:
        raise ValidationError("concurrency must be between 1 and 100")
    run = await _prepare_batch_run(rows, service, settings)
    if settings.dry_run:
        return await _dry_run_report(run)
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    terminal_usage = await _execute_batch(run)
    report = _batch_report(run, terminal_usage)
    if settings.write_report:
        atomic_write_json(run.report, report.to_dict())
    return report


async def _prepare_batch_run(
    rows: _RowSpool,
    service: TextToSpeechService,
    settings: _BatchSettings,
) -> _BatchRun:
    run = _BatchRun(
        rows=rows,
        service=service,
        settings=settings,
        run_id=uuid4().hex,
        started_at=datetime.now(UTC).isoformat(),
        journal=settings.journal_path or settings.out_dir / "batch-journal.ndjson",
        report=settings.report_path or settings.out_dir / "batch-report.json",
        resumed={},
        append_journal=False,
        results={},
        stop=asyncio.Event(),
    )
    if settings.resume_path is None:
        return run
    (
        run.journal,
        run.run_id,
        run.started_at,
        run.resumed,
        prior_usage,
    ) = _load_resume(
        settings.resume_path,
        input_digest=rows.input_digest,
        options=settings.journal_options(),
        resolved_rows=_resolved_rows(run),
        row_count=rows.count,
        output_format=settings.output_format,
        default_voice=settings.default_voice,
    )
    run.append_journal = True
    await _restore_usage(settings.usage_gate, prior_usage)
    return run


async def _restore_usage(
    usage_gate: HostedUsageGate | None,
    prior_usage: HostedUsageSnapshot,
) -> None:
    if usage_gate is None:
        return
    await usage_gate.restore_prior_usage(
        logical_items=prior_usage.logical_items,
        provider_attempts=prior_usage.provider_attempts,
        submitted_characters=prior_usage.submitted_characters,
        ambiguous_attempts=prior_usage.ambiguous_attempts,
    )


async def _dry_run_report(run: _BatchRun) -> BatchReport:
    settings = run.settings
    results = tuple(
        _validation_result(
            row,
            destination,
            settings.default_voice,
            settings.output_format,
        )
        for row, destination in _resolved_rows(run)
    )
    usage = (
        await settings.usage_gate.snapshot()
        if settings.usage_gate is not None
        else None
    )
    return BatchReport(
        schema_version=1,
        run_id=run.run_id,
        started_at=run.started_at,
        ended_at=datetime.now(UTC).isoformat(),
        journal_path=run.journal,
        results=results,
        aborted=False,
        abort_reason=None,
        usage=usage,
        preflight=settings.preflight,
    )


async def _execute_batch(run: _BatchRun) -> HostedUsageSnapshot | None:
    async with BatchJournalWriter(run.journal, append=run.append_journal) as writer:
        await _record_batch_start(run, writer)
        _set_usage_recorder(run, writer)
        try:
            await _run_batch_tasks(run, writer)
        except asyncio.CancelledError:
            await _finalize_interruption(run, writer, "batch cancelled")
            raise
        except Exception as error:
            await _finalize_interruption(
                run, writer, f"batch stopped after {stable_error_code(error)}"
            )
            raise
        finally:
            _set_usage_recorder(run, None)
        usage = await _usage_snapshot(run)
        await _record_batch_completion(run, writer, usage)
        return usage


async def _record_batch_start(run: _BatchRun, writer: BatchJournalWriter) -> None:
    if run.append_journal:
        timestamp = utc_timestamp()
        await writer.append_record(
            {
                "$schema": schema_uri("batch-journal"),
                "schema_version": 1,
                "record_type": "resumed",
                "run_id": run.run_id,
                "yakbox_version": __version__,
                "timestamp": timestamp,
                "resumed_at": timestamp,
            }
        )
        return
    await writer.append_record(
        {
            "$schema": schema_uri("batch-journal"),
            "schema_version": 1,
            "record_type": "header",
            "run_id": run.run_id,
            "input_sha256": run.rows.input_digest,
            "yakbox_version": __version__,
            "timestamp": run.started_at,
            "started_at": run.started_at,
            "options": run.settings.journal_options(),
        }
    )


def _set_usage_recorder(
    run: _BatchRun,
    writer: BatchJournalWriter | None,
) -> None:
    usage_gate = run.settings.usage_gate
    if usage_gate is None:
        return
    if writer is None:
        usage_gate.set_recorder(None)
        return

    async def record(
        snapshot: HostedUsageSnapshot,
        submitted_characters: int,
    ) -> None:
        await _record_usage_reservation(
            writer,
            run.run_id,
            snapshot,
            submitted_characters,
        )

    usage_gate.set_recorder(record)


async def _record_usage_reservation(
    writer: BatchJournalWriter,
    run_id: str,
    snapshot: HostedUsageSnapshot,
    submitted_characters: int,
) -> None:
    await writer.append_record(
        {
            "$schema": schema_uri("batch-journal"),
            "schema_version": 1,
            "record_type": "usage_reserved",
            "run_id": run_id,
            "yakbox_version": __version__,
            "timestamp": utc_timestamp(),
            "submitted_characters_this_attempt": submitted_characters,
            "usage": _usage_dict(snapshot),
        }
    )


async def _run_batch_tasks(run: _BatchRun, writer: BatchJournalWriter) -> None:
    queue: asyncio.Queue[tuple[BatchRow, Path] | None] = asyncio.Queue(
        maxsize=max(1, 2 * run.settings.concurrency)
    )
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(_produce_batch(run, writer, queue))
        for _ in range(run.settings.concurrency):
            tasks.create_task(_batch_worker(run, writer, queue))


async def _produce_batch(
    run: _BatchRun,
    writer: BatchJournalWriter,
    queue: asyncio.Queue[tuple[BatchRow, Path] | None],
) -> None:
    for row, destination in _resolved_rows(run):
        if row.index in run.resumed:
            await _store_result(run, writer, run.resumed[row.index])
        elif run.stop.is_set():
            result = _not_run(
                row, destination, run.settings.output_format, "batch aborted"
            )
            await _store_result(run, writer, result)
        else:
            await queue.put((row, destination))
    for _ in range(run.settings.concurrency):
        await queue.put(None)


async def _batch_worker(
    run: _BatchRun,
    writer: BatchJournalWriter,
    queue: asyncio.Queue[tuple[BatchRow, Path] | None],
) -> None:
    while (item := await queue.get()) is not None:
        row, destination = item
        result = await _process_batch_item(run, row, destination)
        if result.error_code in {"authentication", "hosted_budget_exceeded"}:
            run.abort_reason = result.error_message
            run.stop.set()
        await _store_result(run, writer, result)


async def _process_batch_item(
    run: _BatchRun,
    row: BatchRow,
    destination: Path,
) -> BatchResult:
    settings = run.settings
    if run.stop.is_set():
        return _not_run(row, destination, settings.output_format, "batch aborted")
    return await _synthesize_row(
        row,
        destination,
        run.service,
        default_voice=settings.default_voice,
        project_uuid=settings.project_uuid,
        output_format=settings.output_format,
        use_hd=settings.use_hd,
        precision=settings.precision,
        sample_rate=settings.sample_rate,
        apply_custom_pronunciations=settings.apply_custom_pronunciations,
        overwrite=settings.overwrite,
    )


async def _store_result(
    run: _BatchRun,
    writer: BatchJournalWriter,
    result: BatchResult,
) -> None:
    run.results[result.index] = result
    await _record(writer, result)
    if run.settings.progress is not None:
        run.settings.progress(result)


async def _finalize_interruption(
    run: _BatchRun,
    writer: BatchJournalWriter,
    reason: str,
) -> None:
    for row, destination in _resolved_rows(run):
        if row.index in run.results:
            continue
        result = _not_run(row, destination, run.settings.output_format, reason)
        run.results[row.index] = result
        await asyncio.shield(_record(writer, result))
    usage = await _usage_snapshot(run)
    await _record_interruption(writer, run.run_id, reason, usage)
    if run.settings.write_report:
        atomic_write_json(
            run.report,
            _batch_report(run, usage, interruption_reason=reason).to_dict(),
        )


async def _usage_snapshot(run: _BatchRun) -> HostedUsageSnapshot | None:
    usage_gate = run.settings.usage_gate
    return await usage_gate.snapshot() if usage_gate is not None else None


async def _record_batch_completion(
    run: _BatchRun,
    writer: BatchJournalWriter,
    usage: HostedUsageSnapshot | None,
) -> None:
    await writer.append_record(
        {
            "$schema": schema_uri("batch-journal"),
            "schema_version": 1,
            "record_type": "complete" if run.abort_reason is None else "aborted",
            "run_id": run.run_id,
            "yakbox_version": __version__,
            "timestamp": utc_timestamp(),
            "ended_at": utc_timestamp(),
            "abort_reason": run.abort_reason,
            "usage": _usage_dict(usage) if usage is not None else None,
        }
    )


def _batch_report(
    run: _BatchRun,
    usage: HostedUsageSnapshot | None,
    *,
    interruption_reason: str | None = None,
) -> BatchReport:
    reason = interruption_reason or run.abort_reason
    return BatchReport(
        schema_version=1,
        run_id=run.run_id,
        started_at=run.started_at,
        ended_at=datetime.now(UTC).isoformat(),
        journal_path=run.journal.resolve(),
        results=tuple(run.results[row.index] for row in run.rows.rows()),
        aborted=reason is not None,
        abort_reason=reason,
        usage=usage,
        preflight=run.settings.preflight,
    )


def _resolved_rows(run: _BatchRun) -> Iterator[tuple[BatchRow, Path]]:
    return _resolve_outputs(
        run.rows.rows(),
        run.settings.out_dir,
        run.settings.output_format,
    )


async def _synthesize_row(
    row: BatchRow,
    destination: Path,
    service: TextToSpeechService,
    *,
    default_voice: str | None,
    project_uuid: str | None,
    output_format: AudioFormat,
    use_hd: bool,
    precision: Precision | None,
    sample_rate: int | None,
    apply_custom_pronunciations: bool,
    overwrite: bool,
) -> BatchResult:
    started = time.monotonic()
    text_hash = hashlib.sha256(row.text.encode()).hexdigest()
    voice = row.voice_uuid or default_voice
    request_hash = _request_hash(
        row,
        voice,
        project_uuid,
        output_format,
        use_hd,
        precision=precision,
        sample_rate=sample_rate,
        apply_custom_pronunciations=apply_custom_pronunciations,
    )
    validation = _validation_error(row, voice)
    if validation is not None:
        return _error_result(
            row,
            destination,
            output_format,
            started=started,
            text_hash=text_hash,
            request_hash=request_hash,
            code="validation",
            message=validation,
        )
    try:
        artifact = await service.synthesize_to_file(
            SpeechSynthesisRequest(
                text=row.text,
                voice=voice or "",
                backend=service.capabilities.name,
                output_format=output_format,
                title=row.title,
                use_hd=use_hd,
                precision=precision,
                sample_rate=sample_rate,
                apply_custom_pronunciations=apply_custom_pronunciations,
                project=project_uuid,
            ),
            destination,
            overwrite=overwrite,
        )
    except (
        CloudError,
        ArtifactError,
        ValidationError,
        OSError,
    ) as error:
        code, attempts = _synthesis_error_details(error)
        return _error_result(
            row,
            destination,
            output_format,
            started=started,
            text_hash=text_hash,
            request_hash=request_hash,
            code=code,
            message=str(error),
            attempts=attempts,
        )
    return BatchResult(
        index=row.index,
        row_id=row.row_id,
        status=BatchStatus.OK,
        path=artifact.path,
        bytes_written=artifact.bytes_written,
        output_format=artifact.output_format,
        duration_seconds=artifact.duration_seconds,
        elapsed_seconds=time.monotonic() - started,
        attempts=artifact.attempts,
        request_id=artifact.request_id,
        issues=(),
        error_code=None,
        error_message=None,
        text_sha256=text_hash,
        request_sha256=request_hash,
        output_sha256=artifact.sha256,
    )


def _synthesis_error_details(error: Exception) -> tuple[str, int]:
    if isinstance(error, HostedBudgetExceeded):
        return "hosted_budget_exceeded", 0
    if isinstance(error, RetryExhaustedError):
        return "retry_exhausted", error.attempts
    if isinstance(error, ProviderError):
        code = "authentication" if error.status_code in {401, 403} else "provider"
        return code, 0
    return stable_error_code(error), 0


def _resolve_outputs(
    rows: Iterable[BatchRow], out_dir: Path, output_format: AudioFormat
) -> Iterator[tuple[BatchRow, Path]]:
    used: dict[str, int] = {}
    for row in rows:
        requested = row.output
        if requested:
            name = validate_output_name(requested)
            if Path(name).suffix.casefold() != f".{output_format.value}":
                name = f"{Path(name).stem}.{output_format.value}"
        else:
            label = row.row_id or row.title or row.text[:40]
            name = f"{row.index:06d}-{slugify(label)}.{output_format.value}"
        stem, suffix = Path(name).stem, Path(name).suffix
        count = used.get(name.casefold(), 0) + 1
        used[name.casefold()] = count
        if count > 1:
            name = f"{stem}-{count}{suffix}"
        yield row, (out_dir / name).resolve()


def _validation_error(row: BatchRow, voice: str | None) -> str | None:
    if row.validation_error:
        return row.validation_error
    if not row.text.strip():
        return "text must not be empty"
    if len(row.text) > MAX_SYNTHESIS_CHARACTERS:
        return (
            "text exceeds 3000 characters; split it into rows or use "
            "local yakbox batch --chunk-chars"
        )
    if not voice:
        return "voice_uuid is required"
    return None


def _validation_result(
    row: BatchRow,
    destination: Path,
    default_voice: str | None,
    output_format: AudioFormat,
) -> BatchResult:
    text_hash = hashlib.sha256(row.text.encode()).hexdigest()
    request_hash = _request_hash(
        row, row.voice_uuid or default_voice, None, output_format, False
    )
    error = _validation_error(row, row.voice_uuid or default_voice)
    if error:
        return _error_result(
            row,
            destination,
            output_format,
            started=time.monotonic(),
            text_hash=text_hash,
            request_hash=request_hash,
            code="validation",
            message=error,
        )
    return _not_run(row, destination, output_format, "dry run")


def _error_result(
    row: BatchRow,
    destination: Path,
    output_format: AudioFormat,
    *,
    started: float,
    text_hash: str,
    request_hash: str,
    code: str,
    message: str,
    attempts: int = 0,
) -> BatchResult:
    return BatchResult(
        index=row.index,
        row_id=row.row_id,
        status=BatchStatus.ERROR,
        path=destination,
        bytes_written=None,
        output_format=output_format,
        duration_seconds=None,
        elapsed_seconds=time.monotonic() - started,
        attempts=attempts,
        request_id=None,
        issues=(),
        error_code=code,
        error_message=message,
        text_sha256=text_hash,
        request_sha256=request_hash,
        output_sha256=None,
    )


def _not_run(
    row: BatchRow, destination: Path, output_format: AudioFormat, reason: str
) -> BatchResult:
    text_hash = hashlib.sha256(row.text.encode()).hexdigest()
    return BatchResult(
        index=row.index,
        row_id=row.row_id,
        status=BatchStatus.NOT_RUN,
        path=destination,
        bytes_written=None,
        output_format=output_format,
        duration_seconds=None,
        elapsed_seconds=0,
        attempts=0,
        request_id=None,
        issues=(),
        error_code="not_run",
        error_message=reason,
        text_sha256=text_hash,
        request_sha256="",
        output_sha256=None,
    )


def _request_hash(
    row: BatchRow,
    voice: str | None,
    project: str | None,
    output_format: AudioFormat,
    use_hd: bool,
    *,
    precision: Precision | None = None,
    sample_rate: int | None = None,
    apply_custom_pronunciations: bool = False,
) -> str:
    payload = json.dumps(
        {
            "text_sha256": hashlib.sha256(row.text.encode()).hexdigest(),
            "voice": voice,
            "project": project,
            "format": output_format.value,
            "hd": use_hd,
            "precision": precision,
            "sample_rate": sample_rate,
            "apply_custom_pronunciations": apply_custom_pronunciations,
            "title": row.title,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


async def _record(writer: BatchJournalWriter, result: BatchResult) -> None:
    record = result.to_dict()
    record.update(
        {
            "$schema": schema_uri("batch-journal"),
            "schema_version": 1,
            "yakbox_version": __version__,
            "timestamp": utc_timestamp(),
            "record_type": "row",
        }
    )
    await writer.append_record(record)


async def _record_interruption(
    writer: BatchJournalWriter,
    run_id: str,
    reason: str,
    usage: HostedUsageSnapshot | None,
) -> None:
    await asyncio.shield(
        writer.append_record(
            {
                "$schema": schema_uri("batch-journal"),
                "schema_version": 1,
                "record_type": "interrupted",
                "run_id": run_id,
                "yakbox_version": __version__,
                "timestamp": utc_timestamp(),
                "ended_at": utc_timestamp(),
                "abort_reason": reason,
                "usage": _usage_dict(usage) if usage is not None else None,
            }
        )
    )


def _load_resume(
    path: Path,
    *,
    input_digest: str,
    options: dict[str, object],
    resolved_rows: Iterable[tuple[BatchRow, Path]],
    row_count: int,
    output_format: AudioFormat,
    default_voice: str | None,
) -> tuple[Path, str, str, dict[int, BatchResult], HostedUsageSnapshot]:
    resume_path = _resolve_resume_path(path)
    records = read_journal(resume_path)
    run_id, started_at = _resume_identity(records, input_digest, options)
    successful = _successful_resume_records(records)
    attempts_by_index = _resume_attempts(records)
    completed, fallback_attempts, fallback_characters = _restore_completed_rows(
        resolved_rows,
        successful,
        attempts_by_index,
        options=options,
        output_format=output_format,
        default_voice=default_voice,
    )
    prior_usage = _resume_usage(
        records,
        row_count=row_count,
        fallback_attempts=fallback_attempts,
        fallback_characters=fallback_characters,
    )
    return resume_path, run_id, started_at, completed, prior_usage


def _resolve_resume_path(path: Path) -> Path:
    resume_path = path.resolve()
    if resume_path.suffix.casefold() != ".json":
        return resume_path
    try:
        report = json.loads(resume_path.read_text(encoding="utf-8"))
        return Path(str(report["journal_path"])).resolve()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ResumeMismatchError(
            f"Cannot resolve resume report {path}: {error}"
        ) from error


def _resume_identity(
    records: tuple[dict[str, object], ...],
    input_digest: str,
    options: dict[str, object],
) -> tuple[str, str]:
    if not records or records[0].get("record_type") != "header":
        raise ResumeMismatchError("Resume journal lacks a valid header")
    header = records[0]
    if header.get("input_sha256") != input_digest:
        raise ResumeMismatchError("Resume input digest does not match")
    if header.get("options") != options:
        raise ResumeMismatchError("Resume synthesis options do not match")
    run_id = str(header.get("run_id", ""))
    started_at = str(header.get("started_at", ""))
    if not run_id or not started_at:
        raise ResumeMismatchError("Resume journal header is incomplete")
    return run_id, started_at


def _successful_resume_records(
    records: tuple[dict[str, object], ...],
) -> dict[int, dict[str, object]]:
    return {
        index: record
        for record in records[1:]
        if record.get("record_type") == "row"
        and record.get("status") == "ok"
        and isinstance(index := record.get("index"), int)
    }


def _resume_attempts(
    records: tuple[dict[str, object], ...],
) -> dict[int, int]:
    attempts_by_index: dict[int, int] = {}
    for record in records[1:]:
        if record.get("record_type") != "row" or record.get("status") == "skipped":
            continue
        index = _integer_or_none(record.get("index"))
        if index is not None:
            attempts_by_index[index] = attempts_by_index.get(index, 0) + (
                _integer_or_none(record.get("attempts")) or 0
            )
    return attempts_by_index


def _restore_completed_rows(
    resolved_rows: Iterable[tuple[BatchRow, Path]],
    successful: dict[int, dict[str, object]],
    attempts_by_index: dict[int, int],
    *,
    options: dict[str, object],
    output_format: AudioFormat,
    default_voice: str | None,
) -> tuple[dict[int, BatchResult], int, int]:
    completed: dict[int, BatchResult] = {}
    fallback_attempts = 0
    fallback_characters = 0
    for row, destination in resolved_rows:
        count = attempts_by_index.get(row.index, 0)
        fallback_attempts += count
        fallback_characters += count * len(row.text)
        record = successful.get(row.index)
        if record is None:
            continue
        result = _restore_completed_row(
            row,
            destination,
            record,
            options=options,
            output_format=output_format,
            default_voice=default_voice,
        )
        if result is not None:
            completed[row.index] = result
    return completed, fallback_attempts, fallback_characters


def _restore_completed_row(
    row: BatchRow,
    destination: Path,
    record: dict[str, object],
    *,
    options: dict[str, object],
    output_format: AudioFormat,
    default_voice: str | None,
) -> BatchResult | None:
    request_hash = _resume_request_hash(row, options, output_format, default_voice)
    if record.get("request_sha256") != request_hash:
        return None
    recorded_path = Path(str(record.get("path", destination))).resolve()
    recorded_size = record.get("bytes_written")
    recorded_digest = record.get("output_sha256")
    if not _recorded_output_matches(recorded_path, recorded_size, recorded_digest):
        return None
    recorded_size = cast(int, recorded_size)
    recorded_digest = cast(str, recorded_digest)
    return BatchResult(
        index=row.index,
        row_id=row.row_id,
        status=BatchStatus.SKIPPED,
        path=recorded_path,
        bytes_written=recorded_size,
        output_format=output_format,
        duration_seconds=_number_or_none(record.get("duration_seconds")),
        elapsed_seconds=0,
        attempts=_integer_or_none(record.get("attempts")) or 0,
        request_id=_string_or_none(record.get("request_id")),
        issues=_string_tuple(record.get("issues")),
        error_code=None,
        error_message=None,
        text_sha256=str(record.get("text_sha256", "")),
        request_sha256=str(record.get("request_sha256", "")),
        output_sha256=recorded_digest,
    )


def _resume_request_hash(
    row: BatchRow,
    options: dict[str, object],
    output_format: AudioFormat,
    default_voice: str | None,
) -> str:
    return _request_hash(
        row,
        row.voice_uuid or default_voice,
        _string_or_none(options.get("project_uuid")),
        output_format,
        bool(options.get("use_hd", False)),
        precision=(
            Precision(value)
            if (value := _string_or_none(options.get("precision"))) is not None
            else None
        ),
        sample_rate=_integer_or_none(options.get("sample_rate")),
        apply_custom_pronunciations=bool(
            options.get("apply_custom_pronunciations", False)
        ),
    )


def _recorded_output_matches(
    path: Path,
    size: object,
    digest: object,
) -> bool:
    return (
        path.is_file()
        and isinstance(size, int)
        and path.stat().st_size == size
        and isinstance(digest, str)
        and sha256_file(path) == digest
    )


def _resume_usage(
    records: tuple[dict[str, object], ...],
    *,
    row_count: int,
    fallback_attempts: int,
    fallback_characters: int,
) -> HostedUsageSnapshot:
    for record in reversed(records):
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue
        return HostedUsageSnapshot(
            logical_items=_integer_or_none(usage.get("logical_items")) or 0,
            provider_attempts=_integer_or_none(usage.get("provider_attempts")) or 0,
            submitted_characters=_integer_or_none(usage.get("submitted_characters"))
            or 0,
            estimated_spend=None,
            currency=None,
            ambiguous_attempts=_integer_or_none(usage.get("ambiguous_attempts")) or 0,
        )
    return HostedUsageSnapshot(
        logical_items=row_count,
        provider_attempts=fallback_attempts,
        submitted_characters=fallback_characters,
        estimated_spend=None,
        currency=None,
        ambiguous_attempts=0,
    )


def _number_or_none(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _integer_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _usage_dict(snapshot: HostedUsageSnapshot) -> dict[str, object]:
    return {
        "logical_items": snapshot.logical_items,
        "provider_attempts": snapshot.provider_attempts,
        "submitted_characters": snapshot.submitted_characters,
        "estimated_spend": (
            str(snapshot.estimated_spend)
            if snapshot.estimated_spend is not None
            else None
        ),
        "currency": str(snapshot.currency) if snapshot.currency is not None else None,
        "ambiguous_attempts": snapshot.ambiguous_attempts,
        "basis": "estimate" if snapshot.estimated_spend is not None else "usage_only",
    }
