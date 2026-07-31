from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
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
from yakbox.errors import ArtifactError, ValidationError
from yakbox.speech.guardrails import HostedWorkEstimate
from yakbox.speech.models import (
    AudioFormat,
    HostedUsageSnapshot,
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
        value = asdict(self)
        value["status"] = self.status.value
        value["path"] = str(self.path) if self.path else None
        value["output_format"] = self.output_format.value
        value["issues"] = list(self.issues)
        return value


@dataclass(frozen=True, slots=True)
class BatchReport:
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
        return sum(
            result.status in {BatchStatus.OK, BatchStatus.SKIPPED}
            for result in self.results
        )

    @property
    def failed(self) -> int:
        return sum(result.status is BatchStatus.ERROR for result in self.results)

    def to_dict(self) -> dict[str, object]:
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


ProgressCallback = Callable[[BatchResult], None]
MAX_CONCURRENCY = 100
MAX_SYNTHESIS_CHARACTERS = 3_000


class _RowSpool:
    def __init__(self, rows: Iterable[BatchRow]) -> None:
        self._stream = tempfile.SpooledTemporaryFile(  # noqa: SIM115
            max_size=1024 * 1024,
            mode="w+t",
            encoding="utf-8",
            newline="\n",
        )
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
        self._stream.close()


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
    precision: str | None = None,
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
    spool = _RowSpool(rows)
    try:
        return await _run_cloud_batch_spooled(
            spool,
            service,
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
    finally:
        spool.close()


async def _run_cloud_batch_spooled(
    rows: _RowSpool,
    service: TextToSpeechService,
    *,
    default_voice: str | None,
    project_uuid: str | None,
    out_dir: Path,
    concurrency: int = 5,
    output_format: AudioFormat = AudioFormat.WAV,
    use_hd: bool = False,
    precision: str | None = None,
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
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValidationError("concurrency must be between 1 and 100")
    input_digest = rows.input_digest
    run_id = uuid4().hex
    started_at = datetime.now(UTC).isoformat()
    journal = journal_path or out_dir / "batch-journal.ndjson"
    report = report_path or out_dir / "batch-report.json"
    options: dict[str, object] = {
        "project_uuid": project_uuid,
        "format": output_format.value,
        "use_hd": use_hd,
        "precision": precision,
        "sample_rate": sample_rate,
        "apply_custom_pronunciations": apply_custom_pronunciations,
        "concurrency": concurrency,
    }
    resumed: dict[int, BatchResult] = {}
    append_journal = False
    if resume_path is not None:
        journal, run_id, started_at, resumed, prior_usage = _load_resume(
            resume_path,
            input_digest=input_digest,
            options=options,
            resolved_rows=_resolve_outputs(rows.rows(), out_dir, output_format),
            row_count=rows.count,
            output_format=output_format,
            default_voice=default_voice,
        )
        append_journal = True
        if usage_gate is not None:
            await usage_gate.restore_prior_usage(
                logical_items=prior_usage.logical_items,
                provider_attempts=prior_usage.provider_attempts,
                submitted_characters=prior_usage.submitted_characters,
                ambiguous_attempts=prior_usage.ambiguous_attempts,
            )
    if dry_run:
        results = tuple(
            _validation_result(row, destination, default_voice, output_format)
            for row, destination in _resolve_outputs(
                rows.rows(), out_dir, output_format
            )
        )
        return BatchReport(
            schema_version=1,
            run_id=run_id,
            started_at=started_at,
            ended_at=datetime.now(UTC).isoformat(),
            journal_path=journal,
            results=results,
            aborted=False,
            abort_reason=None,
            usage=await usage_gate.snapshot() if usage_gate is not None else None,
            preflight=preflight,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue[tuple[BatchRow, Path] | None] = asyncio.Queue(
        maxsize=max(1, 2 * concurrency)
    )
    results: dict[int, BatchResult] = {}
    stop = asyncio.Event()
    abort_reason: str | None = None

    async with BatchJournalWriter(journal, append=append_journal) as writer:
        if not append_journal:
            await writer.append_record(
                {
                    "$schema": schema_uri("batch-journal"),
                    "schema_version": 1,
                    "record_type": "header",
                    "run_id": run_id,
                    "input_sha256": input_digest,
                    "yakbox_version": __version__,
                    "timestamp": started_at,
                    "started_at": started_at,
                    "options": options,
                }
            )
        else:
            await writer.append_record(
                {
                    "$schema": schema_uri("batch-journal"),
                    "schema_version": 1,
                    "record_type": "resumed",
                    "run_id": run_id,
                    "yakbox_version": __version__,
                    "timestamp": utc_timestamp(),
                    "resumed_at": utc_timestamp(),
                }
            )

        async def producer() -> None:
            for row, destination in _resolve_outputs(
                rows.rows(), out_dir, output_format
            ):
                if row.index in resumed:
                    result = resumed[row.index]
                    results[row.index] = result
                    await _record(writer, result)
                    if progress is not None:
                        progress(result)
                    continue
                if stop.is_set():
                    result = _not_run(row, destination, output_format, "batch aborted")
                    results[row.index] = result
                    await _record(writer, result)
                    continue
                await queue.put((row, destination))
            for _ in range(concurrency):
                await queue.put(None)

        async def worker() -> None:
            nonlocal abort_reason
            while True:
                item = await queue.get()
                if item is None:
                    return
                row, destination = item
                if stop.is_set():
                    result = _not_run(row, destination, output_format, "batch aborted")
                else:
                    result = await _synthesize_row(
                        row,
                        destination,
                        service,
                        default_voice=default_voice,
                        project_uuid=project_uuid,
                        output_format=output_format,
                        use_hd=use_hd,
                        precision=precision,
                        sample_rate=sample_rate,
                        apply_custom_pronunciations=apply_custom_pronunciations,
                        overwrite=overwrite,
                    )
                    if result.error_code in {
                        "authentication",
                        "hosted_budget_exceeded",
                    }:
                        abort_reason = result.error_message
                        stop.set()
                results[row.index] = result
                await _record(writer, result)
                if progress is not None:
                    progress(result)

        async def record_usage_reservation(
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

        async def finalize_interruption(reason: str) -> None:
            for row, destination in _resolve_outputs(
                rows.rows(), out_dir, output_format
            ):
                if row.index in results:
                    continue
                result = _not_run(row, destination, output_format, reason)
                results[row.index] = result
                await asyncio.shield(_record(writer, result))
            usage = await usage_gate.snapshot() if usage_gate is not None else None
            await _record_interruption(writer, run_id, reason, usage)
            partial = BatchReport(
                schema_version=1,
                run_id=run_id,
                started_at=started_at,
                ended_at=datetime.now(UTC).isoformat(),
                journal_path=journal.resolve(),
                results=tuple(results[row.index] for row in rows.rows()),
                aborted=True,
                abort_reason=reason,
                usage=usage,
                preflight=preflight,
            )
            if write_report:
                atomic_write_json(report, partial.to_dict())

        if usage_gate is not None:
            usage_gate.set_recorder(record_usage_reservation)
        try:
            try:
                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(producer())
                    for _ in range(concurrency):
                        tasks.create_task(worker())
            except asyncio.CancelledError:
                await finalize_interruption("batch cancelled")
                raise
            except Exception as error:
                await finalize_interruption(
                    f"batch stopped after {type(error).__name__}"
                )
                raise
        finally:
            if usage_gate is not None:
                usage_gate.set_recorder(None)
        terminal_usage = await usage_gate.snapshot() if usage_gate is not None else None
        await writer.append_record(
            {
                "$schema": schema_uri("batch-journal"),
                "schema_version": 1,
                "record_type": "complete" if abort_reason is None else "aborted",
                "run_id": run_id,
                "yakbox_version": __version__,
                "timestamp": utc_timestamp(),
                "ended_at": utc_timestamp(),
                "abort_reason": abort_reason,
                "usage": _usage_dict(terminal_usage)
                if terminal_usage is not None
                else None,
            }
        )
    ordered = tuple(results[row.index] for row in rows.rows())
    batch_report = BatchReport(
        schema_version=1,
        run_id=run_id,
        started_at=started_at,
        ended_at=datetime.now(UTC).isoformat(),
        journal_path=journal.resolve(),
        results=ordered,
        aborted=abort_reason is not None,
        abort_reason=abort_reason,
        usage=terminal_usage,
        preflight=preflight,
    )
    if write_report:
        atomic_write_json(report, batch_report.to_dict())
    return batch_report


async def _synthesize_row(
    row: BatchRow,
    destination: Path,
    service: TextToSpeechService,
    *,
    default_voice: str | None,
    project_uuid: str | None,
    output_format: AudioFormat,
    use_hd: bool,
    precision: str | None,
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
            started,
            text_hash,
            request_hash,
            "validation",
            validation,
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
    except HostedBudgetExceeded as error:
        return _error_result(
            row,
            destination,
            output_format,
            started,
            text_hash,
            request_hash,
            "hosted_budget_exceeded",
            str(error),
        )
    except RetryExhaustedError as error:
        return _error_result(
            row,
            destination,
            output_format,
            started,
            text_hash,
            request_hash,
            "retry_exhausted",
            str(error),
            attempts=error.attempts,
        )
    except ProviderError as error:
        code = "authentication" if error.status_code in {401, 403} else "provider"
        return _error_result(
            row,
            destination,
            output_format,
            started,
            text_hash,
            request_hash,
            code,
            str(error),
        )
    except CloudError as error:
        return _error_result(
            row,
            destination,
            output_format,
            started,
            text_hash,
            request_hash,
            type(error).__name__.casefold(),
            str(error),
        )
    except (ArtifactError, ValidationError, OSError) as error:
        return _error_result(
            row,
            destination,
            output_format,
            started,
            text_hash,
            request_hash,
            type(error).__name__.casefold(),
            str(error),
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
            time.monotonic(),
            text_hash,
            request_hash,
            "validation",
            error,
        )
    return _not_run(row, destination, output_format, "dry run")


def _error_result(
    row: BatchRow,
    destination: Path,
    output_format: AudioFormat,
    started: float,
    text_hash: str,
    request_hash: str,
    code: str,
    message: str,
    *,
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
    precision: str | None = None,
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
    resume_path = path.resolve()
    if resume_path.suffix.casefold() == ".json":
        try:
            report = json.loads(resume_path.read_text(encoding="utf-8"))
            resume_path = Path(str(report["journal_path"])).resolve()
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ResumeMismatchError(
                f"Cannot resolve resume report {path}: {error}"
            ) from error
    records = read_journal(resume_path)
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
    successful = {
        index: record
        for record in records[1:]
        if record.get("record_type") == "row"
        and record.get("status") == "ok"
        and isinstance(index := record.get("index"), int)
    }
    attempts_by_index: dict[int, int] = {}
    for record in records[1:]:
        if record.get("record_type") != "row" or record.get("status") == "skipped":
            continue
        index = _integer_or_none(record.get("index"))
        if index is not None:
            attempts_by_index[index] = attempts_by_index.get(index, 0) + (
                _integer_or_none(record.get("attempts")) or 0
            )
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
        project = _string_or_none(options.get("project_uuid"))
        precision = _string_or_none(options.get("precision"))
        sample_rate = _integer_or_none(options.get("sample_rate"))
        request_hash = _request_hash(
            row,
            row.voice_uuid or default_voice,
            project,
            output_format,
            bool(options.get("use_hd", False)),
            precision=precision,
            sample_rate=sample_rate,
            apply_custom_pronunciations=bool(
                options.get("apply_custom_pronunciations", False)
            ),
        )
        if record.get("request_sha256") != request_hash:
            continue
        recorded_path = Path(str(record.get("path", destination))).resolve()
        recorded_size = record.get("bytes_written")
        recorded_digest = record.get("output_sha256")
        if (
            not recorded_path.is_file()
            or not isinstance(recorded_size, int)
            or recorded_path.stat().st_size != recorded_size
            or not isinstance(recorded_digest, str)
            or sha256_file(recorded_path) != recorded_digest
        ):
            continue
        attempts = _integer_or_none(record.get("attempts")) or 0
        request_id = _string_or_none(record.get("request_id"))
        issues = _string_tuple(record.get("issues"))
        completed[row.index] = BatchResult(
            index=row.index,
            row_id=row.row_id,
            status=BatchStatus.SKIPPED,
            path=recorded_path,
            bytes_written=recorded_size,
            output_format=output_format,
            duration_seconds=_number_or_none(record.get("duration_seconds")),
            elapsed_seconds=0,
            attempts=attempts,
            request_id=request_id,
            issues=issues,
            error_code=None,
            error_message=None,
            text_sha256=str(record.get("text_sha256", "")),
            request_sha256=str(record.get("request_sha256", "")),
            output_sha256=recorded_digest,
        )
    for record in reversed(records):
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue
        prior_usage = HostedUsageSnapshot(
            logical_items=_integer_or_none(usage.get("logical_items")) or 0,
            provider_attempts=_integer_or_none(usage.get("provider_attempts")) or 0,
            submitted_characters=_integer_or_none(usage.get("submitted_characters"))
            or 0,
            estimated_spend=None,
            currency=None,
            ambiguous_attempts=_integer_or_none(usage.get("ambiguous_attempts")) or 0,
        )
        return resume_path, run_id, started_at, completed, prior_usage
    prior_usage = HostedUsageSnapshot(
        logical_items=row_count,
        provider_attempts=fallback_attempts,
        submitted_characters=fallback_characters,
        estimated_spend=None,
        currency=None,
        ambiguous_attempts=0,
    )
    return resume_path, run_id, started_at, completed, prior_usage


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
