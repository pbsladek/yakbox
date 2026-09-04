"""Versioned isolated worker protocol for local model execution."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import redirect_stdout, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

from yakbox._files import atomic_write_json
from yakbox.errors import BackendUnavailableError, BuildError, ValidationError
from yakbox.speech.accelerator import AcceleratorLease, accelerator_operation
from yakbox.speech.capabilities import BackendCapabilities
from yakbox.speech.models import (
    AudioFormat,
    ChatterboxSynthesisOptions,
    SpeechArtifact,
    SpeechSynthesisRequest,
)

WORKER_ARGUMENT_COUNT = 3
WORKER_PROTOCOL_VERSION = 2


@dataclass(frozen=True, slots=True)
class LocalWorkerItem:
    text: str
    voice: str
    destination: Path
    sample_rate: int | None
    reference_audio: Path | None
    chatterbox: ChatterboxSynthesisOptions | None

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "voice": self.voice,
            "destination": str(self.destination),
            "sample_rate": self.sample_rate,
            "reference_audio": str(self.reference_audio)
            if self.reference_audio
            else None,
            "chatterbox": (
                {
                    "cfg_weight": self.chatterbox.cfg_weight,
                    "exaggeration": self.chatterbox.exaggeration,
                    "seed": self.chatterbox.seed,
                }
                if self.chatterbox is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class LocalWorkerRequest:
    schema_version: int
    operation: str
    device: str
    items: tuple[LocalWorkerItem, ...]
    overwrite: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "device": self.device,
            "items": [item.to_dict() for item in self.items],
            "overwrite": self.overwrite,
        }


class IsolatedLocalSpeechService:
    """Runs each local synthesis operation in a cancellable child process."""

    capabilities = BackendCapabilities(
        name="chatterbox-local",
        synthesis=True,
        transformation=False,
        streaming=False,
        hosted=False,
        output_formats=("wav",),
        supports_reference_voice=True,
    )

    def __init__(
        self,
        *,
        device: str = "cpu",
        timeout_seconds: float = 3_600,
        threads_per_process: int = 1,
        heartbeat_seconds: float = 15,
        log_path: Path | None = None,
        accelerator_lease: AcceleratorLease | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValidationError("Local worker timeout must be positive")
        if threads_per_process < 1:
            raise ValidationError("Local worker thread budget must be positive")
        if heartbeat_seconds <= 0:
            raise ValidationError("Local worker heartbeat interval must be positive")
        self.device = device
        self.timeout_seconds = timeout_seconds
        self.threads_per_process = threads_per_process
        self.heartbeat_seconds = heartbeat_seconds
        self.log_path = log_path
        self.accelerator_lease = accelerator_lease
        self._process: asyncio.subprocess.Process | None = None
        self._request_lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = b""

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        artifacts = await self.synthesize_many_to_files(
            ((request, destination),),
            overwrite=overwrite,
        )
        return artifacts[0]

    async def synthesize_many_to_files(
        self,
        requests: tuple[tuple[SpeechSynthesisRequest, Path], ...],
        *,
        overwrite: bool = False,
    ) -> tuple[SpeechArtifact, ...]:
        if not requests:
            return ()
        items = tuple(
            LocalWorkerItem(
                text=request.text,
                voice=request.voice,
                destination=destination.resolve(),
                sample_rate=request.sample_rate,
                reference_audio=request.reference_audio,
                chatterbox=request.chatterbox,
            )
            for request, destination in requests
        )
        for item in items:
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            _remove_stale_part_files(item.destination)
        worker_request = LocalWorkerRequest(
            schema_version=1,
            operation="synthesize_many",
            device=self.device,
            items=items,
            overwrite=overwrite,
        )
        results = await self._run_worker(worker_request)
        if len(results) != len(items):
            raise BuildError("Local Chatterbox worker returned the wrong result count")
        return tuple(_artifact_from_result(result) for result in results)

    async def _run_worker(
        self,
        worker_request: LocalWorkerRequest,
    ) -> tuple[dict[str, object], ...]:
        async with (
            self._request_lock,
            accelerator_operation(
                self.accelerator_lease,
                owner="tts:chatterbox",
                enabled=self.device.casefold() != "cpu",
            ),
        ):
            return await self._run_worker_with_lease(worker_request)

    async def _run_worker_with_lease(
        self,
        worker_request: LocalWorkerRequest,
    ) -> tuple[dict[str, object], ...]:
        process = await self._ensure_worker()
        if process.stdin is None or process.stdout is None:
            await self._discard_worker(process)
            raise BuildError("Local Chatterbox worker pipes are unavailable")
        payload = {
            "protocol_version": WORKER_PROTOCOL_VERSION,
            **worker_request.to_dict(),
        }
        try:
            process.stdin.write(
                json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
            )
            await process.stdin.drain()
            results = await self._read_batch_responses(
                process, len(worker_request.items)
            )
        except asyncio.CancelledError:
            self._log(f"worker_cancelled pid={process.pid}")
            await self._discard_worker(process)
            _cleanup_worker_parts(worker_request.items)
            raise
        except BuildError:
            _cleanup_worker_parts(worker_request.items)
            raise
        except (BrokenPipeError, ConnectionError, OSError, ValueError) as error:
            await self._discard_worker(process)
            _cleanup_worker_parts(worker_request.items)
            detail = self._stderr_tail.decode(errors="replace").strip()[-2048:]
            raise BuildError(
                f"Local Chatterbox worker connection failed: {detail or error}"
            ) from error
        self._log(f"worker_request_completed pid={process.pid} items={len(results)}")
        return results

    async def _read_batch_responses(
        self, process: asyncio.subprocess.Process, expected: int
    ) -> tuple[dict[str, object], ...]:
        results: list[dict[str, object]] = []
        while True:
            response = await self._read_response(process)
            event = response.get("event")
            if event == "started":
                self._log(
                    f"worker_item_started pid={process.pid} "
                    f"index={response.get('index')}"
                )
                continue
            if event == "result":
                result = response.get("item")
                if not isinstance(result, dict):
                    raise BuildError("Invalid local worker item response")
                results.append(cast(dict[str, object], result))
                continue
            if event == "error":
                detail = str(response.get("error", "worker request failed"))[:512]
                self._log(f"worker_request_failed pid={process.pid} detail={detail}")
                raise BuildError(f"Local Chatterbox worker failed: {detail}")
            if event == "complete" and len(results) == expected:
                return tuple(results)
            raise BuildError("Invalid local worker protocol response")

    async def aclose(self) -> None:
        async with self._request_lock:
            process = self._process
            if process is None:
                return
            if process.returncode is None and process.stdin is not None:
                self._log(f"worker_shutdown pid={process.pid}")
                try:
                    process.stdin.write(b'{"operation":"shutdown"}\n')
                    await process.stdin.drain()
                    await asyncio.wait_for(process.wait(), timeout=5)
                except BrokenPipeError, OSError, TimeoutError:
                    await _terminate_process(process)
            await self._finish_stderr_task()
            self._process = None

    async def _ensure_worker(self) -> asyncio.subprocess.Process:
        process = self._process
        if process is not None and process.returncode is None:
            return process
        await self._finish_stderr_task()
        environment = os.environ.copy()
        environment.pop("RESEMBLE_API_KEY", None)
        thread_budget = str(self.threads_per_process)
        environment["OMP_NUM_THREADS"] = thread_budget
        environment["MKL_NUM_THREADS"] = thread_budget
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "yakbox.speech.workers",
            "--serve",
            self.device,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
        self._process = process
        self._stderr_tail = b""
        if process.stderr is not None:
            self._stderr_task = asyncio.create_task(
                self._capture_stderr(process.stderr)
            )
        self._log(
            f"worker_started pid={process.pid} device={self.device} "
            f"threads={self.threads_per_process}"
        )
        return process

    async def _read_response(
        self, process: asyncio.subprocess.Process
    ) -> dict[str, object]:
        if process.stdout is None:
            raise BuildError("Local Chatterbox worker stdout is unavailable")
        response = asyncio.create_task(process.stdout.readline())
        started = time.monotonic()
        while True:
            remaining = self.timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                response.cancel()
                await self._discard_worker(process)
                with suppress(asyncio.CancelledError):
                    await response
                self._log(f"worker_timed_out pid={process.pid}")
                raise BuildError(
                    "Local Chatterbox worker item exceeded "
                    f"{self.timeout_seconds:g}s without progress"
                )
            done, _ = await asyncio.wait(
                {response},
                timeout=min(self.heartbeat_seconds, remaining),
            )
            if response in done:
                line = response.result()
                if not line:
                    await self._discard_worker(process)
                    detail = self._stderr_tail.decode(errors="replace").strip()[-2048:]
                    raise BuildError(
                        "Local Chatterbox worker stopped unexpectedly"
                        + (f": {detail}" if detail else "")
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    await self._discard_worker(process)
                    raise BuildError(
                        "Invalid local worker protocol response"
                    ) from error
                if not isinstance(value, dict):
                    raise BuildError("Invalid local worker protocol response")
                if value.get("protocol_version") != WORKER_PROTOCOL_VERSION:
                    raise BuildError("Unsupported local worker protocol response")
                return value
            self._log(
                f"worker_heartbeat pid={process.pid} "
                f"elapsed={time.monotonic() - started:.1f}s"
            )

    async def _capture_stderr(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(4096):
            self._stderr_tail = (self._stderr_tail + chunk)[-8192:]
            detail = chunk.decode(errors="replace").strip()
            if detail:
                self._log(f"worker_stderr {detail[-2048:]}")

    async def _discard_worker(self, process: asyncio.subprocess.Process) -> None:
        await _terminate_process(process)
        await self._finish_stderr_task()
        if self._process is process:
            self._process = None

    async def _finish_stderr_task(self) -> None:
        task = self._stderr_task
        if task is not None:
            if not task.done():
                try:
                    await asyncio.wait_for(task, timeout=1)
                except TimeoutError:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
            else:
                task.result()
        self._stderr_task = None

    def _log(self, message: str) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{time.time():.3f} {message}\n")
            stream.flush()


async def _worker_main(request_path: Path, result_path: Path) -> None:
    from yakbox.local import (  # noqa: PLC0415 - loaded only in child process
        LocalChatterboxService,
    )

    request = _read_request(request_path)
    service = LocalChatterboxService(device=request.device)
    artifacts = [
        await service.synthesize_to_file(
            SpeechSynthesisRequest(
                text=item.text,
                voice=item.voice,
                backend="chatterbox-local",
                output_format=AudioFormat.WAV,
                sample_rate=item.sample_rate,
                reference_audio=item.reference_audio,
                chatterbox=item.chatterbox,
            ),
            item.destination,
            overwrite=request.overwrite,
        )
        for item in request.items
    ]
    atomic_write_json(
        result_path,
        {
            "schema_version": 1,
            "items": [_artifact_result(artifact) for artifact in artifacts],
        },
    )


async def _worker_server(device: str) -> None:
    protocol_output = sys.stdout
    with redirect_stdout(sys.stderr):
        from yakbox.local import (  # noqa: PLC0415 - loaded only in child process
            LocalChatterboxService,
        )

        service = LocalChatterboxService(device=device)
    while line := sys.stdin.buffer.readline():
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValidationError("Invalid local worker protocol request")
            if raw.get("operation") == "shutdown":
                return
            if raw.get("protocol_version") != WORKER_PROTOCOL_VERSION:
                raise ValidationError("Unsupported local worker protocol")
            request = _request_from_raw(raw)
            for index, item in enumerate(request.items):
                _write_worker_message(
                    protocol_output,
                    {"event": "started", "index": index},
                )
                with redirect_stdout(sys.stderr):
                    artifact = await service.synthesize_to_file(
                        SpeechSynthesisRequest(
                            text=item.text,
                            voice=item.voice,
                            backend="chatterbox-local",
                            output_format=AudioFormat.WAV,
                            sample_rate=item.sample_rate,
                            reference_audio=item.reference_audio,
                            chatterbox=item.chatterbox,
                        ),
                        item.destination,
                        overwrite=request.overwrite,
                    )
                _write_worker_message(
                    protocol_output,
                    {
                        "event": "result",
                        "index": index,
                        "item": _artifact_result(artifact),
                    },
                )
            _write_worker_message(
                protocol_output,
                {"event": "complete", "count": len(request.items)},
            )
        except Exception as error:  # noqa: BLE001 - worker protocol returns safe errors
            _write_worker_message(
                protocol_output,
                {
                    "event": "error",
                    "error": f"{type(error).__name__}: local synthesis failed",
                },
            )


def _write_worker_message(stream: TextIO, payload: dict[str, object]) -> None:
    message = {"protocol_version": WORKER_PROTOCOL_VERSION, **payload}
    stream.write(json.dumps(message, ensure_ascii=False) + "\n")
    stream.flush()


def _cleanup_worker_parts(items: tuple[LocalWorkerItem, ...]) -> None:
    for item in items:
        _remove_stale_part_files(item.destination)


def _remove_stale_part_files(destination: Path) -> None:
    suffixes = (f".part{destination.suffix}", ".part")
    for suffix in suffixes:
        pattern = f".{destination.name}.*{suffix}"
        for candidate in destination.parent.glob(pattern):
            if candidate.is_file():
                candidate.unlink(missing_ok=True)


def _read_request(path: Path) -> LocalWorkerRequest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"Invalid local worker request: {error}") from error
    if not isinstance(raw, dict):
        raise ValidationError("Unsupported local worker request")
    return _request_from_raw(raw)


def _request_from_raw(raw: dict[str, object]) -> LocalWorkerRequest:
    if raw.get("schema_version") != 1:
        raise ValidationError("Unsupported local worker request")
    if raw.get("operation") != "synthesize_many":
        raise ValidationError("Unsupported local worker operation")
    items_raw = raw.get("items")
    if not isinstance(items_raw, list) or not items_raw:
        raise ValidationError("Local worker request requires synthesis items")
    items = tuple(_read_worker_item(item) for item in items_raw)
    return LocalWorkerRequest(
        schema_version=1,
        operation="synthesize_many",
        device=str(raw.get("device", "cpu")),
        items=items,
        overwrite=bool(raw.get("overwrite", False)),
    )


def _read_worker_item(value: object) -> LocalWorkerItem:
    if not isinstance(value, dict):
        raise ValidationError("Local worker synthesis item is invalid")
    text = value.get("text")
    voice = value.get("voice")
    destination = value.get("destination")
    if not all(isinstance(item, str) for item in (text, voice, destination)):
        raise ValidationError("Local worker request fields are invalid")
    reference = value.get("reference_audio")
    chatterbox_raw = value.get("chatterbox")
    chatterbox: ChatterboxSynthesisOptions | None = None
    if chatterbox_raw is not None:
        if not isinstance(chatterbox_raw, dict):
            raise ValidationError("Local worker Chatterbox options are invalid")
        chatterbox = ChatterboxSynthesisOptions(
            cfg_weight=_float_or_none(chatterbox_raw.get("cfg_weight")),
            exaggeration=_float_or_none(chatterbox_raw.get("exaggeration")),
            seed=_int_or_none(chatterbox_raw.get("seed")),
        )
    return LocalWorkerItem(
        text=str(text),
        voice=str(voice),
        destination=Path(str(destination)),
        sample_rate=_int_or_none(value.get("sample_rate")),
        reference_audio=Path(reference) if isinstance(reference, str) else None,
        chatterbox=chatterbox,
    )


def _read_results(path: Path) -> tuple[dict[str, object], ...]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BuildError(f"Invalid local worker result: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise BuildError("Unsupported local worker result")
    items = raw.get("items")
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise BuildError("Local worker result items are invalid")
    return tuple(items)


def _artifact_result(artifact: SpeechArtifact) -> dict[str, object]:
    return {
        "path": str(artifact.path),
        "backend": artifact.backend,
        "voice": artifact.voice,
        "output_format": artifact.output_format.value,
        "bytes_written": artifact.bytes_written,
        "sha256": artifact.sha256,
        "duration_seconds": artifact.duration_seconds,
        "sample_rate": artifact.sample_rate,
    }


def _artifact_from_result(result: dict[str, object]) -> SpeechArtifact:
    return SpeechArtifact(
        path=Path(str(result["path"])),
        backend=str(result["backend"]),
        voice=str(result["voice"]),
        output_format=AudioFormat(str(result["output_format"])),
        bytes_written=_required_int(result, "bytes_written"),
        sha256=str(result["sha256"]),
        duration_seconds=_float_or_none(result.get("duration_seconds")),
        sample_rate=_int_or_none(result.get("sample_rate")),
    )


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _required_int(value: dict[str, object], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int):
        raise BuildError(f"Local worker result field {key!r} must be an integer")
    return item


def _float_or_none(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        process.kill()
        await process.wait()


def main() -> None:
    if len(sys.argv) != WORKER_ARGUMENT_COUNT:
        raise SystemExit(
            "usage: python -m yakbox.speech.workers REQUEST RESULT | --serve DEVICE"
        )
    try:
        if sys.argv[1] == "--serve":
            asyncio.run(_worker_server(sys.argv[2]))
        else:
            asyncio.run(_worker_main(Path(sys.argv[1]), Path(sys.argv[2])))
    except (BackendUnavailableError, BuildError, ValidationError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
