"""Supervisor for dependency-family-isolated speech-analysis workers."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import os
import sys
import time
import wave
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from yakbox._files import safe_child, sha256_file
from yakbox.errors import SpeechAnalysisError, ValidationError, WorkerProtocolError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_models import (
    AlignmentPurpose,
    AudioSpan,
    ForcedAlignmentResult,
    RecognitionResult,
    VerifiedTextSpan,
)
from yakbox.speech.analysis_protocol import (
    MAXIMUM_WORKER_FRAME_BYTES,
    AnalysisWorkerRequest,
    AnalysisWorkerResponse,
    CancellationWorkerRequest,
    ForcedAlignmentManyWorkerRequest,
    ForcedAlignmentWorkerRequest,
    RecognitionManyWorkerRequest,
    RecognitionWorkerRequest,
    ShutdownWorkerRequest,
    StatusWorkerRequest,
    UnloadWorkerRequest,
    WorkerBatchFinished,
    WorkerErrorCode,
    WorkerFailure,
    WorkerItemFailure,
    WorkerItemSuccess,
    WorkerStatus,
    WorkerSuccess,
    encode_worker_request,
    parse_worker_handshake,
    parse_worker_response,
)
from yakbox.speech.analysis_scheduler import (
    BatchCompletion,
    BatchOperation,
    MemoryWatermark,
    OperationTerminalStatus,
    WorkerHandshake,
    build_worker_handshake,
)
from yakbox.speech.analysis_worker_artifact import worker_artifact_bytes

_ENGINE_FAMILIES: Mapping[str, str] = {
    "whisper": "whisper",
    "parakeet": "parakeet",
    "qwen": "qwen",
    "qwen-forced": "qwen",
}
_PASSTHROUGH_ENVIRONMENT = (
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "PATH",
    "SYSTEMROOT",
    "TMPDIR",
    "WINDIR",
)
_GIB = 1024 * 1024 * 1024
_PCM_S16_SAMPLE_WIDTH = 2
_MINIMUM_WORKER_TIMEOUT_SECONDS = 0.001
_MAXIMUM_WORKER_TIMEOUT_SECONDS = 3_600
_GRACEFUL_SHUTDOWN_SECONDS = 2
_FORCED_TERMINATION_SECONDS = 5


@dataclass(frozen=True, slots=True)
class WorkerPackage:
    """Exact reviewed distribution owned by a built-in worker definition."""

    name: str
    version: str

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValidationError("Analysis worker package constraint is incomplete")


@dataclass(frozen=True, slots=True)
class WorkerDefinition:
    """Trusted runtime definition owned by Yakbox, never by a manifest."""

    family: str
    engines: tuple[str, ...]
    packages: tuple[WorkerPackage, ...]
    maximum_peak_memory_bytes: int
    python_version: str = "3.14"
    module: str = "yakbox.speech.analysis_worker"

    def __post_init__(self) -> None:
        if (
            not self.family
            or not self.engines
            or not self.packages
            or not self.module
            or self.maximum_peak_memory_bytes <= 0
        ):
            raise ValidationError("Analysis worker definition is incomplete")
        if self.python_version != "3.14":
            raise ValidationError("Analysis worker definition requires Python 3.14")
        names = tuple(item.name for item in self.packages)
        if len(names) != len(set(names)):
            raise ValidationError("Analysis worker packages must be unique")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("analysis-worker-definition-v1", self)

    def environment_fingerprint(self, lock_digest: str) -> str:
        return semantic_fingerprint(
            "analysis-worker-environment-v1",
            {
                "definition_fingerprint": self.fingerprint,
                "lock_digest": lock_digest,
            },
        )


BUILT_IN_WORKERS: Mapping[str, WorkerDefinition] = {
    "whisper": WorkerDefinition(
        "whisper",
        ("whisper",),
        (WorkerPackage("mlx-whisper", "0.4.3"),),
        4 * _GIB,
    ),
    "parakeet": WorkerDefinition(
        "parakeet",
        ("parakeet",),
        (WorkerPackage("parakeet-mlx", "0.5.2"),),
        4 * _GIB,
    ),
    "qwen": WorkerDefinition(
        "qwen",
        ("qwen", "qwen-forced"),
        (WorkerPackage("mlx-audio", "0.4.8"),),
        8 * _GIB,
    ),
}


class IsolatedAnalysisWorker:
    """One restartable subprocess serving exactly one dependency family."""

    def __init__(
        self,
        definition: WorkerDefinition,
        *,
        audio_root: Path,
        model_root: Path,
        calibration_fingerprint: str,
        worker_artifact_digest: str,
        lock_digest: str,
        python_executable: Path | None = None,
        worker_artifact_path: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.definition = definition
        self.audio_root = audio_root.resolve()
        self.model_root = model_root.resolve()
        self.calibration_fingerprint = calibration_fingerprint
        self.worker_artifact_digest = worker_artifact_digest
        self.lock_digest = lock_digest
        # Preserve a virtual-environment launcher symlink. Resolving it selects the
        # base interpreter and silently drops the environment's installed packages.
        self.python_executable = (python_executable or Path(sys.executable)).absolute()
        self.worker_artifact_path = (
            worker_artifact_path.absolute()
            if worker_artifact_path is not None
            else None
        )
        self.environment = _offline_environment(environment or os.environ)
        self._process: asyncio.subprocess.Process | None = None
        self._request_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._generation = 0
        self._active_request_id: str | None = None
        self._cancelled_request_ids: set[str] = set()
        self._handshake: WorkerHandshake | None = None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def handshake(self) -> WorkerHandshake | None:
        return self._handshake

    async def start(self) -> WorkerHandshake:
        """Start the exact planned runtime and return its validated handshake."""
        await self._ensure_process()
        if self._handshake is None:
            raise WorkerProtocolError("Analysis worker did not retain its handshake")
        return self._handshake

    async def request(
        self,
        request: AnalysisWorkerRequest,
        *,
        timeout_seconds: float,
        restart_once: bool = True,
    ) -> AnalysisWorkerResponse:
        if timeout_seconds <= 0:
            raise ValidationError("Analysis worker timeout must be positive")
        if isinstance(
            request, RecognitionManyWorkerRequest | ForcedAlignmentManyWorkerRequest
        ):
            raise WorkerProtocolError("Use request_many for a batch operation")
        self._require_allowed_engine(getattr(request, "engine", None))
        attempts = 2 if restart_once else 1
        async with self._request_lock:
            self._active_request_id = request.request_id
            self._idle.clear()
            try:
                for attempt in range(attempts):
                    response = await self._request_attempt(
                        request,
                        timeout_seconds=timeout_seconds,
                        attempt=attempt,
                        attempts=attempts,
                    )
                    if response is not None:
                        return await self._accepted_response(
                            request.request_id, response
                        )
                raise AssertionError("Analysis worker retry loop did not return")
            finally:
                self._active_request_id = None
                self._cancelled_request_ids.discard(request.request_id)
                self._idle.set()

    async def _request_attempt(
        self,
        request: AnalysisWorkerRequest,
        *,
        timeout_seconds: float,
        attempt: int,
        attempts: int,
    ) -> AnalysisWorkerResponse | None:
        try:
            return await asyncio.wait_for(
                self._exchange(request),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            await self._terminate()
            return self._retry_failure(
                request.request_id,
                attempt,
                attempts,
                WorkerErrorCode.TIMEOUT,
                "Analysis worker failed or exceeded its timeout",
            )
        except EOFError, BrokenPipeError:
            await self._terminate()
            return self._retry_failure(
                request.request_id,
                attempt,
                attempts,
                WorkerErrorCode.WORKER_TERMINATED,
                "Analysis worker terminated before returning a result",
            )
        except WorkerProtocolError as error:
            return self._cancelled_protocol_failure(request.request_id, error)

    def _retry_failure(
        self,
        request_id: str,
        attempt: int,
        attempts: int,
        code: WorkerErrorCode,
        detail: str,
    ) -> WorkerFailure | None:
        if request_id in self._cancelled_request_ids:
            return _cancelled_failure(request_id)
        if attempt + 1 < attempts:
            return None
        return WorkerFailure(request_id, code, detail, True)

    def _cancelled_protocol_failure(
        self,
        request_id: str,
        error: WorkerProtocolError,
    ) -> WorkerFailure:
        if request_id not in self._cancelled_request_ids:
            raise error
        return _cancelled_failure(request_id)

    async def _accepted_response(
        self,
        request_id: str,
        response: AnalysisWorkerResponse,
    ) -> AnalysisWorkerResponse:
        if request_id not in self._cancelled_request_ids:
            return response
        await self._terminate()
        return _cancelled_failure(request_id)

    async def request_many(
        self,
        request: RecognitionManyWorkerRequest | ForcedAlignmentManyWorkerRequest,
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[WorkerItemSuccess | WorkerItemFailure]:
        """Stream validated item frames; never replay a partially completed batch."""
        if timeout_seconds <= 0:
            raise ValidationError("Analysis worker timeout must be positive")
        self._require_allowed_engine(request.engine)
        expected = {item.index: item.request_fingerprint for item in request.items}
        completed: set[int] = set()
        finished = False
        async with self._request_lock:
            self._active_request_id = request.request_id
            self._idle.clear()
            deadline = time.monotonic() + timeout_seconds
            try:
                process = await self._ensure_process()
                if process.stdin is None:
                    raise WorkerProtocolError(
                        "Analysis worker input pipe is unavailable"
                    )
                async with self._write_lock:
                    process.stdin.write(encode_worker_request(request) + b"\n")
                    await process.stdin.drain()
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError
                    response = await asyncio.wait_for(
                        self._read_response(process), timeout=remaining
                    )
                    if (
                        request.request_id in self._cancelled_request_ids
                        and isinstance(response, WorkerSuccess)
                        and response.result is None
                    ):
                        continue
                    if _validate_batch_response(request, response, expected, completed):
                        finished = True
                        return
                    item_response = cast(
                        WorkerItemSuccess | WorkerItemFailure, response
                    )
                    completed.add(item_response.index)
                    if request.request_id not in self._cancelled_request_ids:
                        yield item_response
            finally:
                self._active_request_id = None
                cancelled = request.request_id in self._cancelled_request_ids
                self._cancelled_request_ids.discard(request.request_id)
                self._idle.set()
                if not finished or cancelled:
                    await self._terminate()

    def _require_allowed_engine(self, engine: object) -> None:
        if engine is not None and engine not in self.definition.engines:
            raise WorkerProtocolError(
                f"Engine {engine!r} is outside worker family {self.definition.family!r}"
            )

    async def cancel(self, request_id: str) -> WorkerFailure:
        """Cancel safely by terminating the family process and its model thread."""
        if self._active_request_id is not None:
            if self._active_request_id != request_id:
                return WorkerFailure(
                    request_id,
                    WorkerErrorCode.INVALID_REQUEST,
                    "Analysis worker cancellation target is not active",
                    False,
                )
            self._cancelled_request_ids.add(request_id)
        await self._terminate()
        return _cancelled_failure(request_id)

    async def soft_cancel(self, request_id: str) -> WorkerFailure:
        """Send an in-protocol cancellation before any forced termination."""
        if self._active_request_id != request_id:
            return WorkerFailure(
                request_id,
                WorkerErrorCode.INVALID_REQUEST,
                "Analysis worker cancellation target is not active",
                False,
            )
        self._cancelled_request_ids.add(request_id)
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            return _cancelled_failure(request_id)
        cancellation = CancellationWorkerRequest(request_id, request_id)
        try:
            async with self._write_lock:
                process.stdin.write(encode_worker_request(cancellation) + b"\n")
                await process.stdin.drain()
        except BrokenPipeError, ConnectionResetError:
            return _cancelled_failure(request_id)
        return _cancelled_failure(request_id)

    async def wait_idle(self) -> None:
        await self._idle.wait()

    async def close(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            await self._terminate()
            return
        if self._active_request_id is not None:
            await self._terminate()
            return
        request = ShutdownWorkerRequest(f"shutdown-{self._generation}")
        try:
            response = await asyncio.wait_for(
                self._exchange(request),
                timeout=_GRACEFUL_SHUTDOWN_SECONDS,
            )
            if not isinstance(response, WorkerSuccess) or response.result is not None:
                raise WorkerProtocolError(
                    "Analysis worker graceful shutdown response is invalid"
                )
            await asyncio.wait_for(
                process.wait(),
                timeout=_GRACEFUL_SHUTDOWN_SECONDS,
            )
        except EOFError, BrokenPipeError, TimeoutError, WorkerProtocolError:
            await self._terminate()
            return
        if self._process is process:
            self._process = None
            self._handshake = None

    async def _exchange(self, request: AnalysisWorkerRequest) -> AnalysisWorkerResponse:
        process = await self._ensure_process()
        if process.stdin is None or process.stdout is None:
            raise WorkerProtocolError("Analysis worker pipes are unavailable")
        async with self._write_lock:
            process.stdin.write(encode_worker_request(request) + b"\n")
            await process.stdin.drain()
        while True:
            response = await self._read_response(process)
            if (
                request.request_id in self._cancelled_request_ids
                and isinstance(response, WorkerSuccess)
                and response.result is None
            ):
                continue
            break
        if response.request_id != request.request_id:
            raise WorkerProtocolError("Analysis worker response request ID differs")
        return response

    async def _read_response(
        self, process: asyncio.subprocess.Process
    ) -> AnalysisWorkerResponse:
        if process.stdout is None:
            raise WorkerProtocolError("Analysis worker output pipe is unavailable")
        try:
            frame = await process.stdout.readuntil(b"\n")
        except asyncio.IncompleteReadError as error:
            raise EOFError from error
        except asyncio.LimitOverrunError as error:
            raise WorkerProtocolError(
                "Analysis worker response is too large"
            ) from error
        if len(frame) > MAXIMUM_WORKER_FRAME_BYTES + 1:
            raise WorkerProtocolError("Analysis worker response is too large")
        return parse_worker_response(frame[:-1])

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        process = self._process
        if process is not None and process.returncode is None:
            return process
        self.audio_root.mkdir(parents=True, exist_ok=True)
        self.model_root.mkdir(parents=True, exist_ok=True)
        worker_entrypoint = self._worker_entrypoint()
        command = (
            str(self.python_executable),
            "-I",
            *worker_entrypoint,
            "--family",
            self.definition.family,
            "--audio-root",
            str(self.audio_root),
            "--model-root",
            str(self.model_root),
            "--calibration-fingerprint",
            self.calibration_fingerprint,
            "--worker-artifact-digest",
            self.worker_artifact_digest,
            "--lock-digest",
            self.lock_digest,
            "--definition-fingerprint",
            self.definition.fingerprint,
        )
        self._process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=self.environment,
            limit=MAXIMUM_WORKER_FRAME_BYTES + 2,
        )
        self._generation += 1
        if self._process.stdout is None:
            await self._terminate()
            raise WorkerProtocolError("Analysis worker handshake pipe is unavailable")
        try:
            frame = await self._process.stdout.readuntil(b"\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError) as error:
            await self._terminate()
            raise WorkerProtocolError(
                "Analysis worker handshake is incomplete"
            ) from error
        actual = parse_worker_handshake(frame[:-1])
        planned = build_worker_handshake(
            family=self.definition.family,
            engines=self.definition.engines,
            worker_artifact_fingerprint=self.worker_artifact_digest,
            environment_lock_fingerprint=self.lock_digest,
            adapter_fingerprint=self.definition.fingerprint,
        )
        if actual != planned:
            await self._terminate()
            raise WorkerProtocolError(
                "Analysis worker handshake differs from its planned identity"
            )
        self._handshake = actual
        return self._process

    def _worker_entrypoint(self) -> tuple[str, ...]:
        artifact = self.worker_artifact_path
        if artifact is None:
            return ("-m", self.definition.module)
        if artifact.is_symlink() or not artifact.is_file():
            raise WorkerProtocolError(
                "Analysis worker artifact must be a regular non-symlink file"
            )
        if sha256_file(artifact) != self.worker_artifact_digest:
            raise WorkerProtocolError("Analysis worker artifact digest differs")
        return (str(artifact),)

    async def _terminate(self) -> None:
        process, self._process = self._process, None
        self._handshake = None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=_FORCED_TERMINATION_SECONDS,
            )
        except TimeoutError:
            process.kill()
            await process.wait()


def _validate_batch_response(
    request: RecognitionManyWorkerRequest | ForcedAlignmentManyWorkerRequest,
    response: AnalysisWorkerResponse,
    expected: Mapping[int, str],
    completed: set[int],
) -> bool:
    if response.request_id != request.request_id:
        raise WorkerProtocolError("Analysis worker response request ID differs")
    if isinstance(response, WorkerBatchFinished):
        if (
            response.batch_id != request.batch_id
            or response.item_count != len(request.items)
            or completed != set(expected)
        ):
            raise WorkerProtocolError(
                "Analysis worker batch terminal frame is incomplete"
            )
        return True
    if not isinstance(response, WorkerItemSuccess | WorkerItemFailure):
        raise WorkerProtocolError("Analysis worker returned a non-item batch frame")
    if (
        response.batch_id != request.batch_id
        or expected.get(response.index) != response.request_fingerprint
    ):
        raise WorkerProtocolError("Analysis worker batch item identity differs")
    return False


def _cancelled_failure(request_id: str) -> WorkerFailure:
    return WorkerFailure(
        request_id,
        WorkerErrorCode.CANCELLED,
        "Analysis worker request was cancelled",
        True,
    )


type BatchProtocolRequest = (
    RecognitionManyWorkerRequest | ForcedAlignmentManyWorkerRequest
)
type BatchRequestFactory = Callable[[BatchOperation], BatchProtocolRequest]


class ProtocolSupervisedWorker:
    """Bridge the abstract supervisor to the concrete framed worker protocol."""

    def __init__(
        self,
        worker: IsolatedAnalysisWorker,
        *,
        request_factory: BatchRequestFactory,
    ) -> None:
        self.worker = worker
        self._request_factory = request_factory
        self._identifiers = itertools.count(1)

    async def handshake(self) -> WorkerHandshake:
        return await self.worker.start()

    async def execute(
        self, operation: BatchOperation
    ) -> AsyncIterator[BatchCompletion]:
        request = self._request(operation)
        timeout = operation.remaining_seconds()
        if timeout <= 0:
            raise TimeoutError
        async for response in self.worker.request_many(
            request,
            timeout_seconds=timeout,
        ):
            if isinstance(response, WorkerItemSuccess):
                yield BatchCompletion(
                    response.batch_id,
                    response.index,
                    response.request_fingerprint,
                    OperationTerminalStatus.COMPLETED,
                    response.result.fingerprint,
                    None,
                    response.metrics,
                )
            else:
                yield BatchCompletion(
                    response.batch_id,
                    response.index,
                    response.request_fingerprint,
                    response.terminal_status,
                    None,
                    response.code.value,
                    response.metrics,
                )

    async def soft_cancel(self, operation_id: str) -> None:
        response = await self.worker.soft_cancel(operation_id)
        if response.code not in {
            WorkerErrorCode.CANCELLED,
            WorkerErrorCode.INVALID_REQUEST,
        }:
            raise WorkerProtocolError(
                "Analysis worker cancellation response is invalid"
            )

    async def wait_idle(self) -> None:
        await self.worker.wait_idle()

    async def terminate(self) -> None:
        await self.worker.close()

    async def unload(self, engine: str) -> MemoryWatermark:
        suffix = next(self._identifiers)
        unloaded = await self.worker.request(
            UnloadWorkerRequest(f"unload-{suffix}", engine),
            timeout_seconds=30,
            restart_once=False,
        )
        if not isinstance(unloaded, WorkerSuccess) or unloaded.result is not None:
            raise WorkerProtocolError("Analysis worker unload failed")
        return await self.current_memory()

    async def current_memory(self) -> MemoryWatermark:
        suffix = next(self._identifiers)
        status_response = await self.worker.request(
            StatusWorkerRequest(f"status-{suffix}"),
            timeout_seconds=30,
            restart_once=False,
        )
        if not isinstance(status_response, WorkerSuccess) or not isinstance(
            status_response.result, WorkerStatus
        ):
            raise WorkerProtocolError("Analysis worker memory status failed")
        return MemoryWatermark(
            status_response.result.resident_memory_bytes,
            status_response.result.metal_active_memory_bytes,
        )

    def _request(self, operation: BatchOperation) -> BatchProtocolRequest:
        request = self._request_factory(operation)
        expected_type = (
            RecognitionManyWorkerRequest
            if operation.operation == "recognize_many"
            else ForcedAlignmentManyWorkerRequest
        )
        if not isinstance(request, expected_type):
            raise WorkerProtocolError(
                "Analysis batch request type differs from its planned operation"
            )
        expected_items = tuple(
            (item.index, item.request_fingerprint) for item in operation.items
        )
        actual_items = tuple(
            (item.index, item.request_fingerprint) for item in request.items
        )
        if actual_items != expected_items:
            raise WorkerProtocolError(
                "Analysis protocol batch items differ from the supervisor plan"
            )
        remaining_ms = min(
            request.timeout_milliseconds,
            max(1, round(operation.remaining_seconds() * 1_000)),
        )
        return replace(
            request,
            request_id=operation.operation_id,
            batch_id=operation.batch_id,
            engine=operation.engine,
            engine_fingerprint=operation.engine_fingerprint,
            timeout_milliseconds=remaining_ms,
        )


class AnalysisWorkerPool:
    """Coordinator that isolates failures and model state by dependency family."""

    def __init__(self, workers: Mapping[str, IsolatedAnalysisWorker]) -> None:
        if set(workers) != set(BUILT_IN_WORKERS):
            raise ValidationError("Analysis worker pool requires every built-in family")
        self.workers = dict(workers)

    def for_engine(self, engine: str) -> IsolatedAnalysisWorker:
        try:
            return self.workers[_ENGINE_FAMILIES[engine]]
        except KeyError as error:
            raise ValidationError(
                f"Unknown speech-analysis engine: {engine}"
            ) from error

    async def close(self) -> None:
        await asyncio.gather(*(worker.close() for worker in self.workers.values()))


class WorkerBackedSpeechRecognizer:
    """SpeechRecognizer client for one trusted isolated worker adapter."""

    def __init__(
        self,
        *,
        engine: str,
        worker: IsolatedAnalysisWorker,
        audio_root: Path,
        adapter_fingerprint: str,
        timeout_seconds: float,
    ) -> None:
        if engine not in worker.definition.engines:
            raise ValidationError("Recognizer engine does not match worker family")
        _validate_worker_timeout(timeout_seconds)
        self.engine = engine
        self.worker = worker
        self.audio_root = audio_root.resolve()
        self.adapter_fingerprint = adapter_fingerprint
        self.timeout_seconds = timeout_seconds
        self._identifiers = itertools.count(1)

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "worker-backed-recognizer-v2",
            {
                "engine": self.engine,
                "worker_family": self.worker.definition.family,
                "adapter_fingerprint": self.adapter_fingerprint,
                "worker_artifact_digest": self.worker.worker_artifact_digest,
                "worker_lock_digest": self.worker.lock_digest,
            },
        )

    async def recognize(
        self,
        audio: Path,
        *,
        language: str,
        span: AudioSpan | None = None,
    ) -> RecognitionResult:
        relative, digest, materialized_span = _worker_audio_identity(
            self.audio_root,
            audio,
        )
        _require_materialized_span(span, materialized_span)
        request = RecognitionWorkerRequest(
            request_id=f"recognize-{next(self._identifiers)}",
            engine=self.engine,
            engine_fingerprint=self.adapter_fingerprint,
            relative_audio_path=relative,
            audio_digest=digest,
            language=language,
            span=materialized_span,
            timeout_milliseconds=round(self.timeout_seconds * 1_000),
        )
        response = await self.worker.request(
            request,
            timeout_seconds=self.timeout_seconds,
        )
        if isinstance(response, WorkerFailure):
            raise SpeechAnalysisError(
                f"Isolated {self.engine!r} recognition failed: {response.code.value}"
            )
        if not isinstance(response, WorkerSuccess):
            raise WorkerProtocolError("Worker returned a batch frame for recognition")
        if not isinstance(response.result, RecognitionResult):
            raise WorkerProtocolError("Worker did not return recognition evidence")
        return response.result


class WorkerBackedForcedAligner:
    """ForcedAligner client preserving explicit expected-text authority."""

    def __init__(
        self,
        *,
        engine: str,
        worker: IsolatedAnalysisWorker,
        audio_root: Path,
        adapter_fingerprint: str,
        timeout_seconds: float,
    ) -> None:
        if engine not in worker.definition.engines:
            raise ValidationError("Forced-aligner engine does not match worker family")
        _validate_worker_timeout(timeout_seconds)
        self.engine = engine
        self.worker = worker
        self.audio_root = audio_root.resolve()
        self.adapter_fingerprint = adapter_fingerprint
        self.timeout_seconds = timeout_seconds
        self._identifiers = itertools.count(1)

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "worker-backed-forced-aligner-v2",
            {
                "engine": self.engine,
                "worker_family": self.worker.definition.family,
                "adapter_fingerprint": self.adapter_fingerprint,
                "worker_artifact_digest": self.worker.worker_artifact_digest,
                "worker_lock_digest": self.worker.lock_digest,
            },
        )

    async def force_align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
        purpose: AlignmentPurpose,
        verified_span: VerifiedTextSpan | None = None,
        span: AudioSpan | None = None,
    ) -> ForcedAlignmentResult:
        relative, digest, materialized_span = _worker_audio_identity(
            self.audio_root,
            audio,
        )
        _require_materialized_span(span, materialized_span)
        request = ForcedAlignmentWorkerRequest(
            request_id=f"force-align-{next(self._identifiers)}",
            engine=self.engine,
            engine_fingerprint=self.adapter_fingerprint,
            relative_audio_path=relative,
            audio_digest=digest,
            expected_text=expected_text,
            language=language,
            purpose=purpose,
            span=materialized_span,
            verified_span=verified_span,
            timeout_milliseconds=round(self.timeout_seconds * 1_000),
        )
        response = await self.worker.request(
            request,
            timeout_seconds=self.timeout_seconds,
        )
        if isinstance(response, WorkerFailure):
            raise SpeechAnalysisError(
                f"Isolated {self.engine!r} alignment failed: {response.code.value}"
            )
        if not isinstance(response, WorkerSuccess):
            raise WorkerProtocolError("Worker returned a batch frame for alignment")
        if not isinstance(response.result, ForcedAlignmentResult):
            raise WorkerProtocolError("Worker did not return forced-alignment evidence")
        return response.result


def _worker_audio_identity(
    audio_root: Path,
    audio: Path,
) -> tuple[str, str, AudioSpan]:
    candidate = audio if audio.is_absolute() else audio_root / audio
    if candidate.is_symlink():
        raise ValidationError("Analysis worker client rejects symlink audio")
    path = safe_child(audio_root, candidate)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValidationError("Analysis worker client requires a regular audio file")
    if path.suffix.casefold() != ".wav":
        raise ValidationError("Analysis worker client requires canonical WAV audio")
    try:
        with wave.open(str(path), "rb") as reader:
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != _PCM_S16_SAMPLE_WIDTH
            ):
                raise ValidationError(
                    "Analysis worker client WAV must be mono 16-bit PCM"
                )
            frame_count = reader.getnframes()
            sample_rate = reader.getframerate()
    except (EOFError, OSError, wave.Error) as error:
        raise ValidationError("Analysis worker client WAV is invalid") from error
    digest = sha256_file(path)
    return (
        path.relative_to(audio_root).as_posix(),
        digest,
        AudioSpan(digest, 0, frame_count, sample_rate),
    )


def _require_materialized_span(
    requested: AudioSpan | None,
    materialized: AudioSpan,
) -> None:
    if requested is not None and requested != materialized:
        raise ValidationError(
            "Analysis worker requires a materialized WAV matching the exact span"
        )


def _validate_worker_timeout(timeout_seconds: float) -> None:
    if not (
        _MINIMUM_WORKER_TIMEOUT_SECONDS
        <= timeout_seconds
        <= _MAXIMUM_WORKER_TIMEOUT_SECONDS
    ):
        raise ValidationError("Analysis worker timeout is outside protocol bounds")


def worker_artifact_digest() -> str:
    """Fingerprint the complete reproducible worker artifact."""
    return hashlib.sha256(worker_artifact_bytes()).hexdigest()


def _offline_environment(source: Mapping[str, str]) -> dict[str, str]:
    environment = {
        key: source[key] for key in _PASSTHROUGH_ENVIRONMENT if key in source
    }
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    return environment


__all__ = [
    "BUILT_IN_WORKERS",
    "AnalysisWorkerPool",
    "BatchRequestFactory",
    "IsolatedAnalysisWorker",
    "ProtocolSupervisedWorker",
    "WorkerBackedForcedAligner",
    "WorkerBackedSpeechRecognizer",
    "WorkerDefinition",
    "WorkerPackage",
    "worker_artifact_digest",
]
