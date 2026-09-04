"""Isolated process entry point for one built-in analysis dependency family."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import gc
import os
import platform
import socket
import sys
import time
import wave
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import distributions
from pathlib import Path
from typing import Protocol, cast

from yakbox._files import safe_child, sha256_file
from yakbox.errors import (
    BackendUnavailableError,
    ModelIntegrityError,
    ValidationError,
    WorkerProtocolError,
)
from yakbox.speech.analysis_adapters import (
    MlxAudioQwenForcedAligner,
    MlxAudioQwenRecognizer,
    MlxWhisperRecognizer,
    ParakeetMlxRecognizer,
)
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_models import (
    AudioSpan,
    ForcedAlignmentResult,
    RecognitionResult,
)
from yakbox.speech.analysis_protocol import (
    MAXIMUM_WORKER_FRAME_BYTES,
    AnalysisWorkerRequest,
    AnalysisWorkerResponse,
    CancellationWorkerRequest,
    ForcedAlignmentBatchItem,
    ForcedAlignmentManyWorkerRequest,
    ForcedAlignmentWorkerRequest,
    RecognitionBatchItem,
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
    WorkerModelLoad,
    WorkerStatus,
    WorkerSuccess,
    encode_worker_handshake,
    encode_worker_response,
    parse_worker_request,
)
from yakbox.speech.analysis_runtime_identity import execution_identity_from_digests
from yakbox.speech.analysis_scheduler import (
    OperationMetrics,
    OperationStageTimings,
    OperationTerminalStatus,
    WorkerHandshake,
    build_worker_handshake,
)
from yakbox.speech.analysis_services import ForcedAligner, SpeechRecognizer
from yakbox.speech.model_registry import ModelRegistry

_FAMILY_ENGINES: Mapping[str, tuple[str, ...]] = {
    "whisper": ("whisper",),
    "parakeet": ("parakeet",),
    "qwen": ("qwen", "qwen-forced"),
}
_PCM_S16_SAMPLE_WIDTH = 2
_NETWORK_FAMILIES = frozenset({socket.AF_INET, socket.AF_INET6})


class EngineFactory(Protocol):
    """Trusted factory supplied by worker code, never by a book manifest."""

    def recognizer(self, engine: str) -> SpeechRecognizer: ...

    def forced_aligner(self, engine: str) -> ForcedAligner: ...


class _ResourceUsage(Protocol):
    ru_maxrss: int


class _DarwinTaskInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("total_user", ctypes.c_uint64),
        ("total_system", ctypes.c_uint64),
        ("threads_user", ctypes.c_uint64),
        ("threads_system", ctypes.c_uint64),
        ("policy", ctypes.c_int32),
        ("faults", ctypes.c_int32),
        ("pageins", ctypes.c_int32),
        ("copy_on_write_faults", ctypes.c_int32),
        ("messages_sent", ctypes.c_int32),
        ("messages_received", ctypes.c_int32),
        ("mach_syscalls", ctypes.c_int32),
        ("unix_syscalls", ctypes.c_int32),
        ("context_switches", ctypes.c_int32),
        ("thread_count", ctypes.c_int32),
        ("running_thread_count", ctypes.c_int32),
        ("priority", ctypes.c_int32),
    ]


@dataclass(slots=True)
class BuiltInEngineFactory:
    """Construct lazy adapters for the worker's immutable family definition."""

    registry: ModelRegistry
    audio_root: Path
    calibration_fingerprint: str
    worker_artifact_digest: str
    lock_digest: str

    def recognizer(self, engine: str) -> SpeechRecognizer:
        execution = execution_identity_from_digests(
            worker_artifact_digest=self.worker_artifact_digest,
            lock_digest=self.lock_digest,
        )
        if engine == "whisper":
            return MlxWhisperRecognizer(
                registry=self.registry,
                audio_root=self.audio_root,
                execution=execution,
                calibration_fingerprint=self.calibration_fingerprint,
            )
        if engine == "parakeet":
            return ParakeetMlxRecognizer(
                registry=self.registry,
                audio_root=self.audio_root,
                execution=execution,
                calibration_fingerprint=self.calibration_fingerprint,
            )
        if engine == "qwen":
            return MlxAudioQwenRecognizer(
                registry=self.registry,
                audio_root=self.audio_root,
                execution=execution,
                calibration_fingerprint=self.calibration_fingerprint,
            )
        raise WorkerProtocolError("Worker recognizer engine is not allowed")

    def forced_aligner(self, engine: str) -> ForcedAligner:
        if engine != "qwen-forced":
            raise WorkerProtocolError("Worker forced-aligner engine is not allowed")
        return MlxAudioQwenForcedAligner(
            registry=self.registry,
            audio_root=self.audio_root,
            execution=execution_identity_from_digests(
                worker_artifact_digest=self.worker_artifact_digest,
                lock_digest=self.lock_digest,
            ),
        )


class AnalysisWorkerApplication:
    """Sequential request executor with independently unloadable model objects."""

    def __init__(
        self,
        *,
        family: str,
        audio_root: Path,
        factory: EngineFactory,
        pid: int | None = None,
    ) -> None:
        if family not in _FAMILY_ENGINES:
            raise ValidationError("Unknown built-in analysis worker family")
        self.family = family
        self.allowed_engines = _FAMILY_ENGINES[family]
        self.audio_root = audio_root.resolve()
        self.factory = factory
        self.pid = pid if pid is not None else os.getpid()
        self.environment_fingerprint = _installed_environment_fingerprint()
        self._engines: dict[str, SpeechRecognizer | ForcedAligner] = {}
        self._active_request_id: str | None = None
        self._completed = 0
        self._failed = 0
        self._model_loads: dict[str, int] = {}
        self._verified_loaded_engines: set[str] = set()

    async def handle(self, request: AnalysisWorkerRequest) -> WorkerSuccess:
        """Execute one already-validated protocol request."""
        if isinstance(
            request, RecognitionManyWorkerRequest | ForcedAlignmentManyWorkerRequest
        ):
            raise WorkerProtocolError("Batch requests require item-framed execution")
        if isinstance(request, StatusWorkerRequest):
            return WorkerSuccess(request.request_id, self.status())
        if isinstance(request, UnloadWorkerRequest):
            self._require_allowed(request.engine)
            _unload_engine(self._engines.pop(request.engine, None))
            self._verified_loaded_engines.discard(request.engine)
            _release_model_memory()
            return WorkerSuccess(request.request_id, None)
        if isinstance(request, ShutdownWorkerRequest):
            for engine in self._engines.values():
                _unload_engine(engine)
            self._engines.clear()
            self._verified_loaded_engines.clear()
            _release_model_memory()
            return WorkerSuccess(request.request_id, None)
        if isinstance(request, CancellationWorkerRequest):
            raise WorkerProtocolError(
                "Sequential worker cancellation is performed by process termination"
            )
        return await self._handle_inference(request)

    async def handle_batch(
        self,
        request: RecognitionManyWorkerRequest | ForcedAlignmentManyWorkerRequest,
    ) -> AsyncIterator[WorkerItemSuccess | WorkerItemFailure | WorkerBatchFinished]:
        """Yield independently committable batch items in stable input order."""
        deadline = time.monotonic_ns() + request.timeout_milliseconds * 1_000_000
        for item in request.items:
            remaining_ms = max(0, (deadline - time.monotonic_ns()) // 1_000_000)
            if remaining_ms <= 0:
                yield _timed_out_item(request, item)
                continue
            single = _single_batch_request(request, item, remaining_ms)
            try:
                response = await self._handle_inference(single)
            except Exception as error:  # noqa: BLE001 - redacted worker boundary
                failure = _failure(request.request_id, error)
                yield WorkerItemFailure(
                    request_id=request.request_id,
                    batch_id=request.batch_id,
                    index=item.index,
                    request_fingerprint=item.request_fingerprint,
                    code=failure.code,
                    detail=failure.detail,
                    retryable=failure.retryable,
                    metrics=failure.metrics,
                    terminal_status=_terminal_status(failure.code),
                )
            else:
                result = response.result
                if not isinstance(result, RecognitionResult | ForcedAlignmentResult):
                    raise WorkerProtocolError(
                        "Batch inference returned invalid evidence"
                    )
                yield WorkerItemSuccess(
                    request_id=request.request_id,
                    batch_id=request.batch_id,
                    index=item.index,
                    request_fingerprint=item.request_fingerprint,
                    result=result,
                    metrics=response.metrics,
                )
        yield WorkerBatchFinished(
            request.request_id, request.batch_id, len(request.items)
        )

    async def _handle_inference(
        self,
        request: RecognitionWorkerRequest | ForcedAlignmentWorkerRequest,
    ) -> WorkerSuccess:
        self._require_allowed(request.engine)
        self._active_request_id = request.request_id
        started_ns = time.monotonic_ns()
        loaded_before = request.engine in self._verified_loaded_engines
        requested_frames = 0
        try:
            path, span = _verified_audio(
                self.audio_root,
                request.relative_audio_path,
                request.audio_digest,
                request.span,
            )
            requested_frames = span.end_frame - span.start_frame
            prepared_ns = time.monotonic_ns() - started_ns
            inference_started_ns = time.monotonic_ns()
            async with asyncio.timeout(request.timeout_milliseconds / 1_000):
                result = await self._run_inference(request, path, span)
            inference_ns = time.monotonic_ns() - inference_started_ns
        except Exception:
            self._failed += 1
            raise
        else:
            if request.engine not in self._verified_loaded_engines:
                self._model_loads[request.engine] = (
                    self._model_loads.get(request.engine, 0) + 1
                )
                self._verified_loaded_engines.add(request.engine)
            self._completed += 1
            metal_active, metal_peak = _metal_memory_bytes()
            return WorkerSuccess(
                request.request_id,
                result,
                OperationMetrics(
                    requested_audio_frames=requested_frames,
                    analyzed_audio_frames=requested_frames,
                    wall_time_ns=time.monotonic_ns() - started_ns,
                    peak_resident_memory_bytes=_peak_resident_memory_bytes(),
                    peak_accelerator_memory_bytes=metal_peak or metal_active,
                    cache_hit=False,
                    model_loads=0 if loaded_before else 1,
                    cache_miss_reason="worker_inference_required",
                    stage_timings=OperationStageTimings(
                        preparing_audio_ns=prepared_ns,
                        inference_ns=(
                            0
                            if isinstance(request, ForcedAlignmentWorkerRequest)
                            else inference_ns
                        ),
                        forced_alignment_ns=(
                            inference_ns
                            if isinstance(request, ForcedAlignmentWorkerRequest)
                            else 0
                        ),
                    ),
                ),
            )
        finally:
            self._active_request_id = None

    async def _run_inference(
        self,
        request: RecognitionWorkerRequest | ForcedAlignmentWorkerRequest,
        path: Path,
        span: AudioSpan,
    ) -> RecognitionResult | ForcedAlignmentResult:
        if isinstance(request, RecognitionWorkerRequest):
            recognizer = self._recognizer(request.engine)
            _require_engine_fingerprint(
                request.engine_fingerprint, recognizer.fingerprint
            )
            return await recognizer.recognize(
                path,
                language=request.language,
                span=span,
            )
        aligner = self._forced_aligner(request.engine)
        _require_engine_fingerprint(request.engine_fingerprint, aligner.fingerprint)
        return await aligner.force_align(
            path,
            request.expected_text,
            language=request.language,
            purpose=request.purpose,
            verified_span=request.verified_span,
            span=span,
        )

    def status(self) -> WorkerStatus:
        metal_active, metal_peak = _metal_memory_bytes()
        return WorkerStatus(
            pid=self.pid,
            family=self.family,
            python_version=platform.python_version(),
            environment_fingerprint=self.environment_fingerprint,
            ready=True,
            loaded_engines=tuple(sorted(self._engines)),
            active_request_id=self._active_request_id,
            completed_requests=self._completed,
            failed_requests=self._failed,
            model_loads=tuple(
                WorkerModelLoad(engine, count)
                for engine, count in sorted(self._model_loads.items())
            ),
            peak_resident_memory_bytes=_peak_resident_memory_bytes(),
            metal_active_memory_bytes=metal_active,
            metal_peak_memory_bytes=metal_peak,
            resident_memory_bytes=_resident_memory_bytes(),
        )

    def _require_allowed(self, engine: str) -> None:
        if engine not in self.allowed_engines:
            raise WorkerProtocolError(
                f"Engine {engine!r} is not allowed in the {self.family!r} worker"
            )

    def _recognizer(self, engine: str) -> SpeechRecognizer:
        value = self._engines.get(engine)
        if value is None:
            value = self.factory.recognizer(engine)
            self._engines[engine] = value
        if not isinstance(value, SpeechRecognizer):
            raise WorkerProtocolError("Worker engine is not a recognizer")
        return value

    def _forced_aligner(self, engine: str) -> ForcedAligner:
        value = self._engines.get(engine)
        if value is None:
            value = self.factory.forced_aligner(engine)
            self._engines[engine] = value
        if not isinstance(value, ForcedAligner):
            raise WorkerProtocolError("Worker engine is not a forced aligner")
        return value


def _single_batch_request(
    request: RecognitionManyWorkerRequest | ForcedAlignmentManyWorkerRequest,
    item: RecognitionBatchItem | ForcedAlignmentBatchItem,
    timeout_milliseconds: int,
) -> RecognitionWorkerRequest | ForcedAlignmentWorkerRequest:
    if isinstance(request, RecognitionManyWorkerRequest) and isinstance(
        item, RecognitionBatchItem
    ):
        return RecognitionWorkerRequest(
            request_id=request.request_id,
            engine=request.engine,
            engine_fingerprint=request.engine_fingerprint,
            relative_audio_path=item.relative_audio_path,
            audio_digest=item.audio_digest,
            language=item.language,
            span=item.span,
            timeout_milliseconds=timeout_milliseconds,
        )
    if isinstance(request, ForcedAlignmentManyWorkerRequest) and isinstance(
        item, ForcedAlignmentBatchItem
    ):
        return ForcedAlignmentWorkerRequest(
            request_id=request.request_id,
            engine=request.engine,
            engine_fingerprint=request.engine_fingerprint,
            relative_audio_path=item.relative_audio_path,
            audio_digest=item.audio_digest,
            expected_text=item.expected_text,
            language=item.language,
            purpose=item.purpose,
            span=item.span,
            verified_span=item.verified_span,
            timeout_milliseconds=timeout_milliseconds,
        )
    raise WorkerProtocolError("Analysis batch item does not match its operation")


def _timed_out_item(
    request: RecognitionManyWorkerRequest | ForcedAlignmentManyWorkerRequest,
    item: RecognitionBatchItem | ForcedAlignmentBatchItem,
) -> WorkerItemFailure:
    return WorkerItemFailure(
        request_id=request.request_id,
        batch_id=request.batch_id,
        index=item.index,
        request_fingerprint=item.request_fingerprint,
        code=WorkerErrorCode.TIMEOUT,
        detail=_safe_error_detail(WorkerErrorCode.TIMEOUT),
        retryable=True,
        metrics=OperationMetrics(
            requested_audio_frames=item.span.end_frame - item.span.start_frame,
            analyzed_audio_frames=0,
            wall_time_ns=0,
            peak_resident_memory_bytes=0,
            peak_accelerator_memory_bytes=None,
            cache_hit=False,
            model_loads=0,
        ),
        terminal_status=OperationTerminalStatus.TIMED_OUT,
    )


def _terminal_status(code: WorkerErrorCode) -> OperationTerminalStatus:
    if code is WorkerErrorCode.CANCELLED:
        return OperationTerminalStatus.CANCELLED
    if code is WorkerErrorCode.TIMEOUT:
        return OperationTerminalStatus.TIMED_OUT
    if code is WorkerErrorCode.WORKER_TERMINATED:
        return OperationTerminalStatus.WORKER_CRASHED
    return OperationTerminalStatus.INVALID_RESULT


def _installed_environment_fingerprint() -> str:
    packages = tuple(
        sorted(
            (
                name.casefold().replace("_", "-"),
                distribution.version,
            )
            for distribution in distributions()
            if (name := distribution.metadata.get("Name"))
        )
    )
    return semantic_fingerprint(
        "analysis-worker-installed-environment-v1",
        {
            "python_version": platform.python_version(),
            "packages": packages,
        },
    )


def _verified_audio(
    audio_root: Path,
    relative_path: str,
    expected_digest: str,
    requested_span: AudioSpan | None,
) -> tuple[Path, AudioSpan]:
    candidate = audio_root / relative_path
    if candidate.is_symlink():
        raise ValidationError("Analysis worker rejects symlink audio inputs")
    path = safe_child(audio_root, candidate)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValidationError("Analysis worker audio is not a regular file")
    digest = sha256_file(path)
    if digest != expected_digest:
        raise ModelIntegrityError("Analysis worker audio digest does not match")
    try:
        with wave.open(str(path), "rb") as reader:
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != _PCM_S16_SAMPLE_WIDTH
            ):
                raise ValidationError("Analysis worker requires mono PCM16 WAV")
            span = AudioSpan(digest, 0, reader.getnframes(), reader.getframerate())
    except (OSError, EOFError, wave.Error) as error:
        raise ValidationError("Analysis worker audio is not a valid WAV") from error
    if requested_span is not None and requested_span != span:
        raise ValidationError("Analysis worker span does not match the WAV")
    return path, span


def _release_model_memory() -> None:
    gc.collect()
    module = sys.modules.get("mlx.core")
    clear_cache = getattr(module, "clear_cache", None)
    if callable(clear_cache):
        cast(Callable[[], object], clear_cache)()


def _unload_engine(engine: SpeechRecognizer | ForcedAligner | None) -> None:
    if engine is None:
        return
    unload = getattr(engine, "unload", None)
    if callable(unload):
        cast(Callable[[], object], unload)()


def _peak_resident_memory_bytes() -> int:
    """Return the process high-water RSS without adding a runtime dependency."""
    try:
        resource_module = __import__("resource")
        getrusage = cast(Callable[[int], object], resource_module.getrusage)
        usage = cast(
            _ResourceUsage,
            getrusage(resource_module.RUSAGE_SELF),
        )
        high_water = int(usage.ru_maxrss)
    except AttributeError, ImportError, OSError, TypeError, ValueError:
        return 0
    return high_water if sys.platform == "darwin" else high_water * 1024


def _resident_memory_bytes() -> int:
    """Measure current RSS so advisory unload can be verified on macOS."""
    if sys.platform != "darwin":
        return _peak_resident_memory_bytes()
    try:
        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        proc_pidinfo = library.proc_pidinfo
        proc_pidinfo.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        )
        proc_pidinfo.restype = ctypes.c_int
        info = _DarwinTaskInfo()
        size = ctypes.sizeof(info)
        written = proc_pidinfo(os.getpid(), 4, 0, ctypes.byref(info), size)
        if written == size:
            return int(info.resident_size)
    except AttributeError, OSError, TypeError, ValueError:
        pass
    return _peak_resident_memory_bytes()


def _metal_memory_bytes() -> tuple[int | None, int | None]:
    """Read MLX allocator counters only after an adapter imported MLX."""
    module = sys.modules.get("mlx.core")
    if module is None:
        return None, None
    active = getattr(module, "get_active_memory", None)
    peak = getattr(module, "get_peak_memory", None)
    if not callable(active) or not callable(peak):
        return None, None
    try:
        return (
            cast(Callable[[], int], active)(),
            cast(Callable[[], int], peak)(),
        )
    except RuntimeError, TypeError, ValueError:
        return None, None


def _require_engine_fingerprint(requested: str, actual: str) -> None:
    if requested != actual:
        raise WorkerProtocolError(
            "Analysis worker engine differs from its built-in adapter fingerprint"
        )


def _failure(request_id: str, error: Exception) -> WorkerFailure:
    if isinstance(error, WorkerProtocolError | ValidationError):
        code = WorkerErrorCode.INVALID_REQUEST
    elif isinstance(error, BackendUnavailableError):
        code = WorkerErrorCode.MODEL_UNAVAILABLE
    elif isinstance(error, ModelIntegrityError):
        code = WorkerErrorCode.MODEL_INVALID
    elif isinstance(error, TimeoutError):
        code = WorkerErrorCode.TIMEOUT
    else:
        code = WorkerErrorCode.INFERENCE_FAILED
    return WorkerFailure(
        request_id=request_id,
        code=code,
        detail=_safe_error_detail(code),
        retryable=code
        in {
            WorkerErrorCode.TIMEOUT,
            WorkerErrorCode.INFERENCE_FAILED,
        },
    )


def _safe_error_detail(code: WorkerErrorCode) -> str:
    return {
        WorkerErrorCode.INVALID_REQUEST: "The worker rejected the request contract",
        WorkerErrorCode.MODEL_UNAVAILABLE: "The required model runtime is unavailable",
        WorkerErrorCode.MODEL_INVALID: "The required local model failed verification",
        WorkerErrorCode.TIMEOUT: "Model inference exceeded its timeout",
        WorkerErrorCode.WORKER_TERMINATED: (
            "The worker terminated before returning a result"
        ),
        WorkerErrorCode.INFERENCE_FAILED: "Model inference failed",
        WorkerErrorCode.CANCELLED: "Model inference was cancelled",
        WorkerErrorCode.INTERNAL_ERROR: "The worker encountered an internal error",
    }[code]


@dataclass(frozen=True, slots=True)
class _ActiveOperation:
    request_id: str
    task: asyncio.Task[None]


async def _serve(
    application: AnalysisWorkerApplication,
    handshake: WorkerHandshake,
    reader: Callable[[int], bytes],
    writer: Callable[[bytes], object],
) -> int:
    writer(encode_worker_handshake(handshake) + b"\n")
    active: _ActiveOperation | None = None
    while True:
        active = await _reap_active(active)
        frame = await asyncio.to_thread(reader, MAXIMUM_WORKER_FRAME_BYTES + 2)
        if not frame:
            await _finish_active(active)
            return 0
        active = await _reap_active(active)
        active, shutdown = await _dispatch_frame(
            application,
            frame,
            active,
            writer,
        )
        if shutdown:
            return 0


async def _reap_active(
    active: _ActiveOperation | None,
) -> _ActiveOperation | None:
    if active is None or not active.task.done():
        return active
    await active.task
    return None


async def _finish_active(active: _ActiveOperation | None) -> None:
    if active is not None:
        await active.task


async def _dispatch_frame(
    application: AnalysisWorkerApplication,
    frame: bytes,
    active: _ActiveOperation | None,
    writer: Callable[[bytes], object],
) -> tuple[_ActiveOperation | None, bool]:
    if len(frame) > MAXIMUM_WORKER_FRAME_BYTES + 1 or not frame.endswith(b"\n"):
        _write_response(
            writer,
            WorkerFailure(
                "invalid-frame",
                WorkerErrorCode.INVALID_REQUEST,
                "Analysis worker frame is too large",
                False,
            ),
        )
        return active, False
    try:
        request = parse_worker_request(frame[:-1])
    except Exception as error:  # noqa: BLE001 - redact protocol boundary errors
        _write_response(writer, _failure("invalid-frame", error))
        return active, False
    try:
        return await _dispatch_request(application, request, active, writer)
    except Exception as error:  # noqa: BLE001 - redact protocol boundary errors
        _write_response(writer, _failure(request.request_id, error))
        return active, False


async def _dispatch_request(
    application: AnalysisWorkerApplication,
    request: AnalysisWorkerRequest,
    active: _ActiveOperation | None,
    writer: Callable[[bytes], object],
) -> tuple[_ActiveOperation | None, bool]:
    if _is_inference_request(request):
        if active is not None:
            _write_response(writer, _busy_response(request.request_id))
            return active, False
        task = asyncio.create_task(
            _execute_active_request(application, request, writer)
        )
        return _ActiveOperation(request.request_id, task), False
    response = await _control_response(application, request, active)
    _write_response(writer, response)
    shutdown = isinstance(request, ShutdownWorkerRequest) and isinstance(
        response, WorkerSuccess
    )
    return active, shutdown


async def _control_response(
    application: AnalysisWorkerApplication,
    request: AnalysisWorkerRequest,
    active: _ActiveOperation | None,
) -> WorkerSuccess | WorkerFailure:
    if isinstance(request, CancellationWorkerRequest):
        return _cancellation_response(
            request,
            active.request_id if active is not None else None,
        )
    if active is None or isinstance(request, StatusWorkerRequest):
        return await application.handle(request)
    return _busy_response(request.request_id)


def _write_response(
    writer: Callable[[bytes], object],
    response: AnalysisWorkerResponse,
) -> None:
    writer(encode_worker_response(response) + b"\n")


def _is_inference_request(request: AnalysisWorkerRequest) -> bool:
    return isinstance(
        request,
        RecognitionWorkerRequest
        | ForcedAlignmentWorkerRequest
        | RecognitionManyWorkerRequest
        | ForcedAlignmentManyWorkerRequest,
    )


async def _execute_active_request(
    application: AnalysisWorkerApplication,
    request: AnalysisWorkerRequest,
    writer: Callable[[bytes], object],
) -> None:
    try:
        if isinstance(
            request, RecognitionManyWorkerRequest | ForcedAlignmentManyWorkerRequest
        ):
            async for item_response in application.handle_batch(request):
                writer(encode_worker_response(item_response) + b"\n")
            return
        response = await application.handle(request)
    except Exception as error:  # noqa: BLE001 - redact protocol boundary errors
        response = _failure(request.request_id, error)
    writer(encode_worker_response(response) + b"\n")


def _cancellation_response(
    request: CancellationWorkerRequest,
    active_request_id: str | None,
) -> WorkerSuccess | WorkerFailure:
    if active_request_id is None or request.target_request_id != active_request_id:
        return WorkerFailure(
            request.request_id,
            WorkerErrorCode.INVALID_REQUEST,
            "Analysis worker cancellation target is not active",
            False,
        )
    return WorkerSuccess(request.request_id, None)


def _busy_response(request_id: str) -> WorkerFailure:
    return WorkerFailure(
        request_id,
        WorkerErrorCode.INVALID_REQUEST,
        "Analysis worker is already running an inference operation",
        False,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--family", choices=tuple(_FAMILY_ENGINES), required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--calibration-fingerprint", required=True)
    parser.add_argument("--worker-artifact-digest", required=True)
    parser.add_argument("--lock-digest", required=True)
    parser.add_argument("--definition-fingerprint", required=True)
    return parser


def _install_offline_audit_guard() -> None:
    """Deny internet socket creation inside the inference worker process."""

    def deny_network(event: str, arguments: tuple[object, ...]) -> None:
        if event == "socket.__new__" and len(arguments) > 1:
            family = arguments[1]
            if family in _NETWORK_FAMILIES:
                raise PermissionError("Analysis workers cannot create network sockets")
        if event in {
            "socket.getaddrinfo",
            "socket.gethostbyaddr",
            "socket.gethostbyname",
        }:
            raise PermissionError("Analysis workers cannot resolve network addresses")

    sys.addaudithook(deny_network)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run a bounded newline-framed worker over isolated standard streams."""
    options = _argument_parser().parse_args(arguments)
    _install_offline_audit_guard()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    with Path(os.devnull).open("wb") as null_output:
        os.dup2(null_output.fileno(), sys.stdout.fileno())
    factory = BuiltInEngineFactory(
        ModelRegistry(options.model_root),
        options.audio_root,
        options.calibration_fingerprint,
        options.worker_artifact_digest,
        options.lock_digest,
    )
    application = AnalysisWorkerApplication(
        family=options.family,
        audio_root=options.audio_root,
        factory=factory,
    )
    handshake = build_worker_handshake(
        family=options.family,
        engines=_FAMILY_ENGINES[options.family],
        worker_artifact_fingerprint=options.worker_artifact_digest,
        environment_lock_fingerprint=options.lock_digest,
        adapter_fingerprint=options.definition_fingerprint,
    )
    return asyncio.run(
        _serve(application, handshake, sys.stdin.buffer.readline, protocol.write)
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AnalysisWorkerApplication",
    "BuiltInEngineFactory",
    "EngineFactory",
    "main",
]
