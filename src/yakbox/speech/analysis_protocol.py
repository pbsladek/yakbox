"""Versioned, bounded protocol shared by isolated analysis workers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import cast

from yakbox.errors import WorkerProtocolError
from yakbox.speech.analysis_models import (
    AlignmentPurpose,
    AudioSpan,
    ForcedAlignmentResult,
    RecognitionResult,
    VerifiedTextSpan,
)
from yakbox.speech.analysis_scheduler import (
    OperationMetrics,
    OperationStageTimings,
    OperationTerminalStatus,
    WorkerHandshake,
)
from yakbox.speech.analysis_serialization import (
    forced_alignment_from_report,
    forced_alignment_report,
    recognition_from_report,
    recognition_report,
)

ANALYSIS_WORKER_PROTOCOL_VERSION = 2
MAXIMUM_WORKER_FRAME_BYTES = 1024 * 1024
MAXIMUM_WORKER_ERROR_BYTES = 512
MAXIMUM_BATCH_ITEMS = 256
MAXIMUM_TIMEOUT_MILLISECONDS = 3_600_000
_REQUEST_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class WorkerOperation(StrEnum):
    RECOGNIZE = "recognize"
    RECOGNIZE_MANY = "recognize_many"
    FORCE_ALIGN = "force_align"
    FORCE_ALIGN_MANY = "force_align_many"
    CANCEL = "cancel"
    STATUS = "status"
    UNLOAD = "unload"
    SHUTDOWN = "shutdown"


class WorkerErrorCode(StrEnum):
    CANCELLED = "cancelled"
    INVALID_REQUEST = "invalid_request"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_INVALID = "model_invalid"
    TIMEOUT = "timeout"
    WORKER_TERMINATED = "worker_terminated"
    INFERENCE_FAILED = "inference_failed"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class RecognitionWorkerRequest:
    """Independent recognition request with no expected-text field."""

    request_id: str
    engine: str
    engine_fingerprint: str
    relative_audio_path: str
    audio_digest: str
    language: str
    span: AudioSpan
    timeout_milliseconds: int = 300_000


@dataclass(frozen=True, slots=True)
class ForcedAlignmentWorkerRequest:
    """Timing request whose expected text has separate, explicit authority."""

    request_id: str
    engine: str
    engine_fingerprint: str
    relative_audio_path: str
    audio_digest: str
    expected_text: str
    language: str
    purpose: AlignmentPurpose
    span: AudioSpan
    verified_span: VerifiedTextSpan | None = None
    timeout_milliseconds: int = 300_000


@dataclass(frozen=True, slots=True)
class RecognitionBatchItem:
    index: int
    request_fingerprint: str
    relative_audio_path: str
    audio_digest: str
    language: str
    span: AudioSpan


@dataclass(frozen=True, slots=True)
class ForcedAlignmentBatchItem:
    index: int
    request_fingerprint: str
    relative_audio_path: str
    audio_digest: str
    expected_text: str
    language: str
    purpose: AlignmentPurpose
    verified_span: VerifiedTextSpan | None
    span: AudioSpan


@dataclass(frozen=True, slots=True)
class RecognitionManyWorkerRequest:
    request_id: str
    batch_id: str
    engine: str
    engine_fingerprint: str
    timeout_milliseconds: int
    items: tuple[RecognitionBatchItem, ...]


@dataclass(frozen=True, slots=True)
class ForcedAlignmentManyWorkerRequest:
    request_id: str
    batch_id: str
    engine: str
    engine_fingerprint: str
    timeout_milliseconds: int
    items: tuple[ForcedAlignmentBatchItem, ...]


@dataclass(frozen=True, slots=True)
class CancellationWorkerRequest:
    request_id: str
    target_request_id: str


@dataclass(frozen=True, slots=True)
class StatusWorkerRequest:
    request_id: str


@dataclass(frozen=True, slots=True)
class UnloadWorkerRequest:
    request_id: str
    engine: str


@dataclass(frozen=True, slots=True)
class ShutdownWorkerRequest:
    request_id: str


type AnalysisWorkerRequest = (
    RecognitionWorkerRequest
    | RecognitionManyWorkerRequest
    | ForcedAlignmentWorkerRequest
    | ForcedAlignmentManyWorkerRequest
    | CancellationWorkerRequest
    | StatusWorkerRequest
    | UnloadWorkerRequest
    | ShutdownWorkerRequest
)


def _empty_operation_metrics() -> OperationMetrics:
    return OperationMetrics(0, 0, 0, 0, None, False, 0)


@dataclass(frozen=True, slots=True)
class WorkerModelLoad:
    """Count of model constructions in one worker process."""

    engine: str
    count: int

    def __post_init__(self) -> None:
        if not self.engine or self.count < 0:
            raise WorkerProtocolError("Analysis worker model-load count is invalid")


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    """Bounded operational status without local paths or text payloads."""

    pid: int
    family: str
    python_version: str
    environment_fingerprint: str
    ready: bool
    loaded_engines: tuple[str, ...]
    active_request_id: str | None
    completed_requests: int
    failed_requests: int
    model_loads: tuple[WorkerModelLoad, ...]
    peak_resident_memory_bytes: int
    metal_active_memory_bytes: int | None
    metal_peak_memory_bytes: int | None
    resident_memory_bytes: int = 0

    def __post_init__(self) -> None:
        if self.pid <= 0:
            raise WorkerProtocolError("Analysis worker pid is invalid")
        if not self.family or not self.python_version:
            raise WorkerProtocolError("Analysis worker environment identity is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", self.environment_fingerprint):
            raise WorkerProtocolError(
                "Analysis worker environment fingerprint is invalid"
            )
        if self.completed_requests < 0 or self.failed_requests < 0:
            raise WorkerProtocolError("Analysis worker request count is invalid")
        if self.peak_resident_memory_bytes < 0:
            raise WorkerProtocolError("Analysis worker memory sample is invalid")
        if self.resident_memory_bytes < 0:
            raise WorkerProtocolError("Analysis worker resident memory is invalid")
        if (
            self.metal_active_memory_bytes is not None
            and self.metal_active_memory_bytes < 0
        ):
            raise WorkerProtocolError("Analysis worker Metal sample is invalid")
        if (
            self.metal_peak_memory_bytes is not None
            and self.metal_peak_memory_bytes < 0
        ):
            raise WorkerProtocolError("Analysis worker Metal sample is invalid")
        engines = tuple(load.engine for load in self.model_loads)
        if engines != tuple(sorted(set(engines))):
            raise WorkerProtocolError("Analysis worker model loads are not stable")


@dataclass(frozen=True, slots=True)
class WorkerSuccess:
    """Typed successful worker response."""

    request_id: str
    result: RecognitionResult | ForcedAlignmentResult | WorkerStatus | None
    metrics: OperationMetrics = field(default_factory=_empty_operation_metrics)


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    """Redacted worker error that never echoes text or local paths."""

    request_id: str
    code: WorkerErrorCode
    detail: str
    retryable: bool
    metrics: OperationMetrics = field(default_factory=_empty_operation_metrics)

    def __post_init__(self) -> None:
        if len(self.detail.encode("utf-8")) > MAXIMUM_WORKER_ERROR_BYTES:
            raise WorkerProtocolError("Worker error detail exceeds 512 bytes")
        if "\n" in self.detail or "\r" in self.detail:
            raise WorkerProtocolError("Worker error detail must be one line")


@dataclass(frozen=True, slots=True)
class WorkerItemSuccess:
    """One independently committable item completion from a batch."""

    request_id: str
    batch_id: str
    index: int
    request_fingerprint: str
    result: RecognitionResult | ForcedAlignmentResult
    metrics: OperationMetrics
    terminal_status: OperationTerminalStatus = OperationTerminalStatus.COMPLETED


@dataclass(frozen=True, slots=True)
class WorkerItemFailure:
    """One typed terminal item failure that does not abort other items."""

    request_id: str
    batch_id: str
    index: int
    request_fingerprint: str
    code: WorkerErrorCode
    detail: str
    retryable: bool
    metrics: OperationMetrics
    terminal_status: OperationTerminalStatus

    def __post_init__(self) -> None:
        if len(self.detail.encode("utf-8")) > MAXIMUM_WORKER_ERROR_BYTES:
            raise WorkerProtocolError("Worker item error detail exceeds 512 bytes")
        if "\n" in self.detail or "\r" in self.detail:
            raise WorkerProtocolError("Worker item error detail must be one line")


@dataclass(frozen=True, slots=True)
class WorkerBatchFinished:
    """Terminal delimiter proving a batch produced all preceding item frames."""

    request_id: str
    batch_id: str
    item_count: int
    terminal_status: OperationTerminalStatus = OperationTerminalStatus.COMPLETED
    metrics: OperationMetrics = field(default_factory=_empty_operation_metrics)

    def __post_init__(self) -> None:
        if self.terminal_status is not OperationTerminalStatus.COMPLETED:
            raise WorkerProtocolError(
                "Analysis batch delimiter must have completed terminal status"
            )


type AnalysisWorkerResponse = (
    WorkerSuccess
    | WorkerFailure
    | WorkerItemSuccess
    | WorkerItemFailure
    | WorkerBatchFinished
)


def parse_worker_request(payload: bytes) -> AnalysisWorkerRequest:
    """Parse one bounded frame and reject unknown or misplaced fields."""
    if not payload or len(payload) > MAXIMUM_WORKER_FRAME_BYTES:
        raise WorkerProtocolError("Analysis worker frame size is invalid")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerProtocolError(
            "Analysis worker request is not valid JSON"
        ) from error
    if not isinstance(raw, Mapping):
        raise WorkerProtocolError("Analysis worker request must be an object")
    if _integer(raw, "protocol_version") != ANALYSIS_WORKER_PROTOCOL_VERSION:
        raise WorkerProtocolError("Analysis worker protocol version is unsupported")
    try:
        operation = WorkerOperation(_string(raw, "operation"))
    except ValueError as error:
        raise WorkerProtocolError("Analysis worker operation is unsupported") from error
    request_id = _request_identifier(raw, "request_id")
    return _parse_operation(raw, operation, request_id)


def encode_worker_handshake(handshake: WorkerHandshake) -> bytes:
    """Serialize the child's one mandatory pre-request identity frame."""
    value = {
        "event": "handshake",
        "protocol_version": handshake.protocol_version,
        "worker_artifact_fingerprint": handshake.worker_artifact_fingerprint,
        "environment_lock_fingerprint": handshake.environment_lock_fingerprint,
        "adapter_fingerprint": handshake.adapter_fingerprint,
        "python_fingerprint": handshake.python_fingerprint,
        "execution_fingerprint": handshake.execution_fingerprint,
        "capability_fingerprint": handshake.capability_fingerprint,
    }
    return _encode_frame(value, label="handshake")


def parse_worker_handshake(payload: bytes) -> WorkerHandshake:
    """Parse and validate the child's one non-negotiable identity frame."""
    raw = _parse_json_frame(payload, label="handshake")
    _require_keys(
        raw,
        {
            "event",
            "protocol_version",
            "worker_artifact_fingerprint",
            "environment_lock_fingerprint",
            "adapter_fingerprint",
            "python_fingerprint",
            "execution_fingerprint",
            "capability_fingerprint",
        },
    )
    if _string(raw, "event") != "handshake":
        raise WorkerProtocolError("Analysis worker did not send a handshake")
    return WorkerHandshake(
        protocol_version=_integer(raw, "protocol_version"),
        worker_artifact_fingerprint=_sha256(raw, "worker_artifact_fingerprint"),
        environment_lock_fingerprint=_sha256(raw, "environment_lock_fingerprint"),
        adapter_fingerprint=_sha256(raw, "adapter_fingerprint"),
        python_fingerprint=_sha256(raw, "python_fingerprint"),
        execution_fingerprint=_sha256(raw, "execution_fingerprint"),
        capability_fingerprint=_sha256(raw, "capability_fingerprint"),
    )


def _parse_operation(
    raw: Mapping[str, object],
    operation: WorkerOperation,
    request_id: str,
) -> AnalysisWorkerRequest:
    if operation in {
        WorkerOperation.RECOGNIZE,
        WorkerOperation.RECOGNIZE_MANY,
        WorkerOperation.FORCE_ALIGN,
        WorkerOperation.FORCE_ALIGN_MANY,
    }:
        return _parse_inference_operation(raw, operation, request_id)
    return _parse_lifecycle_operation(raw, operation, request_id)


def _parse_inference_operation(
    raw: Mapping[str, object],
    operation: WorkerOperation,
    request_id: str,
) -> AnalysisWorkerRequest:
    if operation is WorkerOperation.RECOGNIZE:
        return _recognition_request(raw, request_id)
    if operation is WorkerOperation.RECOGNIZE_MANY:
        return _recognition_many_request(raw, request_id)
    if operation is WorkerOperation.FORCE_ALIGN:
        return _forced_alignment_request(raw, request_id)
    if operation is WorkerOperation.FORCE_ALIGN_MANY:
        return _forced_alignment_many_request(raw, request_id)
    raise AssertionError("Inference operation dispatch is incomplete")


def _parse_lifecycle_operation(
    raw: Mapping[str, object],
    operation: WorkerOperation,
    request_id: str,
) -> AnalysisWorkerRequest:
    if operation is WorkerOperation.CANCEL:
        _require_keys(
            raw,
            {
                "protocol_version",
                "operation",
                "request_id",
                "target_request_id",
            },
        )
        return CancellationWorkerRequest(
            request_id,
            _request_identifier(raw, "target_request_id"),
        )
    if operation is WorkerOperation.STATUS:
        _require_keys(raw, {"protocol_version", "operation", "request_id"})
        return StatusWorkerRequest(request_id)
    if operation is WorkerOperation.UNLOAD:
        _require_keys(raw, {"protocol_version", "operation", "request_id", "engine"})
        return UnloadWorkerRequest(request_id, _string(raw, "engine"))
    _require_keys(raw, {"protocol_version", "operation", "request_id"})
    return ShutdownWorkerRequest(request_id)


def encode_worker_request(request: AnalysisWorkerRequest) -> bytes:
    """Serialize one typed request as a canonical bounded JSON frame."""
    value: dict[str, object] = {
        "protocol_version": ANALYSIS_WORKER_PROTOCOL_VERSION,
        "request_id": request.request_id,
    }
    if isinstance(request, RecognitionWorkerRequest):
        value.update(
            operation=WorkerOperation.RECOGNIZE.value,
            engine=request.engine,
            engine_fingerprint=request.engine_fingerprint,
            relative_audio_path=request.relative_audio_path,
            audio_digest=request.audio_digest,
            language=request.language,
            span=_span_to_dict(request.span),
            timeout_milliseconds=request.timeout_milliseconds,
        )
    elif isinstance(request, RecognitionManyWorkerRequest):
        value.update(
            operation=WorkerOperation.RECOGNIZE_MANY.value,
            batch_id=request.batch_id,
            engine=request.engine,
            engine_fingerprint=request.engine_fingerprint,
            timeout_milliseconds=request.timeout_milliseconds,
            items=[_recognition_batch_item_to_dict(item) for item in request.items],
        )
    elif isinstance(request, ForcedAlignmentWorkerRequest):
        value.update(
            operation=WorkerOperation.FORCE_ALIGN.value,
            engine=request.engine,
            engine_fingerprint=request.engine_fingerprint,
            relative_audio_path=request.relative_audio_path,
            audio_digest=request.audio_digest,
            expected_text=request.expected_text,
            language=request.language,
            purpose=request.purpose.value,
            verified_span=_verified_span_to_dict(request.verified_span),
            span=_span_to_dict(request.span),
            timeout_milliseconds=request.timeout_milliseconds,
        )
    elif isinstance(request, ForcedAlignmentManyWorkerRequest):
        value.update(
            operation=WorkerOperation.FORCE_ALIGN_MANY.value,
            batch_id=request.batch_id,
            engine=request.engine,
            engine_fingerprint=request.engine_fingerprint,
            timeout_milliseconds=request.timeout_milliseconds,
            items=[
                _forced_alignment_batch_item_to_dict(item) for item in request.items
            ],
        )
    elif isinstance(request, CancellationWorkerRequest):
        value.update(
            operation=WorkerOperation.CANCEL.value,
            target_request_id=request.target_request_id,
        )
    elif isinstance(request, StatusWorkerRequest):
        value["operation"] = WorkerOperation.STATUS.value
    elif isinstance(request, UnloadWorkerRequest):
        value.update(operation=WorkerOperation.UNLOAD.value, engine=request.engine)
    else:
        value["operation"] = WorkerOperation.SHUTDOWN.value
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    if len(encoded) > MAXIMUM_WORKER_FRAME_BYTES:
        raise WorkerProtocolError("Encoded analysis worker frame is too large")
    return encoded


def encode_worker_response(response: AnalysisWorkerResponse) -> bytes:
    """Serialize one bounded, typed response without transcript plaintext."""
    value: dict[str, object] = {
        "protocol_version": ANALYSIS_WORKER_PROTOCOL_VERSION,
        "request_id": response.request_id,
    }
    if isinstance(response, WorkerBatchFinished):
        value.update(
            event="batch_finished",
            batch_id=response.batch_id,
            item_count=response.item_count,
            terminal_status=response.terminal_status.value,
            metrics=asdict(response.metrics),
        )
    elif isinstance(response, WorkerItemFailure):
        value.update(
            event="item_error",
            batch_id=response.batch_id,
            index=response.index,
            request_fingerprint=response.request_fingerprint,
            terminal_status=response.terminal_status.value,
            code=response.code.value,
            detail=response.detail,
            retryable=response.retryable,
            metrics=asdict(response.metrics),
        )
    elif isinstance(response, WorkerItemSuccess):
        value.update(
            event="item_result",
            batch_id=response.batch_id,
            index=response.index,
            request_fingerprint=response.request_fingerprint,
            terminal_status=response.terminal_status.value,
            metrics=asdict(response.metrics),
        )
        value.update(_result_fields(response.result))
    elif isinstance(response, WorkerFailure):
        value.update(
            event="error",
            code=response.code.value,
            detail=response.detail,
            retryable=response.retryable,
            terminal_status=_terminal_status_for_error(response.code).value,
            metrics=asdict(response.metrics),
        )
    else:
        value.update(
            event="result",
            terminal_status=OperationTerminalStatus.COMPLETED.value,
            metrics=asdict(response.metrics),
        )
        value.update(_result_fields(response.result))
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    if len(encoded) > MAXIMUM_WORKER_FRAME_BYTES:
        raise WorkerProtocolError("Encoded analysis worker response is too large")
    return encoded


def parse_worker_response(payload: bytes) -> AnalysisWorkerResponse:
    """Parse one worker response and reconstruct typed, validated evidence."""
    if not payload or len(payload) > MAXIMUM_WORKER_FRAME_BYTES:
        raise WorkerProtocolError("Analysis worker response frame size is invalid")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerProtocolError(
            "Analysis worker response is not valid JSON"
        ) from error
    if not isinstance(raw, Mapping):
        raise WorkerProtocolError("Analysis worker response must be an object")
    if _integer(raw, "protocol_version") != ANALYSIS_WORKER_PROTOCOL_VERSION:
        raise WorkerProtocolError("Analysis worker response version is unsupported")
    request_id = _request_identifier(raw, "request_id")
    event = _string(raw, "event")
    if event == "error":
        return _failure_response(raw, request_id)
    if event == "item_result":
        return _item_success_response(raw, request_id)
    if event == "item_error":
        return _item_failure_response(raw, request_id)
    if event == "batch_finished":
        return _batch_finished_response(raw, request_id)
    if event != "result":
        raise WorkerProtocolError("Analysis worker response event is unsupported")
    _completed_terminal_status(raw)
    return WorkerSuccess(
        request_id,
        _parse_success_result(raw),
        _operation_metrics(raw.get("metrics")),
    )


def _parse_success_result(
    raw: Mapping[str, object],
) -> RecognitionResult | ForcedAlignmentResult | WorkerStatus | None:
    result_type = _string(raw, "result_type")
    result_raw = raw.get("result")
    if result_type == "recognition":
        return recognition_from_report(result_raw)
    if result_type == "forced_alignment":
        return forced_alignment_from_report(result_raw)
    if result_type == "status":
        return _worker_status(result_raw)
    if result_type == "empty" and result_raw is None:
        return None
    raise WorkerProtocolError("Analysis worker result type is unsupported")


def _result_fields(
    result: RecognitionResult | ForcedAlignmentResult | WorkerStatus | None,
) -> dict[str, object]:
    if isinstance(result, RecognitionResult):
        return {
            "result_type": "recognition",
            "result": recognition_report(result),
        }
    if isinstance(result, ForcedAlignmentResult):
        return {
            "result_type": "forced_alignment",
            "result": forced_alignment_report(result),
        }
    if isinstance(result, WorkerStatus):
        return {"result_type": "status", "result": asdict(result)}
    return {"result_type": "empty", "result": None}


def _failure_response(raw: Mapping[str, object], request_id: str) -> WorkerFailure:
    try:
        code = WorkerErrorCode(_string(raw, "code"))
    except ValueError as error:
        raise WorkerProtocolError(
            "Analysis worker error code is unsupported"
        ) from error
    retryable = raw.get("retryable")
    if not isinstance(retryable, bool):
        raise WorkerProtocolError("Analysis worker retryable field must be boolean")
    terminal = _operation_terminal_status(raw)
    if terminal is not _terminal_status_for_error(code):
        raise WorkerProtocolError("Analysis worker error terminal status is invalid")
    return WorkerFailure(
        request_id,
        code,
        _string(raw, "detail"),
        retryable,
        _operation_metrics(raw.get("metrics")),
    )


def _item_success_response(
    raw: Mapping[str, object], request_id: str
) -> WorkerItemSuccess:
    _completed_terminal_status(raw)
    result = _parse_success_result(raw)
    if not isinstance(result, RecognitionResult | ForcedAlignmentResult):
        raise WorkerProtocolError("Analysis batch item result type is invalid")
    return WorkerItemSuccess(
        request_id=request_id,
        batch_id=_request_identifier(raw, "batch_id"),
        index=_nonnegative_integer(raw, "index"),
        request_fingerprint=_sha256(raw, "request_fingerprint"),
        result=result,
        metrics=_operation_metrics(raw.get("metrics")),
    )


def _item_failure_response(
    raw: Mapping[str, object], request_id: str
) -> WorkerItemFailure:
    try:
        code = WorkerErrorCode(_string(raw, "code"))
    except ValueError as error:
        raise WorkerProtocolError(
            "Analysis worker item error code is unsupported"
        ) from error
    retryable = raw.get("retryable")
    if not isinstance(retryable, bool):
        raise WorkerProtocolError("Analysis worker retryable field must be boolean")
    terminal = _operation_terminal_status(raw)
    if terminal is OperationTerminalStatus.COMPLETED:
        raise WorkerProtocolError("Analysis worker item failure cannot be completed")
    return WorkerItemFailure(
        request_id=request_id,
        batch_id=_request_identifier(raw, "batch_id"),
        index=_nonnegative_integer(raw, "index"),
        request_fingerprint=_sha256(raw, "request_fingerprint"),
        code=code,
        detail=_string(raw, "detail"),
        retryable=retryable,
        metrics=_operation_metrics(raw.get("metrics")),
        terminal_status=terminal,
    )


def _batch_finished_response(
    raw: Mapping[str, object], request_id: str
) -> WorkerBatchFinished:
    _completed_terminal_status(raw)
    return WorkerBatchFinished(
        request_id=request_id,
        batch_id=_request_identifier(raw, "batch_id"),
        item_count=_nonnegative_integer(raw, "item_count"),
        metrics=_operation_metrics(raw.get("metrics")),
    )


def _operation_metrics(value: object) -> OperationMetrics:
    raw = _mapping(value, "operation metrics")
    _require_keys(
        raw,
        {
            "requested_audio_frames",
            "analyzed_audio_frames",
            "wall_time_ns",
            "peak_resident_memory_bytes",
            "peak_accelerator_memory_bytes",
            "cache_hit",
            "cache_miss_reason",
            "model_loads",
            "stage_timings",
        },
    )
    cache_hit = raw.get("cache_hit")
    if not isinstance(cache_hit, bool):
        raise WorkerProtocolError("Analysis worker cache-hit metric must be boolean")
    miss_reason = raw.get("cache_miss_reason")
    if miss_reason is not None and not isinstance(miss_reason, str):
        raise WorkerProtocolError("Analysis worker cache-miss reason must be text")
    return OperationMetrics(
        requested_audio_frames=_nonnegative_integer(raw, "requested_audio_frames"),
        analyzed_audio_frames=_nonnegative_integer(raw, "analyzed_audio_frames"),
        wall_time_ns=_nonnegative_integer(raw, "wall_time_ns"),
        peak_resident_memory_bytes=_nonnegative_integer(
            raw, "peak_resident_memory_bytes"
        ),
        peak_accelerator_memory_bytes=_optional_nonnegative_integer(
            raw, "peak_accelerator_memory_bytes"
        ),
        cache_hit=cache_hit,
        model_loads=_nonnegative_integer(raw, "model_loads"),
        cache_miss_reason=miss_reason,
        stage_timings=_operation_stage_timings(raw.get("stage_timings")),
    )


def _operation_stage_timings(value: object) -> OperationStageTimings:
    raw = _mapping(value, "operation stage timings")
    _require_keys(
        raw,
        {
            "preparing_audio_ns",
            "inference_ns",
            "normalizing_tokens_ns",
            "consensus_ns",
            "forced_alignment_ns",
            "writing_evidence_ns",
        },
    )
    return OperationStageTimings(
        preparing_audio_ns=_nonnegative_integer(raw, "preparing_audio_ns"),
        inference_ns=_nonnegative_integer(raw, "inference_ns"),
        normalizing_tokens_ns=_nonnegative_integer(raw, "normalizing_tokens_ns"),
        consensus_ns=_nonnegative_integer(raw, "consensus_ns"),
        forced_alignment_ns=_nonnegative_integer(raw, "forced_alignment_ns"),
        writing_evidence_ns=_nonnegative_integer(raw, "writing_evidence_ns"),
    )


def _operation_terminal_status(
    raw: Mapping[str, object],
) -> OperationTerminalStatus:
    try:
        return OperationTerminalStatus(_string(raw, "terminal_status"))
    except ValueError as error:
        raise WorkerProtocolError(
            "Analysis worker terminal status is unsupported"
        ) from error


def _completed_terminal_status(raw: Mapping[str, object]) -> None:
    if _operation_terminal_status(raw) is not OperationTerminalStatus.COMPLETED:
        raise WorkerProtocolError("Analysis worker result is not completed")


def _terminal_status_for_error(code: WorkerErrorCode) -> OperationTerminalStatus:
    if code is WorkerErrorCode.CANCELLED:
        return OperationTerminalStatus.CANCELLED
    if code is WorkerErrorCode.TIMEOUT:
        return OperationTerminalStatus.TIMED_OUT
    if code is WorkerErrorCode.WORKER_TERMINATED:
        return OperationTerminalStatus.WORKER_CRASHED
    return OperationTerminalStatus.INVALID_RESULT


def _worker_status(value: object) -> WorkerStatus:
    raw = _mapping(value, "worker status")
    _require_keys(
        raw,
        {
            "pid",
            "family",
            "python_version",
            "environment_fingerprint",
            "ready",
            "loaded_engines",
            "active_request_id",
            "completed_requests",
            "failed_requests",
            "model_loads",
            "peak_resident_memory_bytes",
            "metal_active_memory_bytes",
            "metal_peak_memory_bytes",
            "resident_memory_bytes",
        },
    )
    ready = raw.get("ready")
    active = raw.get("active_request_id")
    if not isinstance(ready, bool):
        raise WorkerProtocolError("Analysis worker ready field must be boolean")
    if active is not None and not isinstance(active, str):
        raise WorkerProtocolError("Analysis worker active request must be text")
    engines = raw.get("loaded_engines")
    if not isinstance(engines, list) or not all(
        isinstance(item, str) for item in engines
    ):
        raise WorkerProtocolError("Analysis worker loaded engines must be an array")
    model_loads = raw.get("model_loads")
    if not isinstance(model_loads, list):
        raise WorkerProtocolError("Analysis worker model loads must be an array")
    return WorkerStatus(
        pid=_integer(raw, "pid"),
        family=_string(raw, "family"),
        python_version=_string(raw, "python_version"),
        environment_fingerprint=_sha256(raw, "environment_fingerprint"),
        ready=ready,
        loaded_engines=tuple(cast(list[str], engines)),
        active_request_id=active,
        completed_requests=_integer(raw, "completed_requests"),
        failed_requests=_integer(raw, "failed_requests"),
        model_loads=tuple(_worker_model_load(item) for item in model_loads),
        peak_resident_memory_bytes=_integer(raw, "peak_resident_memory_bytes"),
        metal_active_memory_bytes=_optional_integer(raw, "metal_active_memory_bytes"),
        metal_peak_memory_bytes=_optional_integer(raw, "metal_peak_memory_bytes"),
        resident_memory_bytes=_integer(raw, "resident_memory_bytes"),
    )


def _worker_model_load(value: object) -> WorkerModelLoad:
    raw = _mapping(value, "worker model load")
    _require_keys(raw, {"engine", "count"})
    return WorkerModelLoad(_string(raw, "engine"), _integer(raw, "count"))


def _recognition_request(
    raw: Mapping[str, object], request_id: str
) -> RecognitionWorkerRequest:
    allowed = {
        "protocol_version",
        "operation",
        "request_id",
        "engine",
        "engine_fingerprint",
        "relative_audio_path",
        "audio_digest",
        "language",
        "span",
        "timeout_milliseconds",
    }
    _require_keys(raw, allowed)
    audio_digest = _sha256(raw, "audio_digest")
    return RecognitionWorkerRequest(
        request_id=request_id,
        engine=_string(raw, "engine"),
        engine_fingerprint=_sha256(raw, "engine_fingerprint"),
        relative_audio_path=_relative_path(raw, "relative_audio_path"),
        audio_digest=audio_digest,
        language=_language(raw),
        span=_matched_audio_span(raw.get("span"), audio_digest),
        timeout_milliseconds=_timeout_milliseconds(raw),
    )


def _forced_alignment_request(
    raw: Mapping[str, object], request_id: str
) -> ForcedAlignmentWorkerRequest:
    allowed = {
        "protocol_version",
        "operation",
        "request_id",
        "engine",
        "engine_fingerprint",
        "relative_audio_path",
        "audio_digest",
        "expected_text",
        "language",
        "purpose",
        "verified_span",
        "span",
        "timeout_milliseconds",
    }
    _require_keys(raw, allowed)
    expected_text = _string(raw, "expected_text")
    if len(expected_text.encode("utf-8")) > 64 * 1024:
        raise WorkerProtocolError("Forced-alignment text exceeds protocol limit")
    try:
        purpose = AlignmentPurpose(_string(raw, "purpose"))
    except ValueError as error:
        raise WorkerProtocolError("Forced-alignment purpose is invalid") from error
    verified_span = _optional_verified_span(raw.get("verified_span"))
    if purpose is not AlignmentPurpose.NON_AUTHORITATIVE and verified_span is None:
        raise WorkerProtocolError(
            "Authoritative forced alignment requires a verified lexical span"
        )
    audio_digest = _sha256(raw, "audio_digest")
    return ForcedAlignmentWorkerRequest(
        request_id=request_id,
        engine=_string(raw, "engine"),
        engine_fingerprint=_sha256(raw, "engine_fingerprint"),
        relative_audio_path=_relative_path(raw, "relative_audio_path"),
        audio_digest=audio_digest,
        expected_text=expected_text,
        language=_language(raw),
        purpose=purpose,
        verified_span=verified_span,
        span=_matched_audio_span(raw.get("span"), audio_digest),
        timeout_milliseconds=_timeout_milliseconds(raw),
    )


def _recognition_many_request(
    raw: Mapping[str, object], request_id: str
) -> RecognitionManyWorkerRequest:
    _require_keys(
        raw,
        {
            "protocol_version",
            "operation",
            "request_id",
            "batch_id",
            "engine",
            "engine_fingerprint",
            "timeout_milliseconds",
            "items",
        },
    )
    items = _batch_item_mappings(raw)
    parsed = tuple(_recognition_batch_item(item) for item in items)
    _validate_batch_indices(tuple(item.index for item in parsed))
    return RecognitionManyWorkerRequest(
        request_id=request_id,
        batch_id=_request_identifier(raw, "batch_id"),
        engine=_string(raw, "engine"),
        engine_fingerprint=_sha256(raw, "engine_fingerprint"),
        timeout_milliseconds=_timeout_milliseconds(raw),
        items=parsed,
    )


def _forced_alignment_many_request(
    raw: Mapping[str, object], request_id: str
) -> ForcedAlignmentManyWorkerRequest:
    _require_keys(
        raw,
        {
            "protocol_version",
            "operation",
            "request_id",
            "batch_id",
            "engine",
            "engine_fingerprint",
            "timeout_milliseconds",
            "items",
        },
    )
    items = _batch_item_mappings(raw)
    parsed = tuple(_forced_alignment_batch_item(item) for item in items)
    _validate_batch_indices(tuple(item.index for item in parsed))
    return ForcedAlignmentManyWorkerRequest(
        request_id=request_id,
        batch_id=_request_identifier(raw, "batch_id"),
        engine=_string(raw, "engine"),
        engine_fingerprint=_sha256(raw, "engine_fingerprint"),
        timeout_milliseconds=_timeout_milliseconds(raw),
        items=parsed,
    )


def _recognition_batch_item(raw: Mapping[str, object]) -> RecognitionBatchItem:
    _require_keys(
        raw,
        {
            "index",
            "request_fingerprint",
            "relative_audio_path",
            "audio_digest",
            "language",
            "span",
        },
    )
    audio_digest = _sha256(raw, "audio_digest")
    span = _matched_audio_span(raw.get("span"), audio_digest)
    return RecognitionBatchItem(
        index=_nonnegative_integer(raw, "index"),
        request_fingerprint=_sha256(raw, "request_fingerprint"),
        relative_audio_path=_relative_path(raw, "relative_audio_path"),
        audio_digest=audio_digest,
        language=_language(raw),
        span=span,
    )


def _forced_alignment_batch_item(
    raw: Mapping[str, object],
) -> ForcedAlignmentBatchItem:
    _require_keys(
        raw,
        {
            "index",
            "request_fingerprint",
            "relative_audio_path",
            "audio_digest",
            "expected_text",
            "language",
            "purpose",
            "verified_span",
            "span",
        },
    )
    expected_text = _string(raw, "expected_text")
    if len(expected_text.encode("utf-8")) > 64 * 1024:
        raise WorkerProtocolError("Forced-alignment text exceeds protocol limit")
    try:
        purpose = AlignmentPurpose(_string(raw, "purpose"))
    except ValueError as error:
        raise WorkerProtocolError("Forced-alignment purpose is invalid") from error
    verified_span = _optional_verified_span(raw.get("verified_span"))
    if purpose is not AlignmentPurpose.NON_AUTHORITATIVE and verified_span is None:
        raise WorkerProtocolError(
            "Authoritative forced alignment requires a verified lexical span"
        )
    audio_digest = _sha256(raw, "audio_digest")
    span = _matched_audio_span(raw.get("span"), audio_digest)
    return ForcedAlignmentBatchItem(
        index=_nonnegative_integer(raw, "index"),
        request_fingerprint=_sha256(raw, "request_fingerprint"),
        relative_audio_path=_relative_path(raw, "relative_audio_path"),
        audio_digest=audio_digest,
        expected_text=expected_text,
        language=_language(raw),
        purpose=purpose,
        verified_span=verified_span,
        span=span,
    )


def _recognition_batch_item_to_dict(
    item: RecognitionBatchItem,
) -> dict[str, object]:
    return {
        "index": item.index,
        "request_fingerprint": item.request_fingerprint,
        "relative_audio_path": item.relative_audio_path,
        "audio_digest": item.audio_digest,
        "language": item.language,
        "span": _span_to_dict(item.span),
    }


def _forced_alignment_batch_item_to_dict(
    item: ForcedAlignmentBatchItem,
) -> dict[str, object]:
    return {
        "index": item.index,
        "request_fingerprint": item.request_fingerprint,
        "relative_audio_path": item.relative_audio_path,
        "audio_digest": item.audio_digest,
        "expected_text": item.expected_text,
        "language": item.language,
        "purpose": item.purpose.value,
        "verified_span": _verified_span_to_dict(item.verified_span),
        "span": _span_to_dict(item.span),
    }


def _batch_item_mappings(
    raw: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    value = raw.get("items")
    if not isinstance(value, list) or not value or len(value) > MAXIMUM_BATCH_ITEMS:
        raise WorkerProtocolError("Analysis batch requires between 1 and 256 items")
    return tuple(_mapping(item, "batch item") for item in value)


def _validate_batch_indices(indices: tuple[int, ...]) -> None:
    if indices != tuple(sorted(set(indices))):
        raise WorkerProtocolError("Analysis batch indices must be unique and ordered")


def _timeout_milliseconds(raw: Mapping[str, object]) -> int:
    value = _integer(raw, "timeout_milliseconds")
    if not 1 <= value <= MAXIMUM_TIMEOUT_MILLISECONDS:
        raise WorkerProtocolError(
            "Analysis worker timeout must be between 1 and 3600000 milliseconds"
        )
    return value


def _require_keys(raw: Mapping[str, object], allowed: set[str]) -> None:
    unexpected = tuple(sorted(set(raw) - allowed))
    missing = tuple(sorted(allowed - set(raw)))
    if unexpected:
        raise WorkerProtocolError(f"Unexpected analysis worker field: {unexpected[0]}")
    if missing:
        raise WorkerProtocolError(f"Missing analysis worker field: {missing[0]}")


def _span_to_dict(span: AudioSpan | None) -> dict[str, object] | None:
    if span is None:
        return None
    return {
        "audio_digest": span.audio_digest,
        "start_frame": span.start_frame,
        "end_frame": span.end_frame,
        "sample_rate": span.sample_rate,
    }


def _verified_span_to_dict(
    span: VerifiedTextSpan | None,
) -> dict[str, object] | None:
    if span is None:
        return None
    return {
        "consensus_fingerprint": span.consensus_fingerprint,
        "token_start": span.token_start,
        "token_end": span.token_end,
        "lexical_span_hash": span.lexical_span_hash,
    }


def _optional_span(value: object) -> AudioSpan | None:
    if value is None:
        return None
    raw = _mapping(value, "span")
    _require_keys(raw, {"audio_digest", "start_frame", "end_frame", "sample_rate"})
    return AudioSpan(
        _sha256(raw, "audio_digest"),
        _integer(raw, "start_frame"),
        _integer(raw, "end_frame"),
        _integer(raw, "sample_rate"),
    )


def _required_span(value: object) -> AudioSpan:
    span = _optional_span(value)
    if span is None:
        raise WorkerProtocolError("Analysis inference request requires a sample span")
    return span


def _matched_audio_span(value: object, audio_digest: str) -> AudioSpan:
    span = _required_span(value)
    if span.audio_digest != audio_digest:
        raise WorkerProtocolError("Analysis request span and audio digest do not match")
    return span


def _optional_verified_span(value: object) -> VerifiedTextSpan | None:
    if value is None:
        return None
    raw = _mapping(value, "verified_span")
    _require_keys(
        raw,
        {
            "consensus_fingerprint",
            "token_start",
            "token_end",
            "lexical_span_hash",
        },
    )
    return VerifiedTextSpan(
        _sha256(raw, "consensus_fingerprint"),
        _integer(raw, "token_start"),
        _integer(raw, "token_end"),
        _sha256(raw, "lexical_span_hash"),
    )


def _encode_frame(value: Mapping[str, object], *, label: str) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    if not encoded or len(encoded) > MAXIMUM_WORKER_FRAME_BYTES:
        raise WorkerProtocolError(f"Encoded analysis worker {label} is too large")
    return encoded


def _parse_json_frame(payload: bytes, *, label: str) -> Mapping[str, object]:
    if not payload or len(payload) > MAXIMUM_WORKER_FRAME_BYTES:
        raise WorkerProtocolError(f"Analysis worker {label} size is invalid")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerProtocolError(
            f"Analysis worker {label} is not valid JSON"
        ) from error
    return _mapping(raw, label)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise WorkerProtocolError(f"Analysis worker {label} must be an object")
    return cast(Mapping[str, object], value)


def _request_identifier(raw: Mapping[str, object], key: str) -> str:
    value = _string(raw, key)
    if _REQUEST_ID.fullmatch(value) is None:
        raise WorkerProtocolError(f"Analysis worker {key} is invalid")
    return value


def _relative_path(raw: Mapping[str, object], key: str) -> str:
    value = _string(raw, key)
    candidate = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        "\\" in value
        or candidate.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in candidate.parts
        or value in {"", "."}
    ):
        raise WorkerProtocolError("Analysis worker audio path is unsafe")
    return candidate.as_posix()


def _language(raw: Mapping[str, object]) -> str:
    value = _string(raw, "language")
    if value != "en":
        raise WorkerProtocolError("Analysis worker currently accepts en only")
    return value


def _sha256(raw: Mapping[str, object], key: str) -> str:
    value = _string(raw, key)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise WorkerProtocolError(f"Analysis worker {key} must be a SHA-256")
    return value


def _string(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise WorkerProtocolError(f"Analysis worker {key} must be a string")
    return value


def _integer(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerProtocolError(f"Analysis worker {key} must be an integer")
    return value


def _nonnegative_integer(raw: Mapping[str, object], key: str) -> int:
    value = _integer(raw, key)
    if value < 0:
        raise WorkerProtocolError(f"Analysis worker {key} cannot be negative")
    return value


def _optional_integer(raw: Mapping[str, object], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerProtocolError(f"Analysis worker {key} must be an integer or null")
    return value


def _optional_nonnegative_integer(raw: Mapping[str, object], key: str) -> int | None:
    value = _optional_integer(raw, key)
    if value is not None and value < 0:
        raise WorkerProtocolError(f"Analysis worker {key} cannot be negative")
    return value


__all__ = [
    "ANALYSIS_WORKER_PROTOCOL_VERSION",
    "MAXIMUM_WORKER_FRAME_BYTES",
    "AnalysisWorkerRequest",
    "AnalysisWorkerResponse",
    "CancellationWorkerRequest",
    "ForcedAlignmentBatchItem",
    "ForcedAlignmentManyWorkerRequest",
    "ForcedAlignmentWorkerRequest",
    "RecognitionBatchItem",
    "RecognitionManyWorkerRequest",
    "RecognitionWorkerRequest",
    "ShutdownWorkerRequest",
    "StatusWorkerRequest",
    "UnloadWorkerRequest",
    "WorkerBatchFinished",
    "WorkerErrorCode",
    "WorkerFailure",
    "WorkerItemFailure",
    "WorkerItemSuccess",
    "WorkerModelLoad",
    "WorkerOperation",
    "WorkerStatus",
    "WorkerSuccess",
    "encode_worker_handshake",
    "encode_worker_request",
    "encode_worker_response",
    "parse_worker_handshake",
    "parse_worker_request",
    "parse_worker_response",
]
