from __future__ import annotations

import json
from dataclasses import replace

import pytest
from tests.unit.test_speech_analysis_contracts import _alignment

from yakbox.errors import ValidationError, WorkerProtocolError
from yakbox.speech.analysis_models import AlignmentPurpose, AudioSpan, VerifiedTextSpan
from yakbox.speech.analysis_protocol import (
    ForcedAlignmentBatchItem,
    ForcedAlignmentManyWorkerRequest,
    ForcedAlignmentWorkerRequest,
    WorkerErrorCode,
    WorkerFailure,
    WorkerItemFailure,
    WorkerItemSuccess,
    WorkerStatus,
    WorkerSuccess,
    encode_worker_request,
    encode_worker_response,
    parse_worker_request,
    parse_worker_response,
)
from yakbox.speech.analysis_scheduler import OperationMetrics, OperationTerminalStatus

SHA_A = "a" * 64
SHA_B = "b" * 64


def _request(
    purpose: AlignmentPurpose = AlignmentPurpose.VERIFIED_TARGET,
) -> ForcedAlignmentWorkerRequest:
    return ForcedAlignmentWorkerRequest(
        "request-1",
        "qwen-forced",
        SHA_B,
        "window.wav",
        SHA_A,
        "hello",
        "en",
        purpose,
        AudioSpan(SHA_A, 0, 1600, 16000),
        VerifiedTextSpan(SHA_B, 0, 1, SHA_A)
        if purpose is not AlignmentPurpose.NON_AUTHORITATIVE
        else None,
    )


def _batch() -> ForcedAlignmentManyWorkerRequest:
    request = _request()
    item = ForcedAlignmentBatchItem(
        0,
        SHA_B,
        request.relative_audio_path,
        request.audio_digest,
        request.expected_text,
        request.language,
        request.purpose,
        request.verified_span,
        request.span,
    )
    return ForcedAlignmentManyWorkerRequest(
        "request-1", "batch-1", "qwen-forced", SHA_B, 1000, (item,)
    )


@pytest.mark.parametrize("purpose", list(AlignmentPurpose))
def test_forced_alignment_frames_preserve_lexical_authority(
    purpose: AlignmentPurpose,
) -> None:
    request = _request(purpose)
    assert parse_worker_request(encode_worker_request(request)) == request
    batch = _batch()
    batch = replace(
        batch,
        items=(
            replace(
                batch.items[0], purpose=purpose, verified_span=request.verified_span
            ),
        ),
    )
    assert parse_worker_request(encode_worker_request(batch)) == batch


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_text", "x" * (64 * 1024 + 1)),
        ("purpose", "unsupported"),
        ("verified_span", None),
        ("verified_span", []),
        (
            "verified_span",
            {
                "consensus_fingerprint": SHA_B,
                "token_start": 1,
                "token_end": 0,
                "lexical_span_hash": SHA_A,
            },
        ),
        ("relative_audio_path", "../outside.wav"),
        ("relative_audio_path", "C:\\outside.wav"),
        ("audio_digest", "invalid"),
        ("language", ""),
        ("language", "x" * 100),
        ("span", None),
        (
            "span",
            {
                "audio_digest": SHA_B,
                "start_frame": 0,
                "end_frame": 1600,
                "sample_rate": 16000,
            },
        ),
        ("unexpected", True),
    ],
)
@pytest.mark.parametrize("batched", [False, True])
def test_forced_alignment_rejects_untrusted_item_fields(
    field: str, value: object, batched: bool
) -> None:
    raw = json.loads(encode_worker_request(_batch() if batched else _request()))
    item = raw["items"][0] if batched else raw
    item[field] = value
    with pytest.raises((WorkerProtocolError, ValidationError)):
        parse_worker_request(json.dumps(raw).encode())


@pytest.mark.parametrize("items", [[], None, [False], [{"index": -1}], "invalid"])
def test_forced_alignment_batch_requires_a_nonempty_typed_item_array(
    items: object,
) -> None:
    raw = json.loads(encode_worker_request(_batch()))
    raw["items"] = items
    with pytest.raises((WorkerProtocolError, ValidationError)):
        parse_worker_request(json.dumps(raw).encode())


def test_forced_alignment_batch_rejects_duplicate_and_negative_indices() -> None:
    batch = _batch()
    for items in [
        (batch.items[0], batch.items[0]),
        (replace(batch.items[0], index=-1),),
    ]:
        with pytest.raises((WorkerProtocolError, ValidationError)):
            parse_worker_request(encode_worker_request(replace(batch, items=items)))


@pytest.mark.parametrize("code", list(WorkerErrorCode))
def test_worker_error_frames_roundtrip_all_stable_codes(code: WorkerErrorCode) -> None:
    response = WorkerFailure("request-1", code, "Bounded failure", False)
    assert parse_worker_response(encode_worker_response(response)) == response
    item = WorkerItemFailure(
        "request-1",
        "batch-1",
        0,
        SHA_A,
        code,
        "Bounded failure",
        False,
        OperationMetrics(0, 0, 0, 0, None, False, 0),
        OperationTerminalStatus.INVALID_RESULT,
    )
    assert parse_worker_response(encode_worker_response(item)) == item


@pytest.mark.parametrize(
    "field,value",
    [
        ("code", "unsupported"),
        ("retryable", 1),
        ("terminal_status", "completed"),
        ("detail", "unsafe\nline"),
        ("detail", "x" * 513),
        ("index", True),
        ("request_fingerprint", "invalid"),
    ],
)
def test_batch_item_failure_frames_reject_invalid_contracts(
    field: str, value: object
) -> None:
    item = WorkerItemFailure(
        "request-1",
        "batch-1",
        0,
        SHA_A,
        WorkerErrorCode.TIMEOUT,
        "Bounded failure",
        True,
        OperationMetrics(0, 0, 0, 0, None, False, 0),
        OperationTerminalStatus.INVALID_RESULT,
    )
    raw = json.loads(encode_worker_response(item))
    raw[field] = value
    with pytest.raises((WorkerProtocolError, ValidationError)):
        parse_worker_response(json.dumps(raw).encode())


def _status() -> WorkerStatus:
    return WorkerStatus(
        123, "qwen", "3.14", SHA_A, True, (), None, 0, 0, (), 0, None, None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("ready", "true"),
        ("active_request_id", 10),
        ("loaded_engines", [1]),
        ("model_loads", {}),
        ("model_loads", [{"engine": "qwen", "count": -1}]),
        ("pid", 0),
        ("completed_requests", -1),
        ("failed_requests", -1),
        ("peak_resident_memory_bytes", -1),
        ("resident_memory_bytes", -1),
        ("metal_active_memory_bytes", -1),
        ("metal_peak_memory_bytes", -1),
    ],
)
def test_status_frames_validate_process_identity_counters_and_memory(
    field: str, value: object
) -> None:
    raw = json.loads(encode_worker_response(WorkerSuccess("status-1", _status())))
    raw["result"][field] = value
    with pytest.raises((WorkerProtocolError, ValidationError)):
        parse_worker_response(json.dumps(raw).encode())


@pytest.mark.parametrize(
    "field,value",
    [
        ("cache_hit", 1),
        ("cache_miss_reason", 42),
        ("model_loads", -1),
        ("peak_accelerator_memory_bytes", -1),
        ("requested_audio_frames", True),
        ("stage_timings", []),
        ("stage_timings", {"unexpected": 1}),
    ],
)
def test_worker_metrics_reject_invalid_types_and_negative_values(
    field: str, value: object
) -> None:
    raw = json.loads(encode_worker_response(WorkerSuccess("status-1", _status())))
    raw["metrics"][field] = value
    with pytest.raises((WorkerProtocolError, ValidationError)):
        parse_worker_response(json.dumps(raw).encode())


def test_forced_alignment_result_roundtrips_as_an_independently_committable_item() -> (
    None
):
    item = WorkerItemSuccess(
        "request-1",
        "batch-1",
        0,
        SHA_A,
        _alignment(),
        OperationMetrics(16000, 16000, 100, 100, 0, False, 1),
    )
    assert parse_worker_response(encode_worker_response(item)) == item
