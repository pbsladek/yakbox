from __future__ import annotations

import io
import json
import socket
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.unit.test_speech_analysis_contracts import (
    SHA_A,
    SHA_B,
    SHA_C,
    _alignment,
    _FakeFactory,
    _recognition,
    _write_wav,
)

import yakbox.speech.analysis_worker as worker
import yakbox.speech.workers as local_worker
from yakbox.errors import (
    BackendUnavailableError,
    ModelIntegrityError,
    ValidationError,
    WorkerProtocolError,
)
from yakbox.speech.analysis_models import AlignmentPurpose
from yakbox.speech.analysis_protocol import (
    ForcedAlignmentBatchItem,
    ForcedAlignmentManyWorkerRequest,
    ShutdownWorkerRequest,
    WorkerBatchFinished,
    WorkerErrorCode,
    WorkerFailure,
    WorkerItemFailure,
    WorkerItemSuccess,
    parse_worker_handshake,
    parse_worker_response,
)
from yakbox.speech.analysis_scheduler import (
    OperationTerminalStatus,
    build_worker_handshake,
)
from yakbox.speech.analysis_services import FakeForcedAligner, FakeSpeechRecognizer
from yakbox.speech.services import FakeSpeechService


def _application(root: Path) -> worker.AnalysisWorkerApplication:
    return worker.AnalysisWorkerApplication(
        family="qwen",
        audio_root=root,
        factory=_FakeFactory(
            FakeSpeechRecognizer(
                SHA_B,
                lambda _path, _language, span: _recognition(
                    "qwen", ("hello",), span=span
                ),
            ),
            FakeForcedAligner(
                SHA_C,
                lambda _path, _text, _language, _purpose, _verified, span: _alignment(
                    span
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_forced_alignment_batch_isolates_corrupt_audio_and_reuses_model(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "clip.wav"
    span = _write_wav(audio)
    application = _application(tmp_path)
    good = ForcedAlignmentBatchItem(
        1,
        SHA_A,
        audio.name,
        span.audio_digest,
        "hello",
        "en",
        AlignmentPurpose.NON_AUTHORITATIVE,
        None,
        span,
    )
    bad = replace(good, index=0, audio_digest=SHA_B)
    request = ForcedAlignmentManyWorkerRequest(
        "request-1",
        "batch-1",
        "qwen-forced",
        SHA_C,
        1000,
        (bad, good, replace(good, index=2)),
    )
    results = [item async for item in application.handle_batch(request)]
    assert isinstance(results[0], WorkerItemFailure)
    assert results[0].code is WorkerErrorCode.MODEL_INVALID
    assert isinstance(results[1], WorkerItemSuccess)
    assert isinstance(results[2], WorkerItemSuccess)
    assert results[1].metrics.model_loads == 1
    assert results[2].metrics.model_loads == 0
    assert results[-1] == WorkerBatchFinished("request-1", "batch-1", 3)
    assert application.status().completed_requests == 2
    assert application.status().failed_requests == 1
    await application.handle(ShutdownWorkerRequest("shutdown-1"))
    assert not application.status().loaded_engines


@pytest.mark.asyncio
async def test_expired_analysis_batch_reports_every_item_without_loading_models(
    tmp_path: Path,
) -> None:
    span = _write_wav(tmp_path / "clip.wav")
    item = ForcedAlignmentBatchItem(
        0,
        SHA_A,
        "clip.wav",
        span.audio_digest,
        "hello",
        "en",
        AlignmentPurpose.NON_AUTHORITATIVE,
        None,
        span,
    )
    application = _application(tmp_path)
    request = ForcedAlignmentManyWorkerRequest(
        "request-1", "batch-1", "qwen-forced", SHA_C, 0, (item,)
    )
    results = [response async for response in application.handle_batch(request)]
    assert isinstance(results[0], WorkerItemFailure)
    assert results[0].terminal_status is OperationTerminalStatus.TIMED_OUT
    assert results[0].metrics.analyzed_audio_frames == 0
    assert not application.status().loaded_engines


@pytest.mark.asyncio
async def test_worker_protocol_recovers_after_bad_frames_and_finishes_at_eof(
    tmp_path: Path,
) -> None:
    handshake = build_worker_handshake(
        family="qwen",
        engines=("qwen", "qwen-forced"),
        worker_artifact_fingerprint=SHA_A,
        environment_lock_fingerprint=SHA_B,
        adapter_fingerprint=SHA_C,
    )
    frames = iter([b"no newline", b"not-json\n", b"[]\n", b""])
    output: list[bytes] = []
    result = await worker._serve(
        _application(tmp_path), handshake, lambda _limit: next(frames), output.append
    )
    assert result == 0
    assert parse_worker_handshake(output[0].strip()) == handshake
    responses = [parse_worker_response(frame.strip()) for frame in output[1:]]
    assert len(responses) == 3
    assert all(
        isinstance(response, WorkerFailure)
        and response.code is WorkerErrorCode.INVALID_REQUEST
        for response in responses
    )


@pytest.mark.parametrize(
    "error,code",
    [
        (WorkerProtocolError("private manuscript"), WorkerErrorCode.INVALID_REQUEST),
        (ValidationError("private manuscript"), WorkerErrorCode.INVALID_REQUEST),
        (BackendUnavailableError("private path"), WorkerErrorCode.MODEL_UNAVAILABLE),
        (ModelIntegrityError("private path"), WorkerErrorCode.MODEL_INVALID),
        (TimeoutError("private path"), WorkerErrorCode.TIMEOUT),
        (RuntimeError("private manuscript"), WorkerErrorCode.INFERENCE_FAILED),
    ],
)
def test_worker_failures_preserve_codes_and_redact_exception_details(
    error: Exception, code: WorkerErrorCode
) -> None:
    response = worker._failure("request-1", error)
    assert response.code is code
    assert "private" not in response.detail
    assert response.retryable is (
        code in {WorkerErrorCode.TIMEOUT, WorkerErrorCode.INFERENCE_FAILED}
    )


def test_worker_network_audit_denies_internet_but_allows_local_ipc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hooks: list[Callable[[str, tuple[object, ...]], None]] = []
    monkeypatch.setattr(worker.sys, "addaudithook", hooks.append)
    worker._install_offline_audit_guard()
    hook = hooks[0]
    for family in (socket.AF_INET, socket.AF_INET6):
        with pytest.raises(PermissionError, match="cannot create network"):
            hook("socket.__new__", (None, family))
    for event in ("socket.getaddrinfo", "socket.gethostbyaddr", "socket.gethostbyname"):
        with pytest.raises(PermissionError, match="cannot resolve"):
            hook(event, ("example.invalid",))
    hook("socket.__new__", (None, socket.AF_UNIX))
    hook("open", ("local.wav",))


def _local_request(destination: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_version": local_worker.WORKER_PROTOCOL_VERSION,
        "operation": "synthesize_many",
        "device": "cpu",
        "items": [
            {
                "text": "A sentence.",
                "voice": "narrator",
                "destination": str(destination),
            }
        ],
    }


@pytest.mark.asyncio
async def test_local_worker_file_protocol_produces_verified_artifact_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = Mock(return_value=FakeSpeechService())
    monkeypatch.setattr("yakbox.local.LocalChatterboxService", factory)
    request = tmp_path / "request.json"
    result = tmp_path / "result.json"
    request.write_text(json.dumps(_local_request(tmp_path / "speech.wav")))
    await local_worker._worker_main(request, result)
    items = local_worker._read_results(result)
    assert len(items) == 1
    artifact = local_worker._artifact_from_result(items[0])
    assert artifact.path.is_file()
    assert artifact.bytes_written == artifact.path.stat().st_size
    factory.assert_called_once_with(device="cpu")


@pytest.mark.asyncio
async def test_local_worker_server_recovers_after_invalid_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = Mock(return_value=FakeSpeechService())
    monkeypatch.setattr("yakbox.local.LocalChatterboxService", factory)
    frames = [
        [],
        {"operation": "bad"},
        _local_request(tmp_path / "speech.wav"),
        {"operation": "shutdown"},
    ]
    input_stream = io.BytesIO(
        b"\n".join(json.dumps(frame).encode() for frame in frames) + b"\n"
    )
    output = io.StringIO()
    monkeypatch.setattr(local_worker.sys, "stdin", SimpleNamespace(buffer=input_stream))
    monkeypatch.setattr(local_worker.sys, "stdout", output)
    await local_worker._worker_server("cpu")
    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [item["event"] for item in messages] == [
        "error",
        "error",
        "started",
        "result",
        "complete",
    ]
    assert messages[-1]["count"] == 1
    assert "bad" not in messages[1]["error"]
    factory.assert_called_once_with(device="cpu")


@pytest.mark.parametrize(
    "content",
    [
        "invalid-json",
        "[]",
        '{"schema_version":99}',
        '{"schema_version":1,"items":[null]}',
    ],
)
def test_local_worker_rejects_malformed_result_documents(
    tmp_path: Path, content: str
) -> None:
    result = tmp_path / "result.json"
    result.write_text(content)
    with pytest.raises(local_worker.BuildError):
        local_worker._read_results(result)
