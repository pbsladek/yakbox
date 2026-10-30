from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from tests.unit.test_speech_analysis_contracts import (
    SHA_A,
    SHA_B,
    SHA_C,
    _alignment,
    _recognition,
    _write_wav,
)

from yakbox.errors import SpeechAnalysisError, ValidationError, WorkerProtocolError
from yakbox.speech.analysis_models import AlignmentPurpose
from yakbox.speech.analysis_protocol import (
    AnalysisWorkerResponse,
    RecognitionBatchItem,
    RecognitionManyWorkerRequest,
    WorkerBatchFinished,
    WorkerErrorCode,
    WorkerFailure,
    WorkerItemFailure,
    WorkerItemSuccess,
    WorkerSuccess,
)
from yakbox.speech.analysis_runtime import (
    BUILT_IN_WORKERS,
    IsolatedAnalysisWorker,
    WorkerBackedForcedAligner,
)
from yakbox.speech.analysis_scheduler import OperationMetrics


def _worker(root: Path, family: str = "whisper") -> IsolatedAnalysisWorker:
    return IsolatedAnalysisWorker(
        BUILT_IN_WORKERS[family],
        audio_root=root,
        model_root=root,
        calibration_fingerprint=SHA_A,
        worker_artifact_digest=SHA_B,
        lock_digest=SHA_C,
    )


def _request() -> RecognitionManyWorkerRequest:
    item = RecognitionBatchItem(
        0, SHA_C, "window.wav", SHA_A, "en", _recognition("whisper", ("hello",)).span
    )
    return RecognitionManyWorkerRequest(
        "request-1", "batch-1", "whisper", SHA_B, 1000, (item,)
    )


def _success() -> WorkerItemSuccess:
    return WorkerItemSuccess(
        "request-1",
        "batch-1",
        0,
        SHA_C,
        _recognition("whisper", ("hello",)),
        OperationMetrics(16000, 16000, 1, 1, None, False, 1),
    )


@pytest.mark.asyncio
async def test_analysis_batch_streams_verified_items_and_keeps_worker_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker(tmp_path)
    process = SimpleNamespace(stdin=SimpleNamespace(write=Mock(), drain=AsyncMock()))
    monkeypatch.setattr(worker, "_ensure_process", AsyncMock(return_value=process))
    monkeypatch.setattr(
        worker,
        "_read_response",
        AsyncMock(
            side_effect=[_success(), WorkerBatchFinished("request-1", "batch-1", 1)]
        ),
    )
    terminate = AsyncMock()
    monkeypatch.setattr(worker, "_terminate", terminate)
    responses = [
        item async for item in worker.request_many(_request(), timeout_seconds=1)
    ]
    assert responses == [_success()]
    assert b'"recognize_many"' in process.stdin.write.call_args.args[0]
    assert worker._active_request_id is None
    await worker.wait_idle()
    terminate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        WorkerBatchFinished("request-1", "batch-1", 1),
        WorkerBatchFinished("wrong", "batch-1", 1),
        WorkerSuccess("request-1", None),
        replace(_success(), request_fingerprint=SHA_B),
        replace(_success(), batch_id="wrong"),
    ],
)
async def test_analysis_batch_discards_workers_after_invalid_or_incomplete_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: AnalysisWorkerResponse
) -> None:
    worker = _worker(tmp_path)
    monkeypatch.setattr(worker, "_start_batch", AsyncMock(return_value=object()))
    monkeypatch.setattr(worker, "_read_response", AsyncMock(return_value=response))
    terminate = AsyncMock()
    monkeypatch.setattr(worker, "_terminate", terminate)
    with pytest.raises(WorkerProtocolError):
        _ = [item async for item in worker.request_many(_request(), timeout_seconds=1)]
    terminate.assert_awaited_once()
    assert worker._active_request_id is None
    await worker.wait_idle()


@pytest.mark.asyncio
async def test_analysis_batch_deadline_includes_stalled_request_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker(tmp_path)

    async def stalled(_request: object) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(worker, "_start_batch", stalled)
    terminate = AsyncMock()
    monkeypatch.setattr(worker, "_terminate", terminate)
    with pytest.raises(TimeoutError):
        _ = [
            item async for item in worker.request_many(_request(), timeout_seconds=0.01)
        ]
    terminate.assert_awaited_once()
    with pytest.raises(ValidationError):
        _ = [item async for item in worker.request_many(_request(), timeout_seconds=0)]


@pytest.mark.asyncio
async def test_closing_a_partially_consumed_analysis_batch_terminates_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker(tmp_path)
    monkeypatch.setattr(worker, "_start_batch", AsyncMock(return_value=object()))
    monkeypatch.setattr(worker, "_read_response", AsyncMock(return_value=_success()))
    terminate = AsyncMock()
    monkeypatch.setattr(worker, "_terminate", terminate)
    stream = cast(
        AsyncGenerator[WorkerItemSuccess | WorkerItemFailure],
        worker.request_many(_request(), timeout_seconds=1),
    )
    assert await anext(stream) == _success()
    await stream.aclose()
    terminate.assert_awaited_once()
    assert worker._active_request_id is None


@pytest.mark.asyncio
async def test_analysis_soft_cancel_sends_protocol_request_without_killing_worker(
    tmp_path: Path,
) -> None:
    worker = _worker(tmp_path)
    assert (await worker.soft_cancel("unknown")).code is WorkerErrorCode.INVALID_REQUEST
    process = SimpleNamespace(
        returncode=None, stdin=SimpleNamespace(write=Mock(), drain=AsyncMock())
    )
    worker._process = cast(asyncio.subprocess.Process, process)
    worker._active_request_id = "request-1"
    assert (await worker.soft_cancel("request-1")).code is WorkerErrorCode.CANCELLED
    assert b'"cancel"' in process.stdin.write.call_args.args[0]
    assert "request-1" in worker._cancelled_request_ids
    process.stdin.drain.side_effect = BrokenPipeError
    assert (await worker.soft_cancel("request-1")).code is WorkerErrorCode.CANCELLED


@pytest.mark.asyncio
async def test_forced_aligner_client_preserves_exact_materialized_audio_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "window.wav"
    span = _write_wav(audio)
    worker = _worker(tmp_path, "qwen")
    evidence = _alignment(span)
    request = AsyncMock(return_value=WorkerSuccess("force-align-1", evidence))
    monkeypatch.setattr(worker, "request", request)
    aligner = WorkerBackedForcedAligner(
        engine="qwen-forced",
        worker=worker,
        audio_root=tmp_path,
        adapter_fingerprint=SHA_B,
        timeout_seconds=2,
    )
    assert aligner.fingerprint
    result = await aligner.force_align(
        audio,
        "hello",
        language="en",
        purpose=AlignmentPurpose.NON_AUTHORITATIVE,
        span=span,
    )
    assert result == evidence
    actual = request.call_args.args[0]
    assert actual.audio_digest == span.audio_digest
    assert actual.relative_audio_path == "window.wav"
    assert actual.expected_text == "hello"
    assert actual.timeout_milliseconds == 2000
    with pytest.raises(ValidationError, match="exact span"):
        await aligner.force_align(
            audio,
            "hello",
            language="en",
            purpose=AlignmentPurpose.NON_AUTHORITATIVE,
            span=replace(span, end_frame=100),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        WorkerFailure(
            "failure", WorkerErrorCode.MODEL_UNAVAILABLE, "Unavailable", False
        ),
        WorkerBatchFinished("request-1", "batch-1", 0),
        WorkerSuccess("success", None),
    ],
)
async def test_forced_aligner_client_rejects_failures_and_wrong_evidence_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: AnalysisWorkerResponse
) -> None:
    audio = tmp_path / "window.wav"
    _write_wav(audio)
    worker = _worker(tmp_path, "qwen")
    monkeypatch.setattr(worker, "request", AsyncMock(return_value=response))
    aligner = WorkerBackedForcedAligner(
        engine="qwen-forced",
        worker=worker,
        audio_root=tmp_path,
        adapter_fingerprint=SHA_B,
        timeout_seconds=2,
    )
    with pytest.raises((SpeechAnalysisError, WorkerProtocolError)):
        await aligner.force_align(
            audio, "hello", language="en", purpose=AlignmentPurpose.NON_AUTHORITATIVE
        )
