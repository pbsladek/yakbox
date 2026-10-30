from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from yakbox.audio.crop import SpeechRegion
from yakbox.errors import BuildError, ValidationError
from yakbox.local_runtime import (
    _MAXIMUM_MESSAGE_BYTES,
    RUNTIME_PROTOCOL_VERSION,
    LocalRuntimeOptions,
    LocalRuntimeStatus,
    PersistentLocalSpeechService,
    PersistentMlxWhisperAligner,
    _artifact_from_runtime,
    _RuntimeServer,
    _send_endpoint,
    _wait_for_runtime,
    ensure_local_runtime,
    local_runtime_status,
    runtime_request,
    stop_local_runtime,
)
from yakbox.speech.alignment import AlignmentResult, AlignmentToken
from yakbox.speech.models import SpeechSynthesisRequest
from yakbox.speech.services import FakeSpeechService


def test_runtime_starts_without_loading_models(tmp_path: Path) -> None:
    server = _RuntimeServer(
        tmp_path,
        LocalRuntimeOptions(idle_timeout_seconds=30),
    )

    status = server.status()

    assert status["running"] is True
    assert status["tts_model_loaded"] is False
    assert status["whisper_models"] == 0
    assert status["conditioning_cache_entries"] == 0


def test_runtime_options_reject_unbounded_values() -> None:
    with pytest.raises(ValidationError, match="idle timeout"):
        LocalRuntimeOptions(idle_timeout_seconds=0)
    with pytest.raises(ValidationError, match="cache size"):
        LocalRuntimeOptions(conditioning_cache_size=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_runtime_idle_timeout_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValidationError, match="positive and finite"):
        LocalRuntimeOptions(idle_timeout_seconds=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("activity", ["recent", "busy", "idle"])
async def test_runtime_idle_timer_rechecks_recent_activity_and_active_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, activity: str
) -> None:
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        "yakbox.local_runtime.time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    server = _RuntimeServer(tmp_path, LocalRuntimeOptions(idle_timeout_seconds=10))
    if activity == "busy":
        await server.operation_lock.acquire()
    closed: list[bool] = []

    async def wait_closed() -> None:
        pass

    async def start_server(*_arguments: object, **_options: object) -> object:
        return SimpleNamespace(
            sockets=[SimpleNamespace(getsockname=lambda: ("127.0.0.1", 12345))],
            close=lambda: closed.append(True),
            wait_closed=wait_closed,
        )

    waits: list[float] = []

    async def wait_for(
        awaitable: Coroutine[object, object, bool],
        *,
        timeout: float,  # noqa: ASYNC109 - matches the mocked asyncio.wait_for boundary
    ) -> bool:
        awaitable.close()
        waits.append(timeout)
        if len(waits) == 1:
            clock.now = 10.0
            if activity == "recent":
                server.last_activity = 9.0
            raise TimeoutError
        assert (tmp_path / ".yakbox/runtime/endpoint.json").is_file()
        server.stop_event.set()
        return True

    monkeypatch.setattr("yakbox.local_runtime.asyncio.start_server", start_server)
    monkeypatch.setattr("yakbox.local_runtime.asyncio.wait_for", wait_for)

    await server.serve()

    expected = [10] if activity == "idle" else [10, 10 if activity == "busy" else 9]
    assert waits == expected
    assert closed == [True]
    assert not (tmp_path / ".yakbox/runtime/endpoint.json").exists()


@pytest_asyncio.fixture
async def running_runtime(tmp_path: Path) -> AsyncIterator[_RuntimeServer]:
    server = _RuntimeServer(tmp_path, LocalRuntimeOptions(idle_timeout_seconds=30))
    task = asyncio.create_task(server.serve())
    try:
        await _wait_for_runtime(tmp_path, timeout_seconds=5)
        yield server
    finally:
        server.stop_event.set()
        await asyncio.wait_for(task, timeout=5)


def _sized_payload(payload: dict[str, object], frame_bytes: int) -> dict[str, object]:
    payload = {**payload, "padding": ""}
    overhead = len(json.dumps(payload, ensure_ascii=False).encode()) + 1
    return {**payload, "padding": "x" * (frame_bytes - overhead)}


@pytest.mark.asyncio
async def test_runtime_synthesizes_132_chunks_above_default_stream_limit(
    running_runtime: _RuntimeServer,
) -> None:
    running_runtime.tts_services["cpu"] = FakeSpeechService()
    service = PersistentLocalSpeechService(
        running_runtime.workspace,
        device="cpu",
        options=running_runtime.options,
        request_timeout_seconds=5,
    )
    requests = tuple(
        (
            SpeechSynthesisRequest(
                text=f"Chunk {index}. " + "A sentence. " * 60, voice="narrator"
            ),
            running_runtime.workspace / f"chunk-{index:03d}.wav",
        )
        for index in range(132)
    )

    artifacts = await service.synthesize_many_to_files(requests)

    assert tuple(artifact.path for artifact in artifacts) == tuple(
        destination for _, destination in requests
    )
    assert all(artifact.path.is_file() for artifact in artifacts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame_bytes",
    [
        65_538,
        _MAXIMUM_MESSAGE_BYTES,
        _MAXIMUM_MESSAGE_BYTES + 1,
        _MAXIMUM_MESSAGE_BYTES + 2,
    ],
)
async def test_runtime_request_message_limit(
    running_runtime: _RuntimeServer, frame_bytes: int
) -> None:
    endpoint: dict[str, object] = {
        "port": running_runtime.port,
        "token": running_runtime.token,
    }
    request = _sized_payload(
        {
            "protocol_version": RUNTIME_PROTOCOL_VERSION,
            "token": running_runtime.token,
            "operation": "status",
        },
        frame_bytes,
    )
    if frame_bytes > _MAXIMUM_MESSAGE_BYTES:
        with pytest.raises(BuildError, match="runtime failed"):
            await _send_endpoint(endpoint, request, timeout_seconds=5)
    else:
        response = await _send_endpoint(endpoint, request, timeout_seconds=5)
        assert response["running"] is True
    healthy = await _send_endpoint(endpoint, {"operation": "status"}, timeout_seconds=5)
    assert healthy["running"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame_bytes",
    [
        65_538,
        _MAXIMUM_MESSAGE_BYTES,
        _MAXIMUM_MESSAGE_BYTES + 1,
        _MAXIMUM_MESSAGE_BYTES + 2,
    ],
)
async def test_runtime_response_message_limit(
    running_runtime: _RuntimeServer, monkeypatch: pytest.MonkeyPatch, frame_bytes: int
) -> None:
    response = _sized_payload(
        {"protocol_version": RUNTIME_PROTOCOL_VERSION, "ok": True}, frame_bytes
    )
    monkeypatch.setattr(running_runtime, "status", lambda: response)
    endpoint: dict[str, object] = {
        "port": running_runtime.port,
        "token": running_runtime.token,
    }
    if frame_bytes > _MAXIMUM_MESSAGE_BYTES:
        with pytest.raises(BuildError, match="no valid response"):
            await _send_endpoint(endpoint, {"operation": "status"}, timeout_seconds=5)
    else:
        result = await _send_endpoint(
            endpoint, {"operation": "status"}, timeout_seconds=5
        )
        assert result == response


@pytest.mark.asyncio
async def test_runtime_status_and_stop_are_authenticated_and_idempotent(
    running_runtime: _RuntimeServer,
) -> None:

    status = await local_runtime_status(running_runtime.workspace)
    assert status.running
    assert status.port == running_runtime.port
    assert "token" not in status.to_dict()
    assert await stop_local_runtime(running_runtime.workspace)
    assert running_runtime.stop_event.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["status", "stop", "synthesize_many"])
async def test_runtime_wrong_token_never_executes_an_operation(
    running_runtime: _RuntimeServer, operation: str
) -> None:
    endpoint: dict[str, object] = {
        "port": running_runtime.port,
        "token": "incorrect-token",
    }
    with pytest.raises(BuildError, match="runtime failed"):
        await _send_endpoint(endpoint, {"operation": operation}, timeout_seconds=2)
    assert not running_runtime.stop_event.is_set()
    assert not running_runtime.tts_services


@pytest.mark.asyncio
async def test_runtime_rejects_unknown_operations_and_incompatible_protocol(
    running_runtime: _RuntimeServer,
) -> None:
    endpoint: dict[str, object] = {
        "port": running_runtime.port,
        "token": running_runtime.token,
    }
    requests: list[dict[str, object]] = [
        {"operation": "unknown"},
        {"operation": "status", "protocol_version": 99},
    ]
    for request in requests:
        with pytest.raises(BuildError, match="runtime failed"):
            await _send_endpoint(endpoint, request, timeout_seconds=2)
    assert running_runtime.status()["running"]


@pytest.mark.asyncio
async def test_runtime_alignment_reuses_model_and_preserves_window_evidence(
    running_runtime: _RuntimeServer, monkeypatch: pytest.MonkeyPatch
) -> None:

    calls: list[tuple[float | None, float | None]] = []

    class Aligner:
        fingerprint = "test-aligner"

        async def align(
            self, _audio: Path, _text: str, *, language: str
        ) -> AlignmentResult:
            calls.append((None, None))
            return AlignmentResult(
                tokens=(AlignmentToken("Hello", 0.1, 0.3, 0.98),),
                speech_regions=(SpeechRegion(0.1, 0.3),),
                backend="test",
                model="fixture",
                fingerprint=self.fingerprint,
                language=language,
                transcript="Hello",
            )

        async def align_window(
            self,
            audio: Path,
            text: str,
            *,
            language: str,
            start_seconds: float,
            end_seconds: float,
        ) -> AlignmentResult:
            result = await self.align(audio, text, language=language)
            calls[-1] = (start_seconds, end_seconds)
            return result

    loads: list[str] = []

    def load(backend: str, **_options: object) -> Aligner:
        loads.append(backend)
        return Aligner()

    monkeypatch.setattr("yakbox.local_alignment.open_local_aligner", load)
    aligner = PersistentMlxWhisperAligner(
        running_runtime.workspace,
        model="fixture",
        revision="reviewed",
        timeout_seconds=2,
        prompted_timing=False,
        decode_consensus=False,
        prompt_sensitivity=False,
        maximum_consensus_timing_delta_ms=100,
        hallucination_silence_threshold=0.5,
        runtime_options=running_runtime.options,
    )
    assert aligner.fingerprint
    audio = running_runtime.workspace / "speech.wav"
    first = await aligner.align(audio, "Hello", language="en")
    second = await aligner.align_window(
        audio, "Hello", language="en", start_seconds=1, end_seconds=2
    )
    assert first == second
    assert first.transcript == "Hello"
    assert first.tokens[0].confidence == 0.98
    assert calls == [(None, None), (1, 2)]
    assert loads == ["mlx-whisper"]
    assert running_runtime.status()["whisper_models"] == 1


@pytest.mark.asyncio
async def test_runtime_enforces_memory_before_and_after_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    server = _RuntimeServer(tmp_path, LocalRuntimeOptions(maximum_memory_bytes=100))
    operation = AsyncMock(return_value={"items": []})
    monkeypatch.setattr(server, "_synthesize", operation)
    monkeypatch.setattr("yakbox.local_runtime._resident_memory_bytes", lambda: 101)
    with pytest.raises(BuildError, match="memory limit"):
        await server.execute({"operation": "synthesize_many"})
    operation.assert_not_called()
    assert server.stop_event.is_set()
    server.stop_event.clear()
    readings = iter([90, 101])
    monkeypatch.setattr(
        "yakbox.local_runtime._resident_memory_bytes", lambda: next(readings)
    )
    assert await server.execute({"operation": "synthesize_many"}) == {"items": []}
    operation.assert_awaited_once()
    assert server.stop_event.is_set()


@pytest.mark.asyncio
async def test_runtime_refuses_a_second_tts_device(tmp_path: Path) -> None:
    server = _RuntimeServer(tmp_path, LocalRuntimeOptions())
    server.tts_services["cpu"] = FakeSpeechService()
    with pytest.raises(ValidationError, match="one resident TTS device"):
        await server.execute({"operation": "synthesize_many", "device": "mps"})
    with pytest.raises(ValidationError, match="must not be empty"):
        await server.execute(
            {"operation": "synthesize_many", "device": "cpu", "items": []}
        )
    with pytest.raises(ValidationError, match="configuration is invalid"):
        await server.execute({"operation": "align"})


@pytest.mark.asyncio
async def test_runtime_starts_under_lock_and_cleans_lock_after_spawn_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    stopped = LocalRuntimeStatus(running=False)
    ready = LocalRuntimeStatus(running=True, pid=123)
    monkeypatch.setattr(
        "yakbox.local_runtime.local_runtime_status", AsyncMock(return_value=stopped)
    )
    wait = AsyncMock(return_value=ready)
    monkeypatch.setattr("yakbox.local_runtime._wait_for_runtime", wait)
    spawn = Mock()
    monkeypatch.setattr("yakbox.local_runtime.subprocess.Popen", spawn)
    assert (
        await ensure_local_runtime(
            tmp_path, options=LocalRuntimeOptions(maximum_memory_bytes=100)
        )
        == ready
    )
    arguments = spawn.call_args.args[0]
    assert arguments[-2:] == ["--maximum-memory-bytes", "100"]
    assert spawn.call_args.kwargs["start_new_session"] is True
    assert not (tmp_path / ".yakbox/runtime/starting.lock").exists()
    spawn.side_effect = OSError("spawn failed")
    with pytest.raises(OSError, match="spawn failed"):
        await ensure_local_runtime(tmp_path)
    assert not (tmp_path / ".yakbox/runtime/starting.lock").exists()


@pytest.mark.asyncio
async def test_runtime_recovers_stale_start_lock_but_respects_an_active_starter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    lock = tmp_path / ".yakbox/runtime/starting.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("broken-pid")
    ready = LocalRuntimeStatus(running=True)
    monkeypatch.setattr(
        "yakbox.local_runtime.local_runtime_status",
        AsyncMock(return_value=LocalRuntimeStatus(running=False)),
    )
    wait = AsyncMock(side_effect=[BuildError("starter timed out"), ready])
    monkeypatch.setattr("yakbox.local_runtime._wait_for_runtime", wait)
    spawn = Mock()
    monkeypatch.setattr("yakbox.local_runtime.subprocess.Popen", spawn)
    assert await ensure_local_runtime(tmp_path) == ready
    assert spawn.call_count == 1
    lock.write_text(str(os.getpid()))
    wait.side_effect = BuildError("starter still active")
    with pytest.raises(BuildError, match="starter still active"):
        await ensure_local_runtime(tmp_path)
    assert lock.exists()
    assert spawn.call_count == 1


@pytest.mark.asyncio
async def test_missing_runtime_endpoint_cannot_be_used_without_ensure(
    tmp_path: Path,
) -> None:

    assert not (await local_runtime_status(tmp_path)).running
    assert not await stop_local_runtime(tmp_path)
    with pytest.raises(BuildError, match="no endpoint"):
        await runtime_request(
            tmp_path,
            {"operation": "status"},
            options=LocalRuntimeOptions(),
            timeout_seconds=1,
            ensure=False,
        )


@pytest.mark.parametrize("value", [None, [], {"output_format": "unknown"}])
def test_runtime_rejects_malformed_synthesis_results(value: object) -> None:

    with pytest.raises(BuildError, match="invalid"):
        _artifact_from_runtime(value)
