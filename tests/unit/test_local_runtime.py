from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from yakbox.errors import BuildError, ValidationError
from yakbox.local_runtime import (
    _MAXIMUM_MESSAGE_BYTES,
    RUNTIME_PROTOCOL_VERSION,
    LocalRuntimeOptions,
    PersistentLocalSpeechService,
    _RuntimeServer,
    _send_endpoint,
    _wait_for_runtime,
)
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
