from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from tests.unit.test_whisper_tooling import _alignment

import yakbox.local_runtime as runtime
from yakbox.contracts import schema_uri
from yakbox.errors import BuildError
from yakbox.speech.alignment import DecodePassEvidence


def test_alignment_roundtrip_preserves_segments_and_independent_decodes() -> None:
    original = replace(
        _alignment("hello"),
        decode_passes=(
            DecodePassEvidence("independent", "hello", ("hello",), (), 0.9, True),
        ),
        consensus_score=1.0,
        maximum_timing_delta_ms=5.0,
        clip_start_seconds=0.0,
        clip_end_seconds=1.0,
    )
    restored = runtime._alignment_from_dict(
        json.loads(json.dumps(runtime._alignment_to_dict(original)))
    )
    assert restored == original
    assert restored.segments[0].average_log_probability == -0.2
    assert restored.decode_passes[0].matches_expected is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("tokens", None),
        ("tokens", [False]),
        ("tokens", [{}]),
        ("speech_regions", [False]),
        ("segments", [False]),
        ("segments", [{}]),
        ("decode_passes", [False]),
        ("decode_passes", [{}]),
        ("consensus_score", True),
    ],
)
def test_persistent_alignment_rejects_malformed_evidence(
    field: str, value: object
) -> None:
    raw = runtime._alignment_to_dict(_alignment("hello"))
    raw[field] = value
    with pytest.raises(BuildError, match="alignment result is invalid"):
        runtime._alignment_from_dict(raw)


@pytest.mark.parametrize(
    "changes",
    [
        {"protocol_version": 99},
        {"workspace": "/other"},
        {"token": None},
        {"port": "invalid"},
        {"$schema": "wrong"},
    ],
)
def test_persistent_endpoint_metadata_must_match_its_workspace_and_protocol(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    raw = {
        "$schema": schema_uri("local-runtime-endpoint"),
        "protocol_version": runtime.RUNTIME_PROTOCOL_VERSION,
        "workspace": str(tmp_path),
        "token": "fixture-token",
        "port": 1234,
        **changes,
    }
    path = tmp_path / ".yakbox/runtime/endpoint.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(raw))
    assert runtime._read_endpoint(tmp_path) is None


@pytest.mark.asyncio
async def test_unreachable_runtime_endpoint_reports_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runtime,
        "_read_endpoint",
        lambda _root: {"port": 1234, "token": "fixture-token"},
    )
    monkeypatch.setattr(
        runtime,
        "_send_endpoint",
        AsyncMock(side_effect=BuildError("connection refused")),
    )
    assert not (await runtime.local_runtime_status(tmp_path)).running
    assert not await runtime.stop_local_runtime(tmp_path)


@pytest.mark.parametrize(
    "kind,value",
    [
        ("integer", True),
        ("integer", "1"),
        ("float", True),
        ("float", "1"),
        ("boolean", 1),
    ],
)
def test_runtime_optional_scalar_fields_reject_boolean_integer_confusion(
    kind: str, value: object
) -> None:
    parsers = {
        "integer": runtime._optional_int,
        "float": runtime._optional_float,
        "boolean": runtime._optional_bool,
    }
    with pytest.raises(TypeError):
        parsers[kind](value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame,reason",
    [
        (b"[]\n", "invalid JSON"),
        (
            b'{"protocol_version":99,"ok":true}\n',
            "Unsupported persistent runtime protocol",
        ),
        (b'{"protocol_version":1,"ok":false,"error":"Rejected"}\n', "Rejected"),
        (b"invalid-json\n", "request failed"),
        (b"", "no valid response"),
    ],
)
async def test_runtime_client_rejects_bad_responses_and_always_closes_connection(
    monkeypatch: pytest.MonkeyPatch, frame: bytes, reason: str
) -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(frame)
    reader.feed_eof()
    writer = SimpleNamespace(
        write=Mock(), drain=AsyncMock(), close=Mock(), wait_closed=AsyncMock()
    )
    monkeypatch.setattr(
        runtime.asyncio, "open_connection", AsyncMock(return_value=(reader, writer))
    )
    with pytest.raises(BuildError, match=reason):
        await runtime._send_endpoint(
            {"port": 1234, "token": "fixture-token"},
            {"operation": "status"},
            timeout_seconds=1,
        )
    writer.close.assert_called_once()
    writer.wait_closed.assert_awaited_once()


@pytest.mark.parametrize(
    "error,stale",
    [(ProcessLookupError(), True), (PermissionError(), False), (None, False)],
)
def test_start_lock_pid_probe_distinguishes_exited_and_inaccessible_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception | None,
    stale: bool,
) -> None:
    lock = tmp_path / "starting.lock"
    lock.write_text("123456")
    probe = Mock(side_effect=error)
    monkeypatch.setattr(runtime.os, "kill", probe)
    assert runtime._stale_start_lock(lock) is stale
    probe.assert_called_once_with(123456, 0)
