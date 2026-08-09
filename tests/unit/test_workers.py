from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from yakbox.errors import BuildError, ValidationError
from yakbox.speech.models import SpeechSynthesisRequest
from yakbox.speech.workers import (
    WORKER_PROTOCOL_VERSION,
    IsolatedLocalSpeechService,
    _read_request,
    _remove_stale_part_files,
)


class _SilentProcess:
    pid = 1234

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.finished = asyncio.Event()
        self.terminated = False
        self.killed = False
        self.stdin = _Writer(self)
        self.stdout = _Reader()
        self.stderr = _Reader(eof=True)

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self.finished.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.finished.set()

    async def wait(self) -> int:
        await self.finished.wait()
        return self.returncode or 0


class _TerminatedProcess:
    pid = 5678
    returncode: int | None = None

    def __init__(self) -> None:
        self.stdin = _Writer(self)
        self.stdout = _Reader(eof=True)
        self.stderr = _Reader(b"worker was terminated", eof=True)

    async def wait(self) -> int:
        return self.returncode or 0

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class _Reader:
    def __init__(self, content: bytes = b"", *, eof: bool = False) -> None:
        self.content = content
        self.eof = eof
        self.released = asyncio.Event()

    async def readline(self) -> bytes:
        if self.eof:
            return b""
        await self.released.wait()
        return self.content

    async def read(self, _size: int = -1) -> bytes:
        content, self.content = self.content, b""
        return content


class _SequenceReader:
    def __init__(self, messages: list[dict[str, object]]) -> None:
        self.lines = [f"{json.dumps(message)}\n".encode() for message in messages]

    async def readline(self) -> bytes:
        return self.lines.pop(0) if self.lines else b""


class _Writer:
    def __init__(self, process: _SilentProcess | _TerminatedProcess) -> None:
        self.process = process
        self.payloads: list[bytes] = []
        self.written = asyncio.Event()

    def write(self, payload: bytes) -> None:
        self.payloads.append(payload)
        self.written.set()
        if b'"operation":"shutdown"' in payload:
            self.process.returncode = 0
            if isinstance(self.process, _SilentProcess):
                self.process.finished.set()

    async def drain(self) -> None:
        return


@pytest.mark.asyncio
async def test_silent_worker_emits_heartbeats_and_is_terminated_at_budget(
    tmp_path: Path,
) -> None:
    log = tmp_path / "worker.log"
    service = IsolatedLocalSpeechService(
        timeout_seconds=0.03,
        heartbeat_seconds=0.005,
        log_path=log,
    )
    process = _SilentProcess()
    service._process = cast(asyncio.subprocess.Process, process)

    with pytest.raises(BuildError, match="exceeded"):
        await service._read_response(
            cast(asyncio.subprocess.Process, process),
        )

    assert process.terminated
    content = log.read_text(encoding="utf-8")
    assert "worker_heartbeat" in content
    assert "worker_timed_out" in content


def test_worker_request_validates_resources_and_typed_controls(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="thread budget"):
        IsolatedLocalSpeechService(threads_per_process=0)

    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operation": "synthesize_many",
                "device": "cpu",
                "items": [
                    {
                        "text": "Hi.",
                        "voice": "narrator",
                        "destination": str(tmp_path / "out.wav"),
                        "sample_rate": None,
                        "reference_audio": None,
                        "chatterbox": {
                            "cfg_weight": 0.3,
                            "exaggeration": 0.6,
                            "seed": 7,
                        },
                    }
                ],
                "overwrite": False,
            }
        ),
        encoding="utf-8",
    )

    parsed = _read_request(request)
    assert parsed.device == "cpu"
    item = parsed.items[0]
    assert item.chatterbox is not None
    assert item.chatterbox.cfg_weight == 0.3
    assert item.chatterbox.exaggeration == 0.6
    assert item.chatterbox.seed == 7


@pytest.mark.asyncio
async def test_local_worker_process_is_reused_until_service_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _SilentProcess()
    starts = 0

    async def create_process(*_args: object, **_kwargs: object) -> _SilentProcess:
        nonlocal starts
        starts += 1
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    service = IsolatedLocalSpeechService(timeout_seconds=30)

    first = await service._ensure_worker()
    second = await service._ensure_worker()
    await service.aclose()

    assert first is second
    assert starts == 1
    assert process.returncode == 0
    assert process.stdin.payloads[-1] == b'{"operation":"shutdown"}\n'


@pytest.mark.asyncio
async def test_worker_batch_protocol_streams_hundreds_of_bounded_results() -> None:
    count = 300
    messages: list[dict[str, object]] = [
        {
            "protocol_version": WORKER_PROTOCOL_VERSION,
            "event": "result",
            "index": index,
            "item": {"path": f"chunk-{index}.wav"},
        }
        for index in range(count)
    ]
    messages.append(
        {
            "protocol_version": WORKER_PROTOCOL_VERSION,
            "event": "complete",
            "count": count,
        }
    )
    process = cast(
        asyncio.subprocess.Process,
        SimpleNamespace(pid=1234, stdout=_SequenceReader(messages)),
    )
    service = IsolatedLocalSpeechService(timeout_seconds=1)

    results = await service._read_batch_responses(process, count)

    assert len(results) == count


def test_worker_cleanup_removes_old_and_format_preserving_parts(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "speech.wav"
    old = tmp_path / ".speech.wav.old.part"
    current = tmp_path / ".speech.wav.new.part.wav"
    old.write_bytes(b"partial")
    current.write_bytes(b"partial")

    _remove_stale_part_files(destination)

    assert not old.exists()
    assert not current.exists()


@pytest.mark.asyncio
async def test_cancelled_worker_request_cleans_child_partial_files(
    tmp_path: Path,
) -> None:
    process = _SilentProcess()
    service = IsolatedLocalSpeechService(timeout_seconds=30)
    service._process = cast(asyncio.subprocess.Process, process)
    destination = tmp_path / "speech.wav"
    task = asyncio.create_task(
        service.synthesize_to_file(
            SpeechSynthesisRequest(text="Hi.", voice="narrator"), destination
        )
    )
    await process.stdin.written.wait()
    partial = tmp_path / ".speech.wav.child.part.wav"
    partial.write_bytes(b"partial")

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not partial.exists()


@pytest.mark.asyncio
async def test_terminated_worker_is_reported_and_protocol_files_are_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def create_process(*_args: object, **_kwargs: object) -> _TerminatedProcess:
        return _TerminatedProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    service = IsolatedLocalSpeechService(timeout_seconds=30)

    with pytest.raises(BuildError, match="worker was terminated"):
        await service.synthesize_to_file(
            SpeechSynthesisRequest(text="Hi.", voice="narrator"),
            tmp_path / "never.wav",
        )

    assert not (tmp_path / "never.wav").exists()
