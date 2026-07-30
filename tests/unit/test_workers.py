from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import pytest

from yakbox.errors import BuildError, ValidationError
from yakbox.speech.models import SpeechSynthesisRequest
from yakbox.speech.workers import (
    IsolatedLocalSpeechService,
    _read_request,
)


class _SilentProcess:
    pid = 1234

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.finished = asyncio.Event()
        self.terminated = False
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        await self.finished.wait()
        return b"", b""

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
    returncode = -9

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", b"worker was terminated"

    async def wait(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


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

    with pytest.raises(BuildError, match="exceeded"):
        await service._communicate(
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

    assert not tuple(tmp_path.glob(".yakbox-worker-*"))
    assert not (tmp_path / "never.wav").exists()
