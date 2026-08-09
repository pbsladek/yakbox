from __future__ import annotations

import asyncio
import importlib.util
import os
import wave
from pathlib import Path

import pytest

from yakbox.speech.models import AudioFormat, SpeechSynthesisRequest
from yakbox.speech.workers import IsolatedLocalSpeechService

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_short_local_chatterbox_canary(tmp_path: Path) -> None:
    if os.environ.get("YAKBOX_RUN_LOCAL_LIVE") != "1":
        pytest.skip("set YAKBOX_RUN_LOCAL_LIVE=1 to run the local model canary")
    if importlib.util.find_spec("chatterbox") is None:
        pytest.fail('local Chatterbox is not installed; install "yakbox[local]"')

    timeout = _bounded_timeout("YAKBOX_LIVE_LOCAL_TIMEOUT_SECONDS", default=300.0)
    destination = tmp_path / "local-canary.wav"
    service = IsolatedLocalSpeechService(
        device=os.environ.get("YAKBOX_LIVE_LOCAL_DEVICE", "cpu"),
        timeout_seconds=timeout,
        threads_per_process=1,
        heartbeat_seconds=10,
        log_path=tmp_path / "local-canary.log",
    )
    try:
        async with asyncio.timeout(timeout + 15):
            artifact = await service.synthesize_to_file(
                SpeechSynthesisRequest(
                    text="Hi.",
                    voice="canary",
                    backend="chatterbox-local",
                    output_format=AudioFormat.WAV,
                ),
                destination,
            )
    finally:
        await service.aclose()

    assert artifact.backend == "chatterbox-local"
    assert artifact.bytes_written == destination.stat().st_size
    with wave.open(str(destination), "rb") as audio:
        assert audio.getnframes() > 0
        assert audio.getframerate() > 0


def _bounded_timeout(name: str, *, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        pytest.fail(f"{name} must be a number")
    if not 30 <= value <= 600:
        pytest.fail(f"{name} must be between 30 and 600 seconds")
    return value
