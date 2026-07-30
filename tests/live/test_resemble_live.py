from __future__ import annotations

import asyncio
import os
import wave
from pathlib import Path

import pytest

from yakbox.cloud import (
    ClientOptions,
    HostedUsageGate,
    ResembleClient,
    ResembleSpeechService,
    RetryPolicy,
)
from yakbox.speech import AudioFormat, HostedUsageBudget, SpeechSynthesisRequest

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_one_request_three_character_resemble_canary(tmp_path: Path) -> None:
    if os.environ.get("YAKBOX_RUN_RESEMBLE_LIVE") != "1":
        pytest.skip("set YAKBOX_RUN_RESEMBLE_LIVE=1 to run the hosted canary")
    api_key = _required_environment("RESEMBLE_API_KEY")
    voice_uuid = _required_environment("YAKBOX_CLOUD_VOICE_UUID")
    text = "Hi."
    usage = HostedUsageGate(
        HostedUsageBudget(
            max_provider_requests=1,
            max_submitted_characters=len(text),
        )
    )
    options = ClientOptions(
        connect_timeout=10,
        read_timeout=60,
        write_timeout=10,
        pool_timeout=5,
        max_connections=1,
        max_keepalive_connections=1,
        retry=RetryPolicy(max_attempts=1),
    )
    destination = tmp_path / "resemble-canary.wav"

    async with (
        asyncio.timeout(90),
        ResembleClient(api_key, options=options, usage_gate=usage) as client,
    ):
        artifact = await ResembleSpeechService(client).synthesize_to_file(
            SpeechSynthesisRequest(
                text=text,
                voice=voice_uuid,
                backend="resemble",
                output_format=AudioFormat.WAV,
                precision="PCM_16",
            ),
            destination,
        )

    snapshot = await usage.snapshot()
    assert snapshot.provider_attempts == 1
    assert snapshot.submitted_characters == len(text)
    assert artifact.attempts == 1
    assert artifact.bytes_written == destination.stat().st_size
    with wave.open(str(destination), "rb") as audio:
        assert audio.getnframes() > 0


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is required when the live canary is enabled")
    return value
