from __future__ import annotations

import importlib
import subprocess
import sys
import time
import tracemalloc
import wave
from collections.abc import Callable
from pathlib import Path

import pytest

from yakbox.audio.crop import wav_duration_seconds
from yakbox.audio.inspect import AudioQualityPolicy, inspect_audio
from yakbox.audio.processes import run_command

pytestmark = pytest.mark.performance


def _write_constant_wave(path: Path, seconds: int) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24_000)
        second = (1000).to_bytes(2, "little") * 24_000
        for _ in range(seconds):
            writer.writeframesraw(second)


def test_ten_minute_duration_has_constant_memory_cost(tmp_path: Path) -> None:
    audio = tmp_path / "chapter.wav"
    _write_constant_wave(audio, 600)
    tracemalloc.start()
    started = time.perf_counter()
    try:
        duration = wav_duration_seconds(audio)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert duration == 600
    assert peak < 1024 * 1024
    assert elapsed < 1


def test_large_tool_diagnostics_have_bounded_memory_cost() -> None:
    tracemalloc.start()
    try:
        result = run_command(
            [
                sys.executable,
                "-c",
                'import sys;sys.stderr.write("x"*8000000)',
            ],
            timeout=10,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.returncode == 0
    assert len(result.stderr) <= 64 * 1024
    assert peak < 2 * 1024 * 1024


def test_policy_changes_launch_one_probe_and_one_signal_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "minute.wav"
    _write_constant_wave(audio, 60)
    module = importlib.import_module("yakbox.audio.inspect")
    original = module.run_command
    commands: list[str] = []

    def run(
        command: list[str],
        *,
        timeout: float,
        stderr_line: Callable[[str], None] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command[0])
        return original(command, timeout=timeout, stderr_line=stderr_line)

    monkeypatch.setattr(module, "run_command", run)
    inspect_audio(audio)
    for peak in (-1, -3, -6):
        result = inspect_audio(
            audio,
            quality=AudioQualityPolicy(
                maximum_true_peak_dbfs=peak,
            ),
        )
        assert result.duration_seconds == 60
    assert commands == ["ffprobe", "ffmpeg"]
