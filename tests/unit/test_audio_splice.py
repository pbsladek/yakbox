from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from yakbox.audio.crop import inspect_signal_quality, wav_duration_seconds
from yakbox.audio.splice import splice_wav_region


def test_adaptive_splice_matches_level_and_uses_overlaps(tmp_path: Path) -> None:
    original = tmp_path / "original.wav"
    replacement = tmp_path / "replacement.wav"
    output = tmp_path / "spliced.wav"
    _write_sections(original, ((220, 5_000, 0.5), (260, 5_000, 0.5), (220, 5_000, 0.5)))
    _write_sections(replacement, ((440, 14_000, 0.5),))

    evidence = splice_wav_region(
        original,
        replacement,
        output,
        start_seconds=0.5,
        end_seconds=1.0,
    )

    assert evidence.gain_db < 0
    assert evidence.leading_crossfade_ms >= 3
    assert evidence.trailing_crossfade_ms >= 3
    assert wav_duration_seconds(output) == pytest.approx(1.5, abs=0.04)
    signal = inspect_signal_quality(output)
    assert signal.clipped_sample_ratio == 0


def _write_sections(
    path: Path,
    sections: tuple[tuple[int, int, float], ...],
    sample_rate: int = 24_000,
) -> None:
    samples = tuple(
        round(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
        for frequency, amplitude, seconds in sections
        for index in range(round(sample_rate * seconds))
    )
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))
