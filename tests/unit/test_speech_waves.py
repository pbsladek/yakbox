from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from yakbox.errors import ArtifactError
from yakbox.speech.waves import (
    WavJoinPart,
    boundary_pause_milliseconds,
    concatenate_wavs,
    wav_join_boundaries,
    write_silence,
)


def _tone(path: Path, *, frames: int = 2_400, sample: int = 10_000) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, 24_000, 0, "NONE", "not compressed"))
        writer.writeframes(struct.pack("<h", sample) * frames)


def test_semantic_join_adds_stable_pause_and_fades_edges(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    output = tmp_path / "joined.wav"
    _tone(first)
    _tone(second)

    concatenate_wavs(
        (
            WavJoinPart(first, "sentence"),
            WavJoinPart(second, "end"),
        ),
        output,
    )

    with wave.open(str(output), "rb") as audio:
        assert audio.getframerate() == 24_000
        assert audio.getnframes() == 2_400 + 2_400 + 2_400
        audio.setpos(2_399)
        assert struct.unpack("<h", audio.readframes(1))[0] == 0
        audio.setpos(4_800)
        assert 0 < struct.unpack("<h", audio.readframes(1))[0] < 10_000


def test_explicit_pause_keeps_native_pcm_format_without_extra_gap(
    tmp_path: Path,
) -> None:
    speech = tmp_path / "speech.wav"
    pause = tmp_path / "pause.wav"
    output = tmp_path / "joined.wav"
    _tone(speech, frames=240)
    write_silence(
        pause,
        100,
        sample_rate=24_000,
        channels=1,
        sample_width=2,
    )

    concatenate_wavs(
        (
            WavJoinPart(speech, "paragraph"),
            WavJoinPart(pause, "explicit_pause", explicit_pause=True),
        ),
        output,
    )

    with wave.open(str(output), "rb") as audio:
        assert audio.getnframes() == 240 + 2_400


def test_join_does_not_overwrite_without_explicit_permission(tmp_path: Path) -> None:
    speech = tmp_path / "speech.wav"
    output = tmp_path / "joined.wav"
    _tone(speech)
    output.write_bytes(b"existing")

    with pytest.raises(ArtifactError, match="already exists"):
        concatenate_wavs((WavJoinPart(speech),), output)

    assert output.read_bytes() == b"existing"


def test_semantic_pause_policy_is_explicit_and_reviewable() -> None:
    assert boundary_pause_milliseconds("sentence") == 100
    assert boundary_pause_milliseconds("paragraph") == 250
    assert boundary_pause_milliseconds("clause") == 40
    assert boundary_pause_milliseconds("word") == 0
    assert boundary_pause_milliseconds("hard") == 0
    assert boundary_pause_milliseconds("unknown") == 0


def test_join_boundaries_match_exact_assembled_part_starts(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    third = tmp_path / "third.wav"
    _tone(first, frames=2_400)
    _tone(second, frames=4_800)
    _tone(third, frames=2_400)
    parts = (
        WavJoinPart(first, "sentence"),
        WavJoinPart(second, "clause"),
        WavJoinPart(third, "end"),
    )

    boundaries = wav_join_boundaries(parts)

    assert boundaries[0].at_seconds == pytest.approx(0.2)
    assert boundaries[0].previous_end_seconds == pytest.approx(0.1)
    assert boundaries[0].inserted_pause_ms == 100
    assert boundaries[1].at_seconds == pytest.approx(0.44)
    assert boundaries[1].previous_end_seconds == pytest.approx(0.4)
    assert boundaries[1].inserted_pause_ms == 40
