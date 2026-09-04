from __future__ import annotations

import math
import wave
from pathlib import Path

import pytest

from yakbox._files import sha256_file
from yakbox.errors import ArtifactError
from yakbox.speech.analysis_models import AudioSpan, FrameCoordinateMap
from yakbox.speech.canonical_audio import (
    CANONICAL_SAMPLE_RATE,
    CanonicalAudioPreparer,
    canonical_to_source_frames,
)


def _write_source(path: Path) -> None:
    sample_rate = 44_100
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        frames = bytearray()
        for index in range(sample_rate):
            sample = round(2_000 * math.sin(2 * math.pi * 220 * index / sample_rate))
            encoded = sample.to_bytes(2, "little", signed=True)
            frames.extend(encoded)
            frames.extend(encoded)
        writer.writeframes(bytes(frames))


def _replace_with_silence(path: Path, frame_count: int) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(CANONICAL_SAMPLE_RATE)
        writer.writeframes(b"\0\0" * frame_count)


def test_canonical_audio_and_windows_are_deterministic_and_self_healing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    _write_source(source)
    preparer = CanonicalAudioPreparer(tmp_path / "managed")

    first = preparer.prepare(source)
    second = preparer.prepare(source)

    assert second == first
    assert first.identity.frame_map.source_rate == 44_100
    assert first.identity.frame_map.analysis_rate == CANONICAL_SAMPLE_RATE
    assert first.identity.frame_map.source_delay_frames == 0
    assert first.path.with_name("identity.json").is_file()

    original_digest = sha256_file(first.path)
    first.path.write_bytes(b"corrupt")
    repaired = preparer.prepare(source)
    assert sha256_file(repaired.path) == original_digest
    assert repaired.identity == first.identity

    span = AudioSpan(
        first.identity.canonical_digest,
        1_000,
        4_000,
        CANONICAL_SAMPLE_RATE,
    )
    initial_window = preparer.materialize_window(repaired, span)
    window_digest = sha256_file(initial_window.path)
    _replace_with_silence(initial_window.path, 3_000)
    restored_window = preparer.materialize_window(repaired, span)

    assert restored_window == initial_window
    assert sha256_file(restored_window.path) == window_digest
    assert restored_window.window_span.start_frame == 0
    assert restored_window.window_span.end_frame == 3_000


def test_frame_mapping_rounds_outward_and_applies_decoder_delay() -> None:
    mapping = FrameCoordinateMap(
        source_rate=44_100,
        analysis_rate=16_000,
        source_frame_count=100_000,
        analysis_frame_count=36_281,
        source_delay_frames=529,
    )

    start, end = canonical_to_source_frames(mapping, 101, 999)

    assert start == math.floor(101 * 44_100 / 16_000) + 529
    assert end == math.ceil(999 * 44_100 / 16_000) + 529


def test_long_chapter_frame_mapping_stays_outward_at_both_edges() -> None:
    source_rate = 48_000
    analysis_rate = 16_000
    source_frames = 4 * 60 * 60 * source_rate
    delay = 1_024
    mapping = FrameCoordinateMap(
        source_rate=source_rate,
        analysis_rate=analysis_rate,
        source_frame_count=source_frames,
        analysis_frame_count=math.ceil(
            (source_frames - delay) * analysis_rate / source_rate
        ),
        source_delay_frames=delay,
    )

    start_edge = canonical_to_source_frames(mapping, 0, 1)
    end_start = mapping.analysis_frame_count - 1
    end_edge = canonical_to_source_frames(
        mapping,
        end_start,
        mapping.analysis_frame_count,
    )

    assert start_edge == (delay, delay + 3)
    assert end_edge[0] <= delay + end_start * 3
    assert end_edge[1] == source_frames


def test_canonical_audio_rejects_symlink_sources(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_source(source)
    link = tmp_path / "linked.wav"
    try:
        link.symlink_to(source)
    except OSError as error:
        pytest.fail(f"Test filesystem cannot create a symlink: {error}")

    with pytest.raises(ArtifactError, match="regular file"):
        CanonicalAudioPreparer(tmp_path / "managed").prepare(link)
