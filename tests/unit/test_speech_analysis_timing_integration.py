from __future__ import annotations

import math
import wave
from pathlib import Path

from yakbox.audio.master import master_wav
from yakbox.speech.analysis_release import (
    TimingMarkerObservation,
    qualify_frame_timing_map,
)

_SOURCE_RATE = 48_000
_DESTINATION_RATE = 44_100
_DURATION_SECONDS = 4
_BURST_SECONDS = 0.08
_MARKER_SECONDS = (0.5, 2.0, 3.5)


def test_real_loudnorm_resample_markers_produce_qualified_boundary_map(
    tmp_path: Path,
) -> None:
    source = tmp_path / "markers.wav"
    mastered = tmp_path / "markers-mastered.wav"
    _write_marker_wav(source)

    master_wav(
        source,
        mastered,
        sample_rate=_DESTINATION_RATE,
        normalize=True,
    )

    destination_rate, destination_frames, samples = _read_pcm24(mastered)
    observations = tuple(
        TimingMarkerObservation(
            round(seconds * _SOURCE_RATE),
            _detect_burst_onset(samples, round(seconds * destination_rate)),
        )
        for seconds in _MARKER_SECONDS
    )
    timing = qualify_frame_timing_map(
        source_rate=_SOURCE_RATE,
        destination_rate=destination_rate,
        source_frame_count=_DURATION_SECONDS * _SOURCE_RATE,
        destination_frame_count=destination_frames,
        markers=observations,
        maximum_residual_frames=32,
    )

    assert timing.qualified
    assert timing.monotonic
    assert timing.uncertainty_frames <= 32
    for observation in observations:
        start, end = timing.map_span_outward(
            observation.source_frame,
            observation.source_frame + round(_BURST_SECONDS * _SOURCE_RATE),
        )
        assert start <= observation.destination_frame
        assert end > observation.destination_frame


def _write_marker_wav(path: Path) -> None:
    marker_starts = {round(seconds * _SOURCE_RATE) for seconds in _MARKER_SECONDS}
    burst_frames = round(_BURST_SECONDS * _SOURCE_RATE)
    frames = bytearray()
    for index in range(_DURATION_SECONDS * _SOURCE_RATE):
        offset = next(
            (
                index - marker
                for marker in marker_starts
                if marker <= index < marker + burst_frames
            ),
            None,
        )
        sample = (
            round(12_000 * math.sin(2 * math.pi * 880 * offset / _SOURCE_RATE))
            if offset is not None
            else 0
        )
        frames.extend(sample.to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(_SOURCE_RATE)
        writer.writeframes(frames)


def _read_pcm24(path: Path) -> tuple[int, int, tuple[int, ...]]:
    with wave.open(str(path), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 3
        rate = reader.getframerate()
        frame_count = reader.getnframes()
        payload = reader.readframes(frame_count)
    samples = tuple(
        int.from_bytes(payload[index : index + 3], "little", signed=True)
        for index in range(0, len(payload), 3)
    )
    return rate, frame_count, samples


def _detect_burst_onset(samples: tuple[int, ...], expected: int) -> int:
    start = max(0, expected - 1_000)
    end = min(len(samples), expected + 5_000)
    window = samples[start:end]
    threshold = max(abs(value) for value in window) // 5
    return next(
        index for index in range(start, end) if abs(samples[index]) >= threshold
    )
