"""PCM WAV assembly primitives shared by speech entry points."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

from yakbox._files import atomic_output_path
from yakbox.errors import ArtifactError

_BOUNDARY_PAUSE_MS = {
    "paragraph": 250,
    "sentence": 100,
    "clause": 40,
    "word": 0,
    "hard": 0,
    "end": 0,
    "explicit_pause": 0,
}
_JOIN_FADE_MS = 5


@dataclass(frozen=True, slots=True)
class WavJoinPart:
    path: Path
    boundary: str = "end"
    explicit_pause: bool = False


@dataclass(frozen=True, slots=True)
class WavJoinBoundary:
    """Exact start of one part in an assembled WAV timeline."""

    part_index: int
    previous_end_seconds: float
    at_seconds: float
    boundary: str
    inserted_pause_ms: int
    adjacent_to_explicit_pause: bool


def wav_join_boundaries(
    parts: tuple[WavJoinPart, ...],
) -> tuple[WavJoinBoundary, ...]:
    """Calculate every physical part transition using source frame counts."""
    if not parts:
        raise ArtifactError("Synthesis produced no chunks")
    sample_rate: int | None = None
    elapsed_frames = 0
    result: list[WavJoinBoundary] = []
    for index, part in enumerate(parts):
        try:
            with wave.open(str(part.path), "rb") as source:
                current_rate = source.getframerate()
                frame_count = source.getnframes()
        except (OSError, EOFError, wave.Error) as error:
            raise ArtifactError(f"Cannot read PCM WAV: {part.path}") from error
        if sample_rate is None:
            sample_rate = current_rate
        elif current_rate != sample_rate:
            raise ArtifactError("Synthesized chunks have incompatible WAV formats")
        elapsed_frames += frame_count
        if index >= len(parts) - 1:
            continue
        previous_end_seconds = elapsed_frames / current_rate
        pause_ms = _boundary_pause(index, parts)
        elapsed_frames += round(current_rate * pause_ms / 1_000)
        result.append(
            WavJoinBoundary(
                part_index=index + 1,
                previous_end_seconds=previous_end_seconds,
                at_seconds=elapsed_frames / current_rate,
                boundary=part.boundary,
                inserted_pause_ms=pause_ms,
                adjacent_to_explicit_pause=(
                    part.explicit_pause or parts[index + 1].explicit_pause
                ),
            )
        )
    return tuple(result)


def concatenate_wavs(
    parts: tuple[WavJoinPart, ...],
    destination: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Join compatible PCM WAVs with stable semantic pauses and edge fades."""

    if not parts:
        raise ArtifactError("Synthesis produced no chunks")
    params: tuple[int, int, int] | None = None
    with (
        atomic_output_path(destination, overwrite=overwrite) as temporary,
        wave.open(str(temporary), "wb") as writer,
    ):
        for index, part in enumerate(parts):
            params = _append_part(writer, part, index, parts, params)
    if params is None:
        raise ArtifactError("Synthesis produced no readable WAV chunks")


def write_silence(
    path: Path,
    milliseconds: int,
    *,
    sample_rate: int,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    remaining = int(sample_rate * milliseconds / 1_000)
    with (
        atomic_output_path(path, overwrite=True) as temporary,
        wave.open(str(temporary), "wb") as writer,
    ):
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        silence = _silence_bytes(
            min(64 * 1024, remaining),
            channels=channels,
            sample_width=sample_width,
        )
        frame_size = channels * sample_width
        while remaining:
            frames = min(remaining, len(silence) // frame_size)
            writer.writeframesraw(silence[: frames * frame_size])
            remaining -= frames


def _append_part(
    writer: wave.Wave_write,
    part: WavJoinPart,
    index: int,
    parts: tuple[WavJoinPart, ...],
    params: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    with wave.open(str(part.path), "rb") as source:
        current = (
            source.getnchannels(),
            source.getsampwidth(),
            source.getframerate(),
        )
        if params is None:
            writer.setnchannels(current[0])
            writer.setsampwidth(current[1])
            writer.setframerate(current[2])
        elif params != current:
            raise ArtifactError("Synthesized chunks have incompatible WAV formats")
        frames = source.readframes(source.getnframes())
    if not part.explicit_pause:
        frames = _fade_chunk_edges(
            frames,
            channels=current[0],
            sample_width=current[1],
            sample_rate=current[2],
            fade_in=index > 0,
            fade_out=index < len(parts) - 1,
        )
    writer.writeframesraw(frames)
    pause_ms = _boundary_pause(index, parts)
    if pause_ms:
        writer.writeframesraw(
            _silence_bytes(
                int(current[2] * pause_ms / 1_000),
                channels=current[0],
                sample_width=current[1],
            )
        )
    return current


def _boundary_pause(index: int, parts: tuple[WavJoinPart, ...]) -> int:
    if index >= len(parts) - 1:
        return 0
    if parts[index].explicit_pause or parts[index + 1].explicit_pause:
        return 0
    return boundary_pause_milliseconds(parts[index].boundary)


def boundary_pause_milliseconds(boundary: str) -> int:
    """Return Yakbox's inserted silence for one semantic chunk boundary."""

    return _BOUNDARY_PAUSE_MS.get(boundary, 0)


def _fade_chunk_edges(
    frames: bytes,
    *,
    channels: int,
    sample_width: int,
    sample_rate: int,
    fade_in: bool,
    fade_out: bool,
) -> bytes:
    if sample_width not in {1, 2, 3, 4} or not (fade_in or fade_out):
        return frames
    frame_size = channels * sample_width
    total_frames = len(frames) // frame_size
    fade_frames = min(int(sample_rate * _JOIN_FADE_MS / 1_000), total_frames // 2)
    if fade_frames < 1:
        return frames
    output = bytearray(frames)
    if fade_in:
        _scale_pcm_edge(
            output,
            start_frame=0,
            fade_frames=fade_frames,
            channels=channels,
            sample_width=sample_width,
            fade_in=True,
        )
    if fade_out:
        _scale_pcm_edge(
            output,
            start_frame=total_frames - fade_frames,
            fade_frames=fade_frames,
            channels=channels,
            sample_width=sample_width,
            fade_in=False,
        )
    return bytes(output)


def _scale_pcm_edge(
    output: bytearray,
    *,
    start_frame: int,
    fade_frames: int,
    channels: int,
    sample_width: int,
    fade_in: bool,
) -> None:
    frame_size = channels * sample_width
    for frame_offset in range(fade_frames):
        scale = _fade_scale(frame_offset, fade_frames, fade_in)
        frame_start = (start_frame + frame_offset) * frame_size
        for channel in range(channels):
            start = frame_start + channel * sample_width
            _scale_pcm_sample(output, start, sample_width, scale)


def _fade_scale(frame: int, total: int, fade_in: bool) -> float:
    if fade_in:
        return (frame + 1) / total
    return (total - frame - 1) / total


def _scale_pcm_sample(
    output: bytearray, start: int, sample_width: int, scale: float
) -> None:
    stop = start + sample_width
    if sample_width == 1:
        centered = output[start] - 128
        output[start] = max(0, min(255, round(centered * scale) + 128))
        return
    sample = int.from_bytes(output[start:stop], "little", signed=True)
    output[start:stop] = round(sample * scale).to_bytes(
        sample_width, "little", signed=True
    )


def _silence_bytes(frames: int, *, channels: int, sample_width: int) -> bytes:
    sample = b"\x80" if sample_width == 1 else b"\0" * sample_width
    return sample * frames * channels
