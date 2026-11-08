"""Bounded PCM preparation and timestamp rebasing for local alignment."""

from __future__ import annotations

import math
import tempfile
import wave
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from yakbox._files import atomic_output_path
from yakbox.errors import ArtifactError
from yakbox.speech.alignment import AlignmentResult


@contextmanager
def alignment_audio(
    audio: Path, clip: tuple[float, float] | None
) -> Generator[tuple[Path, float, tuple[float, float] | None]]:
    """Materialize a requested window with one second of bounded context."""
    if clip is None:
        yield audio, 0.0, None
        return
    with tempfile.TemporaryDirectory(prefix="yakbox-align-") as directory:
        destination = Path(directory) / "window.wav"
        offset = _copy_window(audio, destination, clip)
        yield destination, offset, (clip[0] - offset, clip[1] - offset)


def _copy_window(source: Path, destination: Path, clip: tuple[float, float]) -> float:
    try:
        with wave.open(str(source), "rb") as reader:
            rate = reader.getframerate()
            first = max(0, math.floor((clip[0] - 1.0) * rate))
            last = min(reader.getnframes(), math.ceil((clip[1] + 1.0) * rate))
            reader.setpos(first)
            remaining = last - first
            with (
                atomic_output_path(destination) as temporary,
                wave.open(str(temporary), "wb") as writer,
            ):
                writer.setparams(reader.getparams())
                while remaining:
                    count = min(remaining, 65_536)
                    block = reader.readframes(count)
                    if (
                        len(block)
                        != count * reader.getnchannels() * reader.getsampwidth()
                    ):
                        raise ArtifactError("Alignment input has truncated PCM data")
                    writer.writeframesraw(block)
                    remaining -= count
            return first / rate
    except (OSError, EOFError, wave.Error) as error:
        raise ArtifactError("Cannot prepare alignment audio window") from error


def rebase_alignment(
    result: AlignmentResult, offset: float, clip: tuple[float, float] | None
) -> AlignmentResult:
    """Restore absolute chapter timestamps after bounded-window recognition."""
    return replace(
        result,
        tokens=tuple(
            replace(
                item,
                start_seconds=item.start_seconds + offset,
                end_seconds=item.end_seconds + offset,
            )
            for item in result.tokens
        ),
        segments=tuple(
            replace(
                item,
                start_seconds=item.start_seconds + offset,
                end_seconds=item.end_seconds + offset,
            )
            for item in result.segments
        ),
        speech_regions=tuple(
            replace(
                item,
                start_seconds=item.start_seconds + offset,
                end_seconds=item.end_seconds + offset,
            )
            for item in result.speech_regions
        ),
        clip_start_seconds=clip[0] if clip else None,
        clip_end_seconds=clip[1] if clip else None,
    )
