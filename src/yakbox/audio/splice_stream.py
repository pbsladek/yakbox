"""Bounded PCM source reads and streaming overlap-add output for repairs."""

from __future__ import annotations

import wave
from array import array
from collections.abc import Callable, Iterable, Sequence

from yakbox.audio.pcm import decode_pcm, encode_pcm
from yakbox.errors import ArtifactError


def read_region(reader: wave.Wave_read, first: int, last: int) -> tuple[int, ...]:
    """Read bounded context from immutable frame coordinates."""
    reader.setpos(first)
    data = reader.readframes(last - first)
    if len(data) != (last - first) * reader.getnchannels() * reader.getsampwidth():
        raise ArtifactError("Repair source has truncated PCM data")
    return tuple(decode_pcm(data, reader.getsampwidth()))


class SpliceStream:
    """Keep only enough output tail to compute the next adaptive crossfade."""

    def __init__(
        self,
        writer: wave.Wave_write,
        *,
        channels: int,
        width: int,
        rate: int,
        overlap: Callable[[tuple[int, ...], tuple[int, ...]], int],
        mix: Callable[[tuple[int, ...], tuple[int, ...], int], tuple[int, ...]],
    ) -> None:
        self.writer = writer
        self.channels = channels
        self.width = width
        self.keep = max(2, round(rate * 0.032)) * channels
        self.overlap = overlap
        self.mix = mix
        self.tail = array("i")
        self.samples_written = 0

    def append_region(
        self,
        reader: wave.Wave_read,
        first: int,
        last: int,
        *,
        overlap_frames: int | None = None,
    ) -> None:
        """Copy one unchanged source region without decoding the entire chapter."""

        def blocks() -> Iterable[Sequence[int]]:
            reader.setpos(first)
            remaining = last - first
            while remaining:
                count = min(remaining, 65_536)
                data = reader.readframes(count)
                if len(data) != count * self.channels * self.width:
                    raise ArtifactError("Repair source has truncated PCM data")
                yield decode_pcm(data, self.width)
                remaining -= count

        self._append(blocks(), overlap_frames=overlap_frames)

    def append_samples(self, samples: tuple[int, ...]) -> None:
        """Append one prepared replacement with the same adaptive overlap rule."""
        self._append((samples,))

    def _append(
        self, blocks: Iterable[Sequence[int]], *, overlap_frames: int | None = None
    ) -> None:
        iterator = iter(blocks)
        first = next(iterator, ())
        # Empty source intervals still participate in the legacy join sequence.
        head = tuple(first[: self.keep])
        count = (
            self.overlap(tuple(self.tail), head)
            if overlap_frames is None
            else overlap_frames
        )
        mixed = self.mix(tuple(self.tail), head, count)
        self.tail = array("i")
        self._write(mixed)
        self._write(first[len(head) :])
        for block in iterator:
            self._write(block)

    def _write(self, samples: Sequence[int]) -> None:
        buffer = self.tail + array("i", samples)
        count = max(0, len(buffer) - self.keep)
        if count:
            self.writer.writeframesraw(encode_pcm(buffer[:count], self.width))
            self.samples_written += count
        self.tail = buffer[count:]

    def finish(self) -> int:
        """Flush the final tail and return the output frame count."""
        self.writer.writeframesraw(encode_pcm(self.tail, self.width))
        self.samples_written += len(self.tail)
        self.tail = array("i")
        return self.samples_written // self.channels
