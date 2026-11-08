"""Compact, endian-independent PCM sample conversion."""

from __future__ import annotations

import sys
from array import array
from collections.abc import Sequence

_PCM_TYPES = {2: "h", 4: "i"}


def decode_pcm(data: bytes, width: int) -> array[int]:
    """Decode little-endian WAV samples into a compact signed buffer."""
    if width in {2, 4}:
        samples = array(_PCM_TYPES[width])
        samples.frombytes(data)
        if sys.byteorder != "little":
            samples.byteswap()
        return samples
    if width == 1:
        return array("i", (value - 128 for value in data))
    return array(
        "i",
        (
            int.from_bytes(data[index : index + width], "little", signed=True)
            for index in range(0, len(data), width)
        ),
    )


def encode_pcm(samples: Sequence[int], width: int) -> bytes:
    """Encode bounded signed samples in the source WAV representation."""
    if width in {2, 4}:
        limit = 1 << (width * 8 - 1)
        try:
            values = array(_PCM_TYPES[width], samples)
        except OverflowError:
            values = array(
                _PCM_TYPES[width],
                (max(-limit, min(limit - 1, value)) for value in samples),
            )
        if sys.byteorder != "little":
            values.byteswap()
        return values.tobytes()
    if width == 1:
        return bytes(max(0, min(255, value + 128)) for value in samples)
    limit = 1 << (width * 8 - 1)
    return b"".join(
        max(-limit, min(limit - 1, value)).to_bytes(width, "little", signed=True)
        for value in samples
    )
