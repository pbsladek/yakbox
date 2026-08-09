"""Backend-neutral semantic text chunking for speech synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import regex

from yakbox.errors import ValidationError

CHATTERBOX_CHUNK_CHARACTERS = 500


class ChunkBoundary(StrEnum):
    """Semantic boundary that ends one planned synthesis chunk."""

    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    CLAUSE = "clause"
    WORD = "word"
    HARD = "hard"
    END = "end"
    EXPLICIT_PAUSE = "explicit_pause"


@dataclass(frozen=True, slots=True)
class TextChunk:
    """Text plus the semantic boundary used to split it."""

    text: str
    boundary: ChunkBoundary


_CHUNK_BOUNDARIES = (
    (ChunkBoundary.PARAGRAPH, regex.compile(r"\n\s*\n")),
    (ChunkBoundary.SENTENCE, regex.compile(r"[.!?][\"')\]]*\s+")),
    (ChunkBoundary.CLAUSE, regex.compile(r"(?:[,:;]|[\u2013\u2014])\s+")),
    (ChunkBoundary.WORD, regex.compile(r"\s+")),
)


def chunk_text(text: str, maximum: int) -> tuple[str, ...]:
    """Split text at semantic and Unicode grapheme boundaries."""

    return tuple(chunk.text for chunk in plan_text_chunks(text, maximum))


def plan_text_chunks(text: str, maximum: int) -> tuple[TextChunk, ...]:
    """Return deterministic chunks with their selected semantic boundaries."""

    if maximum < 1:
        raise ValidationError("Chunk size must be positive")
    text = text.strip()
    if len(text) <= maximum:
        return (TextChunk(text, ChunkBoundary.END),) if text else ()
    chunks: list[TextChunk] = []
    remainder = text
    while len(remainder) > maximum:
        end, boundary = _semantic_boundary(remainder, maximum)
        if end < 1:
            end = _unicode_safe_boundary(remainder, maximum)
            boundary = ChunkBoundary.HARD
        chunks.append(TextChunk(remainder[:end].strip(), boundary))
        remainder = remainder[end:].strip()
    if remainder:
        chunks.append(TextChunk(remainder, ChunkBoundary.END))
    return tuple(chunk for chunk in chunks if chunk.text)


def _semantic_boundary(text: str, maximum: int) -> tuple[int, ChunkBoundary]:
    window = text[:maximum]
    for boundary, pattern in _CHUNK_BOUNDARIES:
        matches = tuple(pattern.finditer(window))
        if matches:
            return matches[-1].end(), boundary
    return 0, ChunkBoundary.HARD


def _unicode_safe_boundary(text: str, maximum: int) -> int:
    point = 0
    for grapheme in regex.finditer(r"\X", text):
        if grapheme.end() > maximum:
            break
        point = grapheme.end()
    if point < 1:
        raise ValidationError(
            "A single Unicode grapheme exceeds the configured chunk size"
        )
    return point
