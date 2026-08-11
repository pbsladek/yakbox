"""Durable chunk timelines used for incremental audiobook repair."""

from __future__ import annotations

import hashlib
import json
import wave
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_json, sha256_file
from yakbox.contracts import runtime_metadata, schema_uri
from yakbox.errors import ArtifactError, ValidationError
from yakbox.speech.waves import WavJoinPart, wav_join_boundaries


@dataclass(frozen=True, slots=True)
class AssemblyChunk:
    """One source-addressable audio part in an assembled raw chapter."""

    id: str
    index: int
    kind: str
    text_sha256: str
    characters: int
    speaker: str | None
    profile: str | None
    source_path: str
    source_start_line: int
    source_end_line: int
    boundary: str
    cache_fingerprint: str | None
    audio_sha256: str
    sample_rate: int
    start_frame: int
    end_frame: int
    inserted_pause_after_frames: int

    def to_dict(self) -> dict[str, object]:
        """Serialize one timeline entry without including manuscript text."""
        value = asdict(self)
        value["source"] = {
            "path": value.pop("source_path"),
            "start_line": value.pop("source_start_line"),
            "end_line": value.pop("source_end_line"),
        }
        return value


@dataclass(frozen=True, slots=True)
class AssemblyManifest:
    """Exact ordered chunk provenance for one raw chapter WAV."""

    target: str
    chapter_id: str
    synthesis_fingerprint: str
    raw_audio: Path
    raw_sha256: str
    sample_rate: int
    channels: int
    sample_width: int
    total_frames: int
    chunks: tuple[AssemblyChunk, ...]

    def to_dict(self, *, workspace: Path) -> dict[str, object]:
        """Serialize the timeline as a versioned machine contract."""
        raw = self.raw_audio
        return {
            **runtime_metadata("audiobook-assembly"),
            "target": self.target,
            "chapter_id": self.chapter_id,
            "synthesis_fingerprint": self.synthesis_fingerprint,
            "raw_audio": (
                raw.relative_to(workspace).as_posix()
                if raw.is_relative_to(workspace)
                else raw.as_posix()
            ),
            "raw_sha256": self.raw_sha256,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "total_frames": self.total_frames,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }


def assembly_manifest_path(
    workspace: Path,
    *,
    target: str,
    chapter_id: str,
) -> Path:
    """Return the managed internal timeline path for one chapter."""
    return (
        workspace.resolve() / ".yakbox" / "assemblies" / target / f"{chapter_id}.json"
    )


def create_assembly_manifest(
    *,
    workspace: Path,
    target: str,
    chapter_id: str,
    synthesis_fingerprint: str,
    raw_audio: Path,
    chunk_ids: tuple[str, ...],
    texts: tuple[str, ...],
    speakers: tuple[str | None, ...],
    profiles: tuple[str | None, ...],
    source_paths: tuple[Path, ...],
    source_lines: tuple[tuple[int, int], ...],
    boundaries: tuple[str, ...],
    cache_fingerprints: tuple[str | None, ...],
    parts: tuple[WavJoinPart, ...],
) -> AssemblyManifest:
    """Create an exact timeline from normalized PCM part frame counts."""
    values = (
        chunk_ids,
        texts,
        speakers,
        profiles,
        source_paths,
        source_lines,
        boundaries,
        cache_fingerprints,
        parts,
    )
    if len({len(value) for value in values}) != 1 or not parts:
        raise ArtifactError("Assembly timeline inputs must have equal non-zero lengths")
    joins = wav_join_boundaries(parts)
    starts = (
        0,
        *(round(item.at_seconds * _sample_rate(parts[0].path)) for item in joins),
    )
    pauses = (
        *(
            round(item.inserted_pause_ms * _sample_rate(parts[0].path) / 1_000)
            for item in joins
        ),
        0,
    )
    chunks: list[AssemblyChunk] = []
    for index, values_for_chunk in enumerate(
        zip(
            chunk_ids,
            texts,
            speakers,
            profiles,
            source_paths,
            source_lines,
            boundaries,
            cache_fingerprints,
            parts,
            starts,
            pauses,
            strict=True,
        ),
        start=1,
    ):
        (
            chunk_id,
            text,
            speaker,
            profile,
            source_path,
            lines,
            boundary,
            cache_fingerprint,
            part,
            start_frame,
            pause_frames,
        ) = values_for_chunk
        frame_count, sample_rate = _wave_frames(part.path)
        chunks.append(
            AssemblyChunk(
                id=chunk_id,
                index=index,
                kind="explicit_pause" if part.explicit_pause else "speech",
                text_sha256=_text_sha256(text),
                characters=len(text),
                speaker=speaker,
                profile=profile,
                source_path=_relative_path(source_path, workspace),
                source_start_line=lines[0],
                source_end_line=lines[1],
                boundary=boundary,
                cache_fingerprint=cache_fingerprint,
                audio_sha256=sha256_file(part.path),
                sample_rate=sample_rate,
                start_frame=start_frame,
                end_frame=start_frame + frame_count,
                inserted_pause_after_frames=pause_frames,
            )
        )
    with wave.open(str(raw_audio), "rb") as source:
        return AssemblyManifest(
            target=target,
            chapter_id=chapter_id,
            synthesis_fingerprint=synthesis_fingerprint,
            raw_audio=raw_audio.resolve(),
            raw_sha256=sha256_file(raw_audio),
            sample_rate=source.getframerate(),
            channels=source.getnchannels(),
            sample_width=source.getsampwidth(),
            total_frames=source.getnframes(),
            chunks=tuple(chunks),
        )


def write_assembly_manifest(manifest: AssemblyManifest, *, workspace: Path) -> Path:
    """Atomically persist one current chapter timeline."""
    destination = assembly_manifest_path(
        workspace,
        target=manifest.target,
        chapter_id=manifest.chapter_id,
    )
    atomic_write_json(destination, manifest.to_dict(workspace=workspace))
    return destination


def load_assembly_manifest(path: Path, *, workspace: Path) -> dict[str, object]:
    """Load and minimally validate an assembly timeline for diagnostics."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(
            f"Cannot read assembly manifest {path}: {error}"
        ) from error
    if not isinstance(raw, dict) or raw.get("$schema") != schema_uri(
        "audiobook-assembly"
    ):
        raise ValidationError(f"Unsupported assembly manifest: {path}")
    raw_audio = Path(str(raw.get("raw_audio", "")))
    root = workspace.resolve()
    resolved = (
        raw_audio.resolve() if raw_audio.is_absolute() else (root / raw_audio).resolve()
    )
    if (
        not resolved.is_relative_to(root)
        or not resolved.is_file()
        or raw.get("raw_sha256") != sha256_file(resolved)
    ):
        raise ValidationError(
            f"Assembly manifest raw audio is missing or stale: {path}"
        )
    return raw


def locate_assembly_time(
    workspace: Path,
    *,
    target: str,
    chapter_id: str,
    at_seconds: float,
) -> dict[str, object]:
    """Map a heard timestamp to source, chunk, and adjacent join indices."""
    if at_seconds < 0:
        raise ValidationError("Audio timestamp cannot be negative")
    path = assembly_manifest_path(
        workspace,
        target=target,
        chapter_id=chapter_id,
    )
    raw = load_assembly_manifest(path, workspace=workspace)
    sample_rate = raw.get("sample_rate")
    chunk_values = raw.get("chunks")
    if not isinstance(sample_rate, int) or not isinstance(chunk_values, list):
        raise ValidationError(f"Assembly timeline is invalid: {path}")
    chunks = [
        cast(dict[str, object], chunk)
        for chunk in cast(list[object], chunk_values)
        if isinstance(chunk, dict)
    ]
    frame = round(at_seconds * sample_rate)
    selected = next(
        (
            chunk
            for chunk in chunks
            if _frame_value(chunk, "start_frame")
            <= frame
            < _frame_value(chunk, "end_frame")
        ),
        None,
    )
    if selected is None:
        selected = _nearest_timeline_chunk(chunks, frame)
    if selected is None:
        raise ValidationError(f"Assembly timeline contains no chunks: {path}")
    index = _frame_value(selected, "index") - 1
    return {
        **runtime_metadata("audiobook-repair-location"),
        "target": target,
        "chapter_id": chapter_id,
        "at_seconds": at_seconds,
        "at_frame": frame,
        "chunk": selected,
        "affected_join_indices": [
            value for value in (index - 1, index) if 0 <= value < len(chunks) - 1
        ],
    }


def _wave_frames(path: Path) -> tuple[int, int]:
    try:
        with wave.open(str(path), "rb") as source:
            return source.getnframes(), source.getframerate()
    except (OSError, EOFError, wave.Error) as error:
        raise ArtifactError(f"Cannot read assembly chunk WAV: {path}") from error


def _sample_rate(path: Path) -> int:
    return _wave_frames(path)[1]


def _relative_path(path: Path, workspace: Path) -> str:
    resolved = path.resolve()
    root = workspace.resolve()
    return (
        resolved.relative_to(root).as_posix()
        if resolved.is_relative_to(root)
        else resolved.as_posix()
    )


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _nearest_timeline_chunk(
    chunks: Sequence[dict[str, object]], frame: int
) -> dict[str, object] | None:
    if not chunks:
        return None
    return min(
        chunks,
        key=lambda item: min(
            abs(frame - _frame_value(item, "start_frame")),
            abs(frame - _frame_value(item, "end_frame")),
        ),
    )


def _frame_value(item: dict[str, object], key: str) -> int:
    value = item.get(key)
    if not isinstance(value, int):
        raise ValidationError(f"Assembly timeline field {key!r} is invalid")
    return value
