"""Prepare licensed source windows for the English qualification corpus."""

from __future__ import annotations

import json
import re
import tomllib
import wave
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_json, safe_child, sha256_file
from yakbox.contracts import runtime_metadata
from yakbox.errors import ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_models import AudioSpan
from yakbox.speech.canonical_audio import CanonicalAudioPreparer

_WINDOW_COUNT = 1
_EDGE_GUARD_MS = 250
_BOUNDARY_SEARCH_MS = 1_000
_BOUNDARY_ENERGY_MS = 80
_MINIMUM_WINDOW_MS = 4_000
_PCM16_SAMPLE_WIDTH = 2
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VOICE_ID = re.compile(r"[a-z0-9][a-z0-9-]{1,63}")
_SOURCE_ID = re.compile(r"[a-z0-9][a-z0-9-]{1,95}")
_VOICE_FIELDS = {
    "file",
    "sha256",
    "reader",
    "reader_url",
    "source_work",
    "catalog_url",
    "source_url",
    "source_sha256",
    "license_id",
    "rights_url",
    "source_start_seconds",
    "duration_seconds",
    "sample_rate_hz",
    "channels",
    "pcm_bits",
    "filters",
}
_INVENTORY_SCHEMA = (
    "https://yakbox.dev/schemas/speech-corpus-source-inventory-v1.schema.json"
)
_INVENTORY_FIELDS = {
    "$schema",
    "schema_version",
    "yakbox_version",
    "timestamp",
    "fingerprint",
    "registry_digest",
    "window_count_per_passage",
    "source_cluster_count",
    "window_count",
    "windows",
}
_WINDOW_FIELDS = {
    "source_window_id",
    "source_passage_group",
    "voice",
    "reader",
    "relative_audio_path",
    "audio_digest",
    "sample_rate",
    "frame_count",
    "canonical_source_digest",
    "start_frame",
    "end_frame",
    "archive_start_milliseconds",
    "archive_end_milliseconds",
    "source_url",
    "source_digest",
    "rights_id",
    "rights_url",
    "fingerprint",
}


@dataclass(frozen=True, slots=True)
class LicensedVoiceSource:
    voice: str
    reader: str
    audio: Path
    audio_digest: str
    source_url: str
    source_digest: str
    rights_id: str
    rights_url: str
    source_start_seconds: float
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class CorpusSourceWindow:
    source_window_id: str
    source_passage_group: str
    voice: str
    reader: str
    relative_audio_path: str
    audio_digest: str
    sample_rate: int
    frame_count: int
    canonical_source_digest: str
    start_frame: int
    end_frame: int
    archive_start_milliseconds: int
    archive_end_milliseconds: int
    source_url: str
    source_digest: str
    rights_id: str
    rights_url: str

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-source-window-v1", self)


@dataclass(frozen=True, slots=True)
class CorpusSourceInventory:
    registry_digest: str
    window_count_per_passage: int
    windows: tuple[CorpusSourceWindow, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.registry_digest, "voice registry digest")
        identifiers = tuple(item.source_window_id for item in self.windows)
        paths = tuple(item.relative_audio_path for item in self.windows)
        digests = tuple(item.audio_digest for item in self.windows)
        group_counts = Counter(item.source_passage_group for item in self.windows)
        if (
            self.window_count_per_passage != _WINDOW_COUNT
            or not self.windows
            or identifiers != tuple(sorted(set(identifiers)))
            or len(paths) != len(set(paths))
            or len(digests) != len(set(digests))
            or set(group_counts.values()) != {_WINDOW_COUNT}
        ):
            raise ValidationError("Corpus source inventory is inconsistent")

    @property
    def source_cluster_count(self) -> int:
        return len({item.source_passage_group for item in self.windows})

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-corpus-source-inventory-v1", self)

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("speech-corpus-source-inventory"),
            "fingerprint": self.fingerprint,
            "registry_digest": self.registry_digest,
            "window_count_per_passage": self.window_count_per_passage,
            "source_cluster_count": self.source_cluster_count,
            "window_count": len(self.windows),
            "windows": [
                {
                    "source_window_id": item.source_window_id,
                    "source_passage_group": item.source_passage_group,
                    "voice": item.voice,
                    "reader": item.reader,
                    "relative_audio_path": item.relative_audio_path,
                    "audio_digest": item.audio_digest,
                    "sample_rate": item.sample_rate,
                    "frame_count": item.frame_count,
                    "canonical_source_digest": item.canonical_source_digest,
                    "start_frame": item.start_frame,
                    "end_frame": item.end_frame,
                    "archive_start_milliseconds": item.archive_start_milliseconds,
                    "archive_end_milliseconds": item.archive_end_milliseconds,
                    "source_url": item.source_url,
                    "source_digest": item.source_digest,
                    "rights_id": item.rights_id,
                    "rights_url": item.rights_url,
                    "fingerprint": item.fingerprint,
                }
                for item in self.windows
            ],
        }


def prepare_corpus_source_inventory(
    voice_registry: Path,
    *,
    repository_root: Path,
    output_root: Path,
) -> CorpusSourceInventory:
    """Canonicalize licensed clips and split each at low-energy boundaries."""
    repository_root = repository_root.resolve()
    output_root = output_root.resolve()
    sources = load_licensed_voice_sources(
        voice_registry,
        repository_root=repository_root,
    )
    preparer = CanonicalAudioPreparer(output_root / "cache")
    windows: list[CorpusSourceWindow] = []
    for source in sources:
        prepared = preparer.prepare(source.audio)
        boundaries = _quiet_boundaries(prepared.path, count=_WINDOW_COUNT)
        for index, (start_frame, end_frame) in enumerate(
            pairwise(boundaries),
            start=1,
        ):
            span = AudioSpan(
                prepared.identity.canonical_digest,
                start_frame,
                end_frame,
                prepared.identity.frame_map.analysis_rate,
            )
            materialized = preparer.materialize_window(prepared, span)
            group = f"{source.voice}-passage-01"
            window_id = f"{group}-window-{index:02d}"
            windows.append(
                CorpusSourceWindow(
                    window_id,
                    group,
                    source.voice,
                    source.reader,
                    materialized.path.relative_to(output_root).as_posix(),
                    materialized.window_span.audio_digest,
                    materialized.window_span.sample_rate,
                    materialized.window_span.end_frame,
                    prepared.identity.canonical_digest,
                    start_frame,
                    end_frame,
                    round(source.source_start_seconds * 1_000),
                    round(
                        (source.source_start_seconds + source.duration_seconds) * 1_000
                    ),
                    source.source_url,
                    source.source_digest,
                    source.rights_id,
                    source.rights_url,
                )
            )
    return CorpusSourceInventory(
        sha256_file(voice_registry),
        _WINDOW_COUNT,
        tuple(sorted(windows, key=lambda item: item.source_window_id)),
    )


def write_corpus_source_inventory(
    path: Path,
    inventory: CorpusSourceInventory,
) -> None:
    atomic_write_json(path, inventory.to_dict())


def load_corpus_source_inventory(
    path: Path,
    *,
    audio_root: Path,
) -> CorpusSourceInventory:
    """Load an inventory and reverify every bound canonical WAV window."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError("Cannot read corpus source inventory") from error
    if not isinstance(raw, dict) or set(raw) != _INVENTORY_FIELDS:
        raise ValidationError("Corpus source inventory fields are invalid")
    if (
        raw.get("$schema") != _INVENTORY_SCHEMA
        or raw.get("schema_version") != 1
        or not isinstance(raw.get("yakbox_version"), str)
        or not isinstance(raw.get("timestamp"), str)
    ):
        raise ValidationError("Corpus source inventory metadata is invalid")
    windows_raw = raw.get("windows")
    if not isinstance(windows_raw, list):
        raise ValidationError("Corpus source inventory windows are invalid")
    windows = tuple(
        _load_inventory_window(item, audio_root=audio_root) for item in windows_raw
    )
    inventory = CorpusSourceInventory(
        _mapping_text(raw, "registry_digest"),
        _mapping_integer(raw, "window_count_per_passage"),
        windows,
    )
    if raw.get("source_cluster_count") != inventory.source_cluster_count or raw.get(
        "window_count"
    ) != len(windows):
        raise ValidationError("Corpus source inventory cluster count differs")
    if raw.get("fingerprint") != inventory.fingerprint:
        raise ValidationError("Corpus source inventory fingerprint differs")
    return inventory


def _load_inventory_window(
    value: object,
    *,
    audio_root: Path,
) -> CorpusSourceWindow:
    if not isinstance(value, dict) or set(value) != _WINDOW_FIELDS:
        raise ValidationError("Corpus source window fields are invalid")
    raw = cast(dict[str, object], value)
    relative_text = _mapping_text(raw, "relative_audio_path")
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValidationError("Corpus source window path must be relative")
    window = CorpusSourceWindow(
        _mapping_text(raw, "source_window_id"),
        _mapping_text(raw, "source_passage_group"),
        _mapping_text(raw, "voice"),
        _mapping_text(raw, "reader"),
        relative.as_posix(),
        _mapping_text(raw, "audio_digest"),
        _mapping_integer(raw, "sample_rate"),
        _mapping_integer(raw, "frame_count"),
        _mapping_text(raw, "canonical_source_digest"),
        _mapping_integer(raw, "start_frame"),
        _mapping_integer(raw, "end_frame"),
        _mapping_integer(raw, "archive_start_milliseconds"),
        _mapping_integer(raw, "archive_end_milliseconds"),
        _mapping_text(raw, "source_url"),
        _mapping_text(raw, "source_digest"),
        _mapping_text(raw, "rights_id"),
        _mapping_text(raw, "rights_url"),
    )
    _validate_inventory_window(window, audio_root=audio_root)
    if raw.get("fingerprint") != window.fingerprint:
        raise ValidationError("Corpus source window fingerprint differs")
    return window


def _validate_inventory_window(
    window: CorpusSourceWindow,
    *,
    audio_root: Path,
) -> None:
    for value, label in (
        (window.audio_digest, "corpus source window digest"),
        (window.canonical_source_digest, "canonical source digest"),
        (window.source_digest, "qualification source digest"),
    ):
        _require_sha256(value, label)
    if (
        _SOURCE_ID.fullmatch(window.source_window_id) is None
        or _SOURCE_ID.fullmatch(window.source_passage_group) is None
        or _VOICE_ID.fullmatch(window.voice) is None
        or not window.reader
        or window.rights_id != "LicenseRef-LibriVox-Public-Domain-US"
        or not window.source_url.startswith("https://")
        or not window.rights_url.startswith("https://")
        or window.start_frame < 0
        or window.end_frame <= window.start_frame
        or window.frame_count != window.end_frame - window.start_frame
        or window.archive_start_milliseconds < 0
        or window.archive_end_milliseconds <= window.archive_start_milliseconds
    ):
        raise ValidationError("Corpus source window metadata is invalid")
    candidate = safe_child(audio_root, audio_root / window.relative_audio_path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValidationError("Corpus source window audio is unavailable")
    if sha256_file(candidate) != window.audio_digest:
        raise ValidationError("Corpus source window audio digest differs")
    try:
        with wave.open(str(candidate), "rb") as reader:
            valid_audio = (
                reader.getnchannels() == 1
                and reader.getsampwidth() == _PCM16_SAMPLE_WIDTH
                and reader.getframerate() == window.sample_rate
                and reader.getnframes() == window.frame_count
            )
    except (EOFError, OSError, wave.Error) as error:
        raise ValidationError("Corpus source window audio is invalid") from error
    if not valid_audio:
        raise ValidationError("Corpus source window audio identity differs")


def load_licensed_voice_sources(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[LicensedVoiceSource, ...]:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValidationError("Cannot read qualification voice registry") from error
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "rights_policy",
        "voices",
    }:
        raise ValidationError("Qualification voice registry fields are invalid")
    if raw.get("schema_version") != 1 or not isinstance(raw.get("rights_policy"), str):
        raise ValidationError("Qualification voice registry metadata is invalid")
    voices = raw.get("voices")
    if not isinstance(voices, dict) or not voices:
        raise ValidationError("Qualification voice registry contains no voices")
    result: list[LicensedVoiceSource] = []
    for voice, value in sorted(voices.items()):
        result.append(
            _load_voice_source(
                voice,
                value,
                registry_root=path.parent,
                repository_root=repository_root,
            )
        )
    return tuple(result)


def _load_voice_source(
    voice: object,
    value: object,
    *,
    registry_root: Path,
    repository_root: Path,
) -> LicensedVoiceSource:
    if not isinstance(voice, str) or _VOICE_ID.fullmatch(voice) is None:
        raise ValidationError("Qualification voice identifier is invalid")
    if not isinstance(value, dict) or set(value) != _VOICE_FIELDS:
        raise ValidationError("Qualification voice fields are invalid")
    entry = cast(dict[str, object], value)
    relative = Path(_text(entry, "file"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValidationError("Qualification voice path must be relative")
    audio = safe_child(repository_root, registry_root / relative)
    if audio.is_symlink() or not audio.is_file():
        raise ValidationError("Qualification voice audio is unavailable")
    digest = _text(entry, "sha256")
    _require_sha256(digest, "qualification voice digest")
    if sha256_file(audio) != digest:
        raise ValidationError("Qualification voice audio digest differs")
    source_digest = _text(entry, "source_sha256")
    _require_sha256(source_digest, "qualification source digest")
    rights_id = _text(entry, "license_id")
    if rights_id != "LicenseRef-LibriVox-Public-Domain-US":
        raise ValidationError("Qualification voice rights basis is unsupported")
    source_start = _number(entry, "source_start_seconds")
    duration = _number(entry, "duration_seconds")
    if source_start < 0 or duration < _MINIMUM_WINDOW_MS / 1_000:
        raise ValidationError("Qualification voice source timing is invalid")
    return LicensedVoiceSource(
        voice,
        _text(entry, "reader"),
        audio,
        digest,
        _https_url(entry, "source_url"),
        source_digest,
        rights_id,
        _https_url(entry, "rights_url"),
        source_start,
        duration,
    )


def _quiet_boundaries(path: Path, *, count: int) -> tuple[int, ...]:
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != _PCM16_SAMPLE_WIDTH:
            raise ValidationError("Corpus source must be canonical mono PCM16")
        rate = reader.getframerate()
        frame_count = reader.getnframes()
        samples = memoryview(reader.readframes(frame_count)).cast("h")
    edge = round(rate * _EDGE_GUARD_MS / 1_000)
    minimum = round(rate * _MINIMUM_WINDOW_MS / 1_000)
    if frame_count - 2 * edge < count * minimum:
        raise ValidationError("Qualification voice is too short for source windows")
    boundaries = [edge]
    usable = frame_count - 2 * edge
    search = round(rate * _BOUNDARY_SEARCH_MS / 1_000)
    for index in range(1, count):
        target = edge + round(usable * index / count)
        lower = max(boundaries[-1] + minimum, target - search)
        upper = min(frame_count - edge - (count - index) * minimum, target + search)
        boundaries.append(_quietest_frame(samples, lower, upper, rate))
    boundaries.append(frame_count - edge)
    return tuple(boundaries)


def _quietest_frame(
    samples: memoryview,
    lower: int,
    upper: int,
    sample_rate: int,
) -> int:
    width = max(1, round(sample_rate * _BOUNDARY_ENERGY_MS / 1_000))
    step = max(1, width // 4)
    candidates = range(lower, upper + 1, step)
    return min(
        candidates,
        key=lambda frame: (
            sum(abs(value) for value in samples[frame : frame + width]),
            abs(frame - (lower + upper) // 2),
            frame,
        ),
    )


def _https_url(raw: dict[str, object], key: str) -> str:
    value = _text(raw, key)
    if not value.startswith("https://"):
        raise ValidationError(f"Qualification voice {key} must use HTTPS")
    return value


def _text(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Qualification voice {key} must be text")
    return value


def _mapping_text(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Corpus source inventory {key} must be text")
    return value


def _mapping_integer(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"Corpus source inventory {key} must be an integer")
    return value


def _number(raw: dict[str, object], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"Qualification voice {key} must be numeric")
    return float(value)


def _require_sha256(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label.capitalize()} must be a SHA-256")


__all__ = [
    "CorpusSourceInventory",
    "CorpusSourceWindow",
    "LicensedVoiceSource",
    "load_corpus_source_inventory",
    "load_licensed_voice_sources",
    "prepare_corpus_source_inventory",
    "write_corpus_source_inventory",
]
