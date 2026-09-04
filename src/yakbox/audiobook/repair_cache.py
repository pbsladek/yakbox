"""Content-addressed audio cache and exact reuse evidence for repairs."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from yakbox._files import atomic_output_path, atomic_write_json, safe_child, sha256_file
from yakbox.speech.capabilities import BackendCapabilities
from yakbox.speech.fingerprints import speech_request_fingerprint
from yakbox.speech.models import SpeechArtifact, SpeechSynthesisRequest
from yakbox.speech.services import TextToSpeechService

_CACHE_VERSION = 1


@dataclass(frozen=True, slots=True)
class RepairCacheEvent:
    """One stage lookup result retained in the repair-session report."""

    chunk_id: str
    stage: str
    key: str
    hit: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Serialize exact cache provenance."""
        return {
            "chunk_id": self.chunk_id,
            "stage": self.stage,
            "key": self.key,
            "hit": self.hit,
            "reason": self.reason,
        }


class RepairStageCache:
    """Store immutable WAV stages by deterministic input fingerprints."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def key(self, stage: str, inputs: dict[str, object]) -> str:
        """Fingerprint a stage name, implementation version, and exact inputs."""
        payload = json.dumps(
            {"version": _CACHE_VERSION, "stage": stage, "inputs": inputs},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def restore_audio(
        self,
        *,
        chunk_id: str,
        stage: str,
        key: str,
        destination: Path,
    ) -> RepairCacheEvent:
        """Materialize a verified cached WAV, returning an exact miss reason."""
        audio, metadata = self._paths(stage, key)
        if not metadata.is_file():
            return RepairCacheEvent(chunk_id, stage, key, False, "metadata_missing")
        if not audio.is_file():
            return RepairCacheEvent(chunk_id, stage, key, False, "audio_missing")
        try:
            raw = json.loads(metadata.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            return RepairCacheEvent(chunk_id, stage, key, False, "metadata_invalid")
        if (
            not isinstance(raw, dict)
            or raw.get("cache_version") != _CACHE_VERSION
            or raw.get("key") != key
            or raw.get("sha256") != sha256_file(audio)
        ):
            return RepairCacheEvent(chunk_id, stage, key, False, "integrity_mismatch")
        destination = destination.resolve()
        with atomic_output_path(destination, overwrite=True) as temporary:
            shutil.copyfile(audio, temporary)
        return RepairCacheEvent(chunk_id, stage, key, True, "content_match")

    def store_audio(self, *, stage: str, key: str, source: Path) -> None:
        """Commit an immutable stage only after its caller's quality gates pass."""
        audio, metadata = self._paths(stage, key)
        if not audio.is_file():
            with atomic_output_path(audio, overwrite=False) as temporary:
                shutil.copyfile(source, temporary)
        atomic_write_json(
            metadata,
            {
                "cache_version": _CACHE_VERSION,
                "key": key,
                "sha256": sha256_file(audio),
            },
        )

    def _paths(self, stage: str, key: str) -> tuple[Path, Path]:
        if not stage.replace("-", "").isalnum():
            raise ValueError("Repair cache stage is invalid")
        root = safe_child(self.root, self.root / stage / key[:2])
        return root / f"{key}.wav", root / f"{key}.json"


class CachedRepairSpeechService:
    """Content-address every raw generation request used by a repair."""

    def __init__(
        self,
        delegate: TextToSpeechService,
        cache: RepairStageCache,
        *,
        chunk_id: str,
        events: list[RepairCacheEvent],
    ) -> None:
        self.delegate = delegate
        self.cache = cache
        self.chunk_id = chunk_id
        self.events = events

    @property
    def capabilities(self) -> BackendCapabilities:
        """Expose the wrapped backend capabilities unchanged."""
        return self.delegate.capabilities

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        """Restore exact generation bytes or synthesize and store them once."""
        key = self.cache.key(
            "generation",
            {"speech_request": speech_request_fingerprint(request)},
        )
        event = self.cache.restore_audio(
            chunk_id=self.chunk_id,
            stage="generation",
            key=key,
            destination=destination,
        )
        self.events.append(event)
        if event.hit:
            return SpeechArtifact(
                path=destination.resolve(),
                backend=request.backend,
                voice=request.voice,
                output_format=request.output_format,
                bytes_written=destination.stat().st_size,
                sha256=sha256_file(destination),
                sample_rate=request.sample_rate,
            )
        artifact = await self.delegate.synthesize_to_file(
            request,
            destination,
            overwrite=overwrite,
        )
        self.cache.store_audio(stage="generation", key=key, source=destination)
        return artifact


def load_cache_events(report: Path, *, take: int) -> tuple[RepairCacheEvent, ...]:
    """Load cache evidence for one durable repair take."""
    raw = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("takes"), list):
        raise ValueError("Repair report has no takes")
    selected = next(
        (
            item
            for item in cast(list[object], raw["takes"])
            if isinstance(item, dict) and item.get("number") == take
        ),
        None,
    )
    if not isinstance(selected, dict) or not isinstance(selected.get("cache"), list):
        raise ValueError(f"Repair report has no cache evidence for take {take}")
    selected_raw = cast(dict[str, object], selected)
    return tuple(_event(item) for item in cast(list[object], selected_raw["cache"]))


def _event(value: object) -> RepairCacheEvent:
    if not isinstance(value, dict):
        raise ValueError("Repair cache evidence is invalid")
    raw = cast(dict[str, object], value)
    return RepairCacheEvent(
        chunk_id=str(raw["chunk_id"]),
        stage=str(raw["stage"]),
        key=str(raw["key"]),
        hit=bool(raw["hit"]),
        reason=str(raw["reason"]),
    )


__all__ = [
    "CachedRepairSpeechService",
    "RepairCacheEvent",
    "RepairStageCache",
    "load_cache_events",
]
