"""File-backed synthesis cache inventory and cleanup."""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from yakbox._files import sha256_file
from yakbox.errors import ArtifactError


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """One reusable synthesis cache entry and its validation state."""

    fingerprint: str
    audio_path: Path
    metadata_path: Path
    size: int
    modified_at: datetime
    valid: bool

    def to_dict(self, *, root: Path) -> dict[str, object]:
        """Serialize a cache entry relative to the workspace root."""
        return {
            "fingerprint": self.fingerprint,
            "path": self.audio_path.relative_to(root).as_posix(),
            "size": self.size,
            "modified_at": self.modified_at.isoformat(),
            "valid": self.valid,
        }


@dataclass(frozen=True, slots=True)
class CacheInventory:
    """Snapshot of all synthesis cache entries in a workspace."""

    entries: tuple[CacheEntry, ...]

    @property
    def total_bytes(self) -> int:
        """Return the total bytes represented by all cache entries."""
        return sum(entry.size for entry in self.entries)

    @property
    def invalid_entries(self) -> int:
        """Return the number of entries that failed cache validation."""
        return sum(not entry.valid for entry in self.entries)

    def to_dict(self, *, root: Path) -> dict[str, object]:
        """Serialize the complete cache inventory."""
        return {
            "schema_version": 1,
            "entry_count": len(self.entries),
            "total_bytes": self.total_bytes,
            "invalid_entries": self.invalid_entries,
            "entries": [entry.to_dict(root=root) for entry in self.entries],
        }


@dataclass(frozen=True, slots=True)
class CacheCleanupPlan:
    """Deterministic set of cache entries selected for removal."""

    cache_root: Path
    candidates: tuple[CacheEntry, ...]
    bytes_reclaimed: int

    def to_dict(self, *, workspace: Path) -> dict[str, object]:
        """Serialize a deterministic cache cleanup plan."""
        return {
            "schema_version": 1,
            "cache_root": self.cache_root.relative_to(workspace).as_posix(),
            "candidate_count": len(self.candidates),
            "bytes_reclaimed": self.bytes_reclaimed,
            "candidates": [entry.to_dict(root=workspace) for entry in self.candidates],
        }


def inventory_synthesis_cache(workspace: Path) -> CacheInventory:
    """Return valid and invalid reusable synthesis entries for a workspace."""
    root = _cache_root(workspace)
    if not root.exists():
        return CacheInventory(())
    entries: list[CacheEntry] = []
    for audio in sorted(root.glob("*/*.wav")):
        metadata = audio.with_suffix(".json")
        status = audio.stat()
        fingerprint = audio.stem
        entries.append(
            CacheEntry(
                fingerprint=fingerprint,
                audio_path=audio.resolve(),
                metadata_path=metadata.resolve(),
                size=status.st_size,
                modified_at=datetime.fromtimestamp(status.st_mtime, tz=UTC),
                valid=_valid_entry(audio, metadata, fingerprint),
            )
        )
    return CacheInventory(tuple(entries))


def plan_cache_cleanup(
    workspace: Path,
    *,
    max_age_days: int | None = None,
    max_bytes: int | None = None,
) -> CacheCleanupPlan:
    """Plan deterministic cache cleanup without deleting files."""
    if max_age_days is not None and max_age_days < 0:
        raise ArtifactError("Cache max age must be non-negative")
    if max_bytes is not None and max_bytes < 0:
        raise ArtifactError("Cache byte limit must be non-negative")
    inventory = inventory_synthesis_cache(workspace)
    candidates = {entry for entry in inventory.entries if not entry.valid}
    if max_age_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
        candidates.update(
            entry for entry in inventory.entries if entry.modified_at < cutoff
        )
    if max_bytes is not None:
        retained = sorted(
            (entry for entry in inventory.entries if entry not in candidates),
            key=lambda entry: entry.modified_at,
            reverse=True,
        )
        retained_bytes = sum(entry.size for entry in retained)
        for entry in reversed(retained):
            if retained_bytes <= max_bytes:
                break
            candidates.add(entry)
            retained_bytes -= entry.size
    selected = tuple(sorted(candidates, key=lambda entry: str(entry.audio_path)))
    return CacheCleanupPlan(
        cache_root=_cache_root(workspace),
        candidates=selected,
        bytes_reclaimed=sum(entry.size for entry in selected),
    )


def apply_cache_cleanup(plan: CacheCleanupPlan) -> int:
    """Apply a previously computed cache plan and return removed entry count."""
    root = plan.cache_root.resolve()
    removed = 0
    for entry in plan.candidates:
        audio = entry.audio_path.resolve()
        metadata = entry.metadata_path.resolve()
        if not audio.is_relative_to(root) or not metadata.is_relative_to(root):
            raise ArtifactError("Cache cleanup candidate escapes the cache root")
        if audio.exists():
            audio.unlink()
            removed += 1
        metadata.unlink(missing_ok=True)
    for directory in sorted(root.glob("*"), reverse=True):
        if directory.is_dir():
            with suppress(OSError):
                directory.rmdir()
    return removed


def _cache_root(workspace: Path) -> Path:
    return workspace.resolve() / ".yakbox" / "cache" / "synthesis"


def _valid_entry(audio: Path, metadata: Path, fingerprint: str) -> bool:
    if not metadata.is_file() or audio.stat().st_size == 0:
        return False
    try:
        raw = json.loads(metadata.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return False
    return bool(
        isinstance(raw, dict)
        and raw.get("schema_version") == 1
        and raw.get("fingerprint") == fingerprint
        and raw.get("size") == audio.stat().st_size
        and raw.get("sha256") == sha256_file(audio)
    )
