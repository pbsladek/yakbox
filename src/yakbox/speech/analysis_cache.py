"""Layered, content-addressed storage for speech-analysis evidence.

Cache entries accelerate analysis but are not release authority.  A promoted
release receives a compact, durable evidence snapshot whose references are
pinned independently of ordinary cache retention.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeVar, cast

from yakbox._files import atomic_write_json, safe_child, sha256_bytes
from yakbox.contracts import runtime_metadata
from yakbox.errors import ArtifactError, ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_models import AlignmentPurpose, AudioSpan

ANALYSIS_CACHE_FORMAT_VERSION = 2
_SHA256 = re.compile(r"[0-9a-f]{64}")
_STAGE = re.compile(r"[a-z][a-z0-9-]{0,63}")
_MAX_DIAGNOSTIC_LENGTH = 160
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class RecognitionCacheIdentity:
    """Independent recognition key; expected text is deliberately impossible."""

    model_fingerprint: str
    execution_fingerprint: str
    canonical_audio_fingerprint: str
    span: AudioSpan
    language: str
    normalization_fingerprint: str = "legacy-normalization"
    calibration_fingerprint: str = "legacy-calibration"
    preprocessing_fingerprint: str = "canonical-pcm-v1"
    decode_settings_fingerprint: str = "default-decode-v1"
    cache_format_version: int = ANALYSIS_CACHE_FORMAT_VERSION

    @property
    def fingerprint(self) -> str:
        # Calibration and expected lexical policy are downstream concerns.  The
        # two legacy fields remain constructor-compatible during the cutover but
        # cannot invalidate independent recognition bytes.
        return semantic_fingerprint(
            "recognition-cache-v2",
            {
                "cache_format_version": self.cache_format_version,
                "normalized_pcm_span_hash": self.canonical_audio_fingerprint,
                "preprocessing_fingerprint": self.preprocessing_fingerprint,
                "language": self.language,
                "recognizer_fingerprint": self.model_fingerprint,
                "execution_identity_fingerprint": self.execution_fingerprint,
                "decode_settings_fingerprint": self.decode_settings_fingerprint,
                "span": self.span,
            },
        )


@dataclass(frozen=True, slots=True)
class ForcedAlignmentCacheIdentity:
    """Timing key tied to authorized text and an exact canonical PCM span."""

    model_fingerprint: str
    execution_fingerprint: str
    canonical_audio_fingerprint: str
    span: AudioSpan
    language: str
    aligner_text_hash: str
    expected_lexical_span_hash: str
    purpose: AlignmentPurpose
    preprocessing_fingerprint: str = "canonical-pcm-v1"
    alignment_settings_fingerprint: str = "default-alignment-v1"
    cache_format_version: int = ANALYSIS_CACHE_FORMAT_VERSION

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("forced-alignment-cache-v2", self)


@dataclass(frozen=True, slots=True)
class ConsensusCacheIdentity:
    """Decision key over immutable recognition and lexical-policy evidence."""

    recognition_fingerprints: tuple[str, ...]
    expected_tokens_hash: str
    policy_fingerprint: str
    equivalence_fingerprint: str
    calibration_fingerprint: str
    normalization_policy_fingerprint: str = "english-lexical-v1"
    cache_format_version: int = ANALYSIS_CACHE_FORMAT_VERSION

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("consensus-cache-v2", self)


@dataclass(frozen=True, slots=True)
class VerificationCacheIdentity:
    """Final decision key over evidence, source mapping, and exact artifact bytes."""

    consensus_fingerprint: str
    forced_alignment_fingerprint: str | None
    signal_evidence_fingerprint: str | None
    artifact_digest: str
    policy_fingerprint: str
    human_disposition_fingerprint: str | None
    spoken_text_plan_fingerprint: str = "unmapped-plan"
    assembly_map_fingerprint: str = "unmapped-assembly"
    artifact_identity_fingerprint: str = "raw-artifact"
    calibration_fingerprint: str = "unqualified-calibration"
    cache_format_version: int = ANALYSIS_CACHE_FORMAT_VERSION

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-verification-cache-v2", self)


class CacheLookupState(StrEnum):
    HIT = "hit"
    MISS = "miss"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class CacheLookup:
    """Bounded cache diagnostic safe for logs and reports."""

    stage: str
    key: str
    state: CacheLookupState
    reason: str

    def __post_init__(self) -> None:
        if not _STAGE.fullmatch(self.stage) or not _SHA256.fullmatch(self.key):
            raise ValidationError(
                "Speech-analysis cache diagnostic identity is invalid"
            )
        if not self.reason or len(self.reason.encode("utf-8")) > _MAX_DIAGNOSTIC_LENGTH:
            raise ValidationError("Speech-analysis cache diagnostic is not bounded")

    @property
    def hit(self) -> bool:
        return self.state is CacheLookupState.HIT

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "key": self.key,
            "state": self.state.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class EvidenceCacheEntry:
    """Validated evidence plus dependency graph metadata."""

    stage: str
    key: str
    evidence: dict[str, object]
    evidence_fingerprint: str
    dependencies: tuple[str, ...]
    created_at_unix_ns: int


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """One durable release reference to a cache evidence node."""

    stage: str
    key: str
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        if not _STAGE.fullmatch(self.stage):
            raise ValidationError("Release evidence stage is invalid")
        for value in (self.key, self.evidence_fingerprint):
            if not _SHA256.fullmatch(value):
                raise ValidationError("Release evidence fingerprint is invalid")


@dataclass(frozen=True, slots=True)
class ReleaseEvidenceSnapshot:
    """Compact durable proof graph retained under the verified release root."""

    release_id: str
    release_audio_digest: str
    text_plan_fingerprint: str
    policy_fingerprint: str
    calibration_fingerprint: str
    execution_fingerprints: tuple[str, ...]
    references: tuple[EvidenceReference, ...]
    decision_fingerprint: str
    human_disposition_fingerprint: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.release_id or "/" in self.release_id or "\\" in self.release_id:
            raise ValidationError("Release evidence ID is invalid")
        fingerprints = (
            self.release_audio_digest,
            self.text_plan_fingerprint,
            self.policy_fingerprint,
            self.calibration_fingerprint,
            self.decision_fingerprint,
            *self.execution_fingerprints,
        )
        if any(not _SHA256.fullmatch(value) for value in fingerprints):
            raise ValidationError("Release evidence snapshot fingerprint is invalid")
        if not self.references:
            raise ValidationError(
                "Release evidence snapshot has no evidence references"
            )
        if tuple(sorted(self.references, key=lambda item: (item.stage, item.key))) != (
            self.references
        ):
            raise ValidationError(
                "Release evidence references must use canonical order"
            )

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("release-evidence-snapshot-v1", self)

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("speech-release-evidence-snapshot"),
            "release_id": self.release_id,
            "release_audio_digest": self.release_audio_digest,
            "text_plan_fingerprint": self.text_plan_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "calibration_fingerprint": self.calibration_fingerprint,
            "execution_fingerprints": list(self.execution_fingerprints),
            "references": [asdict(item) for item in self.references],
            "decision_fingerprint": self.decision_fingerprint,
            "human_disposition_fingerprint": self.human_disposition_fingerprint,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class CacheCleanupResult:
    inspected: int
    removed: int
    preserved_pinned: int
    preserved_recent: int
    corrupt_ignored: int


class LayeredEvidenceCache:
    """Atomic v2 cache with process-local single-flight request coalescing."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._inflight: dict[tuple[str, str], asyncio.Task[EvidenceCacheEntry]] = {}
        self._inflight_lock = asyncio.Lock()

    def lookup(
        self, stage: str, key: str
    ) -> tuple[EvidenceCacheEntry | None, CacheLookup]:
        """Load an integrity-checked entry without trusting its path or payload."""
        path = self._entry_path(stage, key)
        if not path.is_file():
            return None, CacheLookup(stage, key, CacheLookupState.MISS, "entry_missing")
        try:
            raw_value = json.loads(path.read_text(encoding="utf-8"))
            raw = cast(dict[str, object], raw_value)
            entry = self._parse_entry(stage, key, raw)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            KeyError,
            ValueError,
        ):
            self._quarantine(path, stage, key)
            return None, CacheLookup(
                stage, key, CacheLookupState.QUARANTINED, "entry_invalid"
            )
        return entry, CacheLookup(stage, key, CacheLookupState.HIT, "content_match")

    async def get_or_compute(
        self,
        *,
        stage: str,
        key: str,
        dependencies: Sequence[str],
        producer: Callable[[], Awaitable[Mapping[str, object]]],
        validator: Callable[[Mapping[str, object]], object] | None = None,
    ) -> tuple[EvidenceCacheEntry, CacheLookup]:
        """Return evidence once; cancelled waiters never cancel the producer."""
        entry, lookup = self.lookup(stage, key)
        if entry is not None:
            return entry, lookup
        identity = (stage, key)
        async with self._inflight_lock:
            task = self._inflight.get(identity)
            if task is None:
                task = asyncio.create_task(
                    self._produce(
                        stage,
                        key,
                        tuple(sorted(set(dependencies))),
                        producer,
                        validator,
                    )
                )
                self._inflight[identity] = task
                task.add_done_callback(
                    lambda completed, item=identity: self._discard_inflight(
                        item, completed
                    )
                )
        result = await asyncio.shield(task)
        return result, lookup

    def store(
        self,
        *,
        stage: str,
        key: str,
        evidence: Mapping[str, object],
        dependencies: Sequence[str] = (),
        validator: Callable[[Mapping[str, object]], object] | None = None,
    ) -> EvidenceCacheEntry:
        """Atomically store already-computed deterministic evidence."""
        value = dict(evidence)
        if validator is not None:
            validator(value)
        evidence_fingerprint = semantic_fingerprint(
            f"speech-analysis-{stage}-evidence-v1", value
        )
        created_at = time.time_ns()
        ordered_dependencies = tuple(sorted(set(dependencies)))
        entry = EvidenceCacheEntry(
            stage,
            key,
            value,
            evidence_fingerprint,
            ordered_dependencies,
            created_at,
        )
        payload = {
            "cache_format_version": ANALYSIS_CACHE_FORMAT_VERSION,
            "stage": stage,
            "key": key,
            "evidence": value,
            "evidence_fingerprint": evidence_fingerprint,
            "dependencies": list(ordered_dependencies),
            "created_at_unix_ns": created_at,
        }
        payload["integrity_sha256"] = _entry_integrity(payload)
        atomic_write_json(self._entry_path(stage, key), payload, overwrite=True)
        return entry

    def promote_release(
        self,
        release_root: Path,
        snapshot: ReleaseEvidenceSnapshot,
    ) -> Path:
        """Atomically persist release proof, then pin every referenced node."""
        references = tuple(
            sorted(snapshot.references, key=lambda item: (item.stage, item.key))
        )
        if references != snapshot.references:
            raise ValidationError("Release evidence references are not canonical")
        for reference in references:
            entry, _ = self.lookup(reference.stage, reference.key)
            if (
                entry is None
                or entry.evidence_fingerprint != reference.evidence_fingerprint
            ):
                raise ArtifactError(
                    "Release evidence reference is unavailable or stale"
                )
        destination = safe_child(
            release_root,
            release_root
            / "release"
            / "verified"
            / snapshot.release_id
            / "speech-evidence.json",
        )
        atomic_write_json(destination, snapshot.to_dict(), overwrite=True)
        self.pin(
            f"release-{snapshot.release_id}",
            ((item.stage, item.key) for item in references),
            kind="release",
        )
        return destination

    def pin(
        self,
        pin_id: str,
        entries: Iterable[tuple[str, str]],
        *,
        kind: str,
    ) -> Path:
        """Atomically bind active-session, release, or retention cache entries."""
        if not pin_id or not pin_id.replace("-", "").replace("_", "").isalnum():
            raise ValidationError("Speech-analysis cache pin ID is invalid")
        if kind not in {"active_session", "release", "retention"}:
            raise ValidationError("Speech-analysis cache pin kind is invalid")
        values = tuple(sorted(set(entries)))
        for stage, key in values:
            self._entry_path(stage, key)
        path = safe_child(self.root, self.root / "pins" / f"{pin_id}.json")
        atomic_write_json(
            path,
            {
                "cache_format_version": ANALYSIS_CACHE_FORMAT_VERSION,
                "pin_id": pin_id,
                "kind": kind,
                "entries": [{"stage": stage, "key": key} for stage, key in values],
            },
            overwrite=True,
        )
        return path

    def cleanup(self, *, older_than_unix_ns: int) -> CacheCleanupResult:
        """Remove only old, valid, unpinned entries from the managed cache root."""
        pinned = self._pinned_entries()
        inspected = removed = preserved_pinned = preserved_recent = corrupt = 0
        entries_root = safe_child(self.root, self.root / "entries")
        if not entries_root.exists():
            return CacheCleanupResult(0, 0, 0, 0, 0)
        for path in sorted(entries_root.glob("*/*/*.json")):
            inspected += 1
            try:
                relative = path.relative_to(entries_root)
                stage = relative.parts[0]
                key = path.stem
                entry, lookup = self.lookup(stage, key)
            except ArtifactError, ValidationError, ValueError:
                corrupt += 1
                continue
            if entry is None:
                if lookup.state is CacheLookupState.QUARANTINED:
                    corrupt += 1
                continue
            if (stage, key) in pinned:
                preserved_pinned += 1
            elif entry.created_at_unix_ns >= older_than_unix_ns:
                preserved_recent += 1
            else:
                path.unlink(missing_ok=True)
                removed += 1
        return CacheCleanupResult(
            inspected, removed, preserved_pinned, preserved_recent, corrupt
        )

    async def _produce(
        self,
        stage: str,
        key: str,
        dependencies: tuple[str, ...],
        producer: Callable[[], Awaitable[Mapping[str, object]]],
        validator: Callable[[Mapping[str, object]], object] | None,
    ) -> EvidenceCacheEntry:
        evidence = await producer()
        return self.store(
            stage=stage,
            key=key,
            evidence=evidence,
            dependencies=dependencies,
            validator=validator,
        )

    def _parse_entry(
        self, stage: str, key: str, raw: dict[str, object]
    ) -> EvidenceCacheEntry:
        if raw.get("cache_format_version") != ANALYSIS_CACHE_FORMAT_VERSION:
            raise ValueError("wrong cache version")
        if raw.get("stage") != stage or raw.get("key") != key:
            raise ValueError("cache identity mismatch")
        integrity = raw.pop("integrity_sha256", None)
        if integrity != _entry_integrity(raw):
            raise ValueError("cache integrity mismatch")
        evidence = raw.get("evidence")
        dependencies = raw.get("dependencies")
        created_at = raw.get("created_at_unix_ns")
        fingerprint = raw.get("evidence_fingerprint")
        if (
            not isinstance(evidence, dict)
            or not isinstance(dependencies, list)
            or not isinstance(created_at, int)
            or not isinstance(fingerprint, str)
            or not _SHA256.fullmatch(fingerprint)
            or any(not isinstance(item, str) for item in dependencies)
        ):
            raise ValueError("cache schema mismatch")
        typed_evidence = cast(dict[str, object], evidence)
        if fingerprint != semantic_fingerprint(
            f"speech-analysis-{stage}-evidence-v1", typed_evidence
        ):
            raise ValueError("evidence fingerprint mismatch")
        return EvidenceCacheEntry(
            stage,
            key,
            typed_evidence,
            fingerprint,
            tuple(cast(list[str], dependencies)),
            created_at,
        )

    def _entry_path(self, stage: str, key: str) -> Path:
        if not _STAGE.fullmatch(stage) or not _SHA256.fullmatch(key):
            raise ValidationError("Speech-analysis cache key is invalid")
        return safe_child(
            self.root,
            self.root / "entries" / stage / key[:2] / f"{key}.json",
        )

    def _quarantine(self, path: Path, stage: str, key: str) -> None:
        quarantine = safe_child(
            self.root,
            self.root / "quarantine" / stage / key[:2] / f"{key}.json",
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.replace(quarantine)
        except OSError:
            # Ignoring an unreadable entry is safe; quarantine is best effort.
            return

    def _pinned_entries(self) -> set[tuple[str, str]]:
        pins: set[tuple[str, str]] = set()
        root = safe_child(self.root, self.root / "pins")
        if not root.exists():
            return pins
        for path in sorted(root.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    continue
                entries = raw.get("entries")
                if not isinstance(entries, list):
                    continue
                for value in entries:
                    if not isinstance(value, dict):
                        continue
                    stage, key = value.get("stage"), value.get("key")
                    if isinstance(stage, str) and isinstance(key, str):
                        self._entry_path(stage, key)
                        pins.add((stage, key))
            except OSError, UnicodeError, json.JSONDecodeError, ValidationError:
                continue
        return pins

    def _discard_inflight(
        self,
        identity: tuple[str, str],
        task: asyncio.Task[EvidenceCacheEntry],
    ) -> None:
        if self._inflight.get(identity) is task:
            self._inflight.pop(identity, None)


def _entry_integrity(raw: Mapping[str, object]) -> str:
    payload = json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


__all__ = [
    "ANALYSIS_CACHE_FORMAT_VERSION",
    "CacheCleanupResult",
    "CacheLookup",
    "CacheLookupState",
    "ConsensusCacheIdentity",
    "EvidenceCacheEntry",
    "EvidenceReference",
    "ForcedAlignmentCacheIdentity",
    "LayeredEvidenceCache",
    "RecognitionCacheIdentity",
    "ReleaseEvidenceSnapshot",
    "VerificationCacheIdentity",
]
