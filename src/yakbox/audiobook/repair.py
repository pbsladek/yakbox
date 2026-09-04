"""Targeted, reviewable regeneration of source-addressed audiobook chunks."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

import regex

from yakbox._files import atomic_write_bytes, atomic_write_json, safe_child, sha256_file
from yakbox.audio.crop import (
    crop_aligned_wav,
    inspect_signal_quality,
    inspect_speech_islands,
    wav_duration_seconds,
)
from yakbox.audio.splice import splice_wav_region
from yakbox.audiobook.assembly_manifest import (
    assembly_manifest_path,
    load_assembly_manifest,
)
from yakbox.audiobook.journal import new_run_id
from yakbox.audiobook.manifest import AudiobookManifest, BackendProfile
from yakbox.audiobook.planner import BuildStage, ChunkRoute, PlanNode, plan_audiobook
from yakbox.audiobook.repair_cache import (
    CachedRepairSpeechService,
    RepairCacheEvent,
    RepairStageCache,
)
from yakbox.audiobook.repairs import (
    ApprovedRepair,
    RepairInstall,
    install_approved_repairs,
)
from yakbox.audiobook.sources import normalize_sources
from yakbox.contracts import runtime_metadata, schema_uri
from yakbox.errors import BuildError, ValidationError
from yakbox.local_alignment import open_local_aligner
from yakbox.speech import (
    AudioFormat,
    SpeechSynthesisRequest,
    TextToSpeechService,
    open_speech_backend,
)
from yakbox.speech.alignment import (
    WindowSpeechAligner,
    lexical_tokens,
    validate_carrier_alignment,
)
from yakbox.speech.fingerprints import speech_request_fingerprint
from yakbox.speech.short_synthesis import synthesize_short_utterance
from yakbox.speech.short_utterances import carrier_recipes
from yakbox.whisper_cache import CachedWhisperAligner
from yakbox.whisper_qa import (
    JoinSpecification,
    classify_clip_type,
    evaluate_alignment,
    inspect_joins,
)

_MAXIMUM_REPAIR_TAKES = 20
_MAXIMUM_SINGLE_UTTERANCE_WORDS = 12


class RepairMode(StrEnum):
    """Scope and synthesis-context policy for one repair session."""

    TARGET_ONLY = "target-only"
    CONTEXT = "context"
    SENTENCE = "sentence"
    CLAUSE = "clause"
    NEIGHBORS = "neighbors"
    PARAGRAPH = "paragraph"
    SCENE = "scene"


@dataclass(frozen=True, slots=True)
class RepairChunk:
    """One exact planned chunk selected for replacement."""

    id: str
    index: int
    text: str
    source_path: Path
    source_start_line: int
    source_end_line: int
    route: ChunkRoute
    boundary: str
    previous_context: str | None
    next_context: str | None
    replacement_text: str | None = None
    replacement_start: int = 0
    replacement_end: int = 0

    @property
    def text_sha256(self) -> str:
        """Return the privacy-safe digest bound into approval decisions."""
        return hashlib.sha256(self.text.encode()).hexdigest()

    def to_dict(self, *, workspace: Path) -> dict[str, object]:
        """Serialize selection provenance without copying manuscript text."""
        source = self.source_path.resolve()
        return {
            "id": self.id,
            "index": self.index,
            "text_sha256": self.text_sha256,
            "characters": len(self.text),
            "replacement": {
                "text_sha256": hashlib.sha256(
                    (self.replacement_text or self.text).encode()
                ).hexdigest(),
                "characters": len(self.replacement_text or self.text),
                "start": self.replacement_start,
                "end": self.replacement_end or len(self.text),
            },
            "speaker": self.route.speaker,
            "profile": self.route.profile,
            "boundary": self.boundary,
            "source": {
                "path": (
                    source.relative_to(workspace).as_posix()
                    if source.is_relative_to(workspace)
                    else source.as_posix()
                ),
                "start_line": self.source_start_line,
                "end_line": self.source_end_line,
            },
        }


@dataclass(frozen=True, slots=True)
class RepairPlan:
    """A privacy-safe localized synthesis plan and its affected joins."""

    target: str
    chapter_id: str
    mode: RepairMode
    chunks: tuple[RepairChunk, ...]
    affected_join_indices: tuple[int, ...]

    def to_dict(self, *, workspace: Path) -> dict[str, object]:
        """Serialize the repair scope without including manuscript text."""
        return {
            "schema_version": 1,
            "target": self.target,
            "chapter_id": self.chapter_id,
            "mode": self.mode.value,
            "synthesis_chunks": len(self.chunks),
            "affected_join_indices": list(self.affected_join_indices),
            "chunks": [item.to_dict(workspace=workspace) for item in self.chunks],
        }


@dataclass(frozen=True, slots=True)
class RepairTake:
    """One generated candidate set covering every chunk in a repair plan."""

    number: int
    accepted: bool
    reason_codes: tuple[str, ...]
    audition_path: Path
    chunk_paths: tuple[Path, ...]
    chunk_sha256s: tuple[str, ...]
    cache_events: tuple[RepairCacheEvent, ...] = ()

    def to_dict(self, *, workspace: Path) -> dict[str, object]:
        """Serialize one take using workspace-relative audition paths."""
        return {
            "number": self.number,
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "cache": [event.to_dict() for event in self.cache_events],
            "audition_path": self.audition_path.relative_to(workspace).as_posix(),
            "chunks": [
                {
                    "path": path.relative_to(workspace).as_posix(),
                    "sha256": digest,
                }
                for path, digest in zip(
                    self.chunk_paths, self.chunk_sha256s, strict=True
                )
            ],
        }


@dataclass(frozen=True, slots=True)
class RepairSession:
    """Durable audition session awaiting an explicit take approval."""

    id: str
    plan: RepairPlan
    takes: tuple[RepairTake, ...]
    report_path: Path
    maximum_takes: int = 0
    minimum_passing_takes: int = 0

    def to_dict(self, *, workspace: Path) -> dict[str, object]:
        """Serialize the versioned session for review and later approval."""
        maximum_takes = self.maximum_takes or len(self.takes)
        minimum_passing = self.minimum_passing_takes or min(2, maximum_takes)
        return {
            **runtime_metadata("audiobook-repair-session"),
            "repair_id": self.id,
            "generation": {
                "maximum_takes": maximum_takes,
                "minimum_passing_takes": minimum_passing,
                "generated_takes": len(self.takes),
                "passing_takes": sum(take.accepted for take in self.takes),
                "stopped_early": len(self.takes) < maximum_takes,
            },
            "plan": self.plan.to_dict(workspace=workspace),
            "takes": [take.to_dict(workspace=workspace) for take in self.takes],
        }


@dataclass(frozen=True, slots=True)
class RepairBatchGeneration:
    """Several repair sessions rendered under one warm runtime lifetime."""

    batch: RepairBatch
    sessions: tuple[RepairSession, ...]
    review_playlist: Path
    report_path: Path

    def to_dict(self, *, workspace: Path) -> dict[str, object]:
        """Serialize the batch review package without manuscript text."""
        return {
            **runtime_metadata("audiobook-repair-batch-generation"),
            "batch_id": self.batch.id,
            "review_playlist": self.review_playlist.relative_to(workspace).as_posix(),
            "sessions": [
                {
                    "repair_id": session.id,
                    "profiles": list(_plan_profiles(session.plan)),
                    "report_path": session.report_path.relative_to(
                        workspace
                    ).as_posix(),
                    "passing_takes": sum(take.accepted for take in session.takes),
                    "takes": [
                        {
                            "number": take.number,
                            "accepted": take.accepted,
                            "audition_path": take.audition_path.relative_to(
                                workspace
                            ).as_posix(),
                        }
                        for take in session.takes
                    ],
                }
                for session in self.sessions
            ],
        }


@dataclass(frozen=True, slots=True)
class RepairApproval:
    """Approved replacements and the chapter that must be rebuilt."""

    target: str
    chapter_id: str
    repairs: tuple[ApprovedRepair, ...]


@dataclass(frozen=True, slots=True)
class RepairBatchEntry:
    """One staged, source-validated repair-session approval."""

    repair_id: str
    take: int
    target: str
    chapter_id: str
    installs: tuple[RepairInstall, ...]


@dataclass(frozen=True, slots=True)
class RepairBatch:
    """Durable set of approvals waiting for one atomic finalize operation."""

    id: str
    entries: tuple[RepairBatchEntry, ...]
    report_path: Path

    def to_dict(self, *, workspace: Path) -> dict[str, object]:
        """Serialize the batch without exposing manuscript text."""
        return {
            **runtime_metadata("audiobook-repair-batch"),
            "batch_id": self.id,
            "entries": [
                {
                    "repair_id": entry.repair_id,
                    "take": entry.take,
                    "target": entry.target,
                    "chapter_id": entry.chapter_id,
                    "chunks": [
                        {
                            "id": install.chunk_id,
                            "text_sha256": install.text_sha256,
                            "profile": install.profile,
                            "candidate_path": install.candidate_audio.relative_to(
                                workspace
                            ).as_posix(),
                            "candidate_sha256": sha256_file(install.candidate_audio),
                            "source": {
                                "path": install.source_path,
                                "start_line": install.source_start_line,
                                "end_line": install.source_end_line,
                            },
                        }
                        for install in entry.installs
                    ],
                }
                for entry in self.entries
            ],
        }


def plan_repair(
    manifest: AudiobookManifest,
    *,
    target_name: str = "default",
    chapter_selector: str | None = None,
    chunk_id: str | None = None,
    source_line: int | None = None,
    text_match: str | None = None,
    speaker: str | None = None,
    mode: RepairMode | str = RepairMode.CONTEXT,
) -> RepairPlan:
    """Resolve one unique source target and expand it to the requested scope."""
    if sum(value is not None for value in (chunk_id, source_line, text_match)) != 1:
        raise ValidationError(
            "Repair requires exactly one of chunk_id, source_line, or text_match"
        )
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
        strip_attribution_tags=manifest.dialogue.strip_attribution_tags,
        dialogue_routes=manifest.dialogue.routes,
        expressive_tag_handling=manifest.dialogue.expressive_tag_handling,
        retain_first_attribution_per_scene=(
            manifest.dialogue.retain_first_attribution_per_scene
        ),
    )
    plan = plan_audiobook(
        manifest,
        document,
        target_name=target_name,
        chapter_selector=chapter_selector,
    )
    synthesis_nodes = tuple(
        node for node in plan.nodes if node.stage is BuildStage.SYNTHESIZE
    )
    matches = _matching_chunks(
        synthesis_nodes,
        chunk_id=chunk_id,
        source_line=source_line,
        text_match=text_match,
        speaker=speaker,
    )
    if not matches:
        raise ValidationError("No repairable speech chunk matched the selector")
    if len(matches) != 1:
        ids = ", ".join(item[0].chunk_ids[item[1]] for item in matches[:8])
        raise ValidationError(
            f"Repair selector matched {len(matches)} chunks; use --chunk-id: {ids}"
        )
    node, target_index = matches[0]
    resolved_mode = mode if isinstance(mode, RepairMode) else RepairMode(mode)
    selected = _selected_indices(node, target_index, resolved_mode)
    chunks = tuple(
        _repair_chunk(
            node,
            index,
            mode=resolved_mode,
            text_match=text_match if index == target_index else None,
        )
        for index in selected
    )
    joins = tuple(
        sorted(
            {
                join
                for index in selected
                for join in (index - 1, index)
                if 0 <= join < len(node.chunks) - 1
            }
        )
    )
    return RepairPlan(
        target=target_name,
        chapter_id=node.chapter_id,
        mode=resolved_mode,
        chunks=chunks,
        affected_join_indices=joins,
    )


async def generate_repair_session(
    manifest: AudiobookManifest,
    plan: RepairPlan,
    *,
    takes: int = 4,
    minimum_passing_takes: int | None = None,
    whisper_qa: bool = True,
    api_key: str | None = None,
) -> RepairSession:
    """Generate reviewable takes while keeping each backend worker warm."""
    if not 1 <= takes <= _MAXIMUM_REPAIR_TAKES:
        raise ValidationError("Repair takes must be between 1 and 20")
    required_passing = (
        min(2, takes) if minimum_passing_takes is None else minimum_passing_takes
    )
    if not 1 <= required_passing <= takes:
        raise ValidationError(
            "Minimum passing repair takes must be between 1 and maximum takes"
        )
    repair_id = new_run_id()
    root = manifest.root / ".yakbox" / "repair-sessions" / repair_id
    aligner = (
        _repair_aligner(manifest)
        if whisper_qa
        or plan.mode in {RepairMode.CONTEXT, RepairMode.SENTENCE, RepairMode.CLAUSE}
        else None
    )
    if aligner is not None and manifest.whisper_qa.cache_enabled:
        aligner = CachedWhisperAligner(aligner, manifest.whisper_qa.cache_directory)
    services: dict[tuple[str, str | None], TextToSpeechService] = {}
    async with AsyncExitStack() as stack:
        session = await _generate_session_with_services(
            manifest,
            plan,
            repair_id=repair_id,
            root=root,
            takes=takes,
            required_passing=required_passing,
            aligner=aligner,
            services=services,
            stack=stack,
            api_key=api_key,
        )
    atomic_write_json(session.report_path, session.to_dict(workspace=manifest.root))
    return session


async def generate_repair_batch(
    manifest: AudiobookManifest,
    plans: tuple[RepairPlan, ...],
    *,
    takes: int = 4,
    minimum_passing_takes: int | None = None,
    whisper_qa: bool = True,
    api_key: str | None = None,
) -> RepairBatchGeneration:
    """Generate several repair sessions with shared models and one review package."""
    if not plans:
        raise ValidationError("Repair batch generation needs at least one plan")
    scopes = {(plan.target, plan.chapter_id) for plan in plans}
    if len(scopes) != 1:
        raise ValidationError("One repair batch must target exactly one chapter")
    if not 1 <= takes <= _MAXIMUM_REPAIR_TAKES:
        raise ValidationError("Repair takes must be between 1 and 20")
    required_passing = (
        min(2, takes) if minimum_passing_takes is None else minimum_passing_takes
    )
    if not 1 <= required_passing <= takes:
        raise ValidationError(
            "Minimum passing repair takes must be between 1 and maximum takes"
        )
    batch = begin_repair_batch(manifest)
    root = batch.report_path.parent / "generation"
    needs_alignment = whisper_qa or any(
        plan.mode in {RepairMode.CONTEXT, RepairMode.SENTENCE, RepairMode.CLAUSE}
        for plan in plans
    )
    aligner = _repair_aligner(manifest) if needs_alignment else None
    if aligner is not None and manifest.whisper_qa.cache_enabled:
        aligner = CachedWhisperAligner(aligner, manifest.whisper_qa.cache_directory)
    services: dict[tuple[str, str | None], TextToSpeechService] = {}
    indexed = tuple(enumerate(plans))
    ordered = sorted(
        indexed,
        key=lambda value: (_plan_profiles(value[1]), value[0]),
    )
    generated: dict[int, RepairSession] = {}
    async with AsyncExitStack() as stack:
        for index, plan in ordered:
            repair_id = new_run_id()
            session_root = manifest.root / ".yakbox" / "repair-sessions" / repair_id
            session = await _generate_session_with_services(
                manifest,
                plan,
                repair_id=repair_id,
                root=session_root,
                takes=takes,
                required_passing=required_passing,
                aligner=aligner,
                services=services,
                stack=stack,
                api_key=api_key,
            )
            atomic_write_json(
                session.report_path,
                session.to_dict(workspace=manifest.root),
            )
            generated[index] = session
    sessions = tuple(generated[index] for index in range(len(plans)))
    playlist = _write_batch_review_playlist(root, sessions)
    result = RepairBatchGeneration(
        batch=batch,
        sessions=sessions,
        review_playlist=playlist,
        report_path=root / "report.json",
    )
    atomic_write_json(result.report_path, result.to_dict(workspace=manifest.root))
    return result


async def _generate_session_with_services(
    manifest: AudiobookManifest,
    plan: RepairPlan,
    *,
    repair_id: str,
    root: Path,
    takes: int,
    required_passing: int,
    aligner: WindowSpeechAligner | None,
    services: dict[tuple[str, str | None], TextToSpeechService],
    stack: AsyncExitStack,
    api_key: str | None,
) -> RepairSession:
    generated: list[RepairTake] = []
    passing = 0
    for number in range(1, takes + 1):
        take = await _generate_take(
            manifest,
            plan,
            number=number,
            root=root,
            aligner=aligner,
            services=services,
            stack=stack,
            api_key=api_key,
        )
        generated.append(take)
        passing += take.accepted
        if passing >= required_passing:
            break
    return RepairSession(
        id=repair_id,
        plan=plan,
        takes=tuple(generated),
        report_path=root / "report.json",
        maximum_takes=takes,
        minimum_passing_takes=required_passing,
    )


def approve_repair_session(
    manifest: AudiobookManifest,
    *,
    repair_id: str,
    take: int,
) -> RepairApproval:
    """Approve one complete take after revalidating source and candidate bytes."""
    entry = _validated_repair_entry(manifest, repair_id=repair_id, take=take)
    repairs = install_approved_repairs(
        manifest.root,
        entry.target,
        installs=entry.installs,
    )
    return RepairApproval(
        target=entry.target,
        chapter_id=entry.chapter_id,
        repairs=repairs,
    )


def begin_repair_batch(manifest: AudiobookManifest) -> RepairBatch:
    """Create an empty durable approval batch without changing build inputs."""
    batch_id = new_run_id()
    report = manifest.root / ".yakbox" / "repair-batches" / batch_id / "report.json"
    batch = RepairBatch(id=batch_id, entries=(), report_path=report)
    atomic_write_json(report, batch.to_dict(workspace=manifest.root))
    return batch


def stage_repair_batch_entry(
    manifest: AudiobookManifest,
    *,
    batch_id: str,
    repair_id: str,
    take: int,
) -> RepairBatch:
    """Validate and stage one take; no approved-repair state is mutated."""
    batch = load_repair_batch(manifest, batch_id=batch_id)
    entry = _validated_repair_entry(manifest, repair_id=repair_id, take=take)
    if any(item.repair_id == repair_id for item in batch.entries):
        raise ValidationError("Repair session is already staged in this batch")
    if batch.entries and any(
        (item.target, item.chapter_id) != (entry.target, entry.chapter_id)
        for item in batch.entries
    ):
        raise ValidationError("One repair batch must target exactly one chapter")
    existing_ids = {
        install.chunk_id for item in batch.entries for install in item.installs
    }
    if existing_ids.intersection(install.chunk_id for install in entry.installs):
        raise ValidationError("Repair batch contains overlapping chunk approvals")
    updated = replace(batch, entries=(*batch.entries, entry))
    atomic_write_json(updated.report_path, updated.to_dict(workspace=manifest.root))
    return updated


def finalize_repair_batch(
    manifest: AudiobookManifest,
    *,
    batch_id: str,
) -> RepairApproval:
    """Revalidate and atomically commit a batch for one later chapter rebuild."""
    batch = load_repair_batch(manifest, batch_id=batch_id)
    if not batch.entries:
        raise ValidationError("Cannot finalize an empty repair batch")
    refreshed = tuple(
        _validated_repair_entry(
            manifest,
            repair_id=entry.repair_id,
            take=entry.take,
        )
        for entry in batch.entries
    )
    first = refreshed[0]
    if any(
        (entry.target, entry.chapter_id) != (first.target, first.chapter_id)
        for entry in refreshed
    ):
        raise ValidationError("Repair batch scope changed after staging")
    installs = tuple(install for entry in refreshed for install in entry.installs)
    repairs = install_approved_repairs(
        manifest.root,
        first.target,
        installs=installs,
    )
    return RepairApproval(
        target=first.target,
        chapter_id=first.chapter_id,
        repairs=repairs,
    )


def load_repair_batch(
    manifest: AudiobookManifest,
    *,
    batch_id: str,
) -> RepairBatch:
    """Load and revalidate a durable repair batch from managed storage."""
    root = safe_child(
        manifest.root / ".yakbox" / "repair-batches",
        manifest.root / ".yakbox" / "repair-batches" / batch_id,
    )
    report = root / "report.json"
    try:
        raw = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"Cannot read repair batch {report}: {error}") from error
    if not isinstance(raw, dict) or raw.get("$schema") != schema_uri(
        "audiobook-repair-batch"
    ):
        raise ValidationError(f"Unsupported repair batch: {report}")
    values = raw.get("entries")
    if not isinstance(values, list):
        raise ValidationError("Repair batch entries are invalid")
    entries: list[RepairBatchEntry] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValidationError("Repair batch entry is invalid")
        entries.append(
            _validated_repair_entry(
                manifest,
                repair_id=str(value.get("repair_id")),
                take=int(cast(int, value.get("take"))),
            )
        )
    return RepairBatch(id=batch_id, entries=tuple(entries), report_path=report)


def _validated_repair_entry(
    manifest: AudiobookManifest,
    *,
    repair_id: str,
    take: int,
) -> RepairBatchEntry:
    root = safe_child(
        manifest.root / ".yakbox" / "repair-sessions",
        manifest.root / ".yakbox" / "repair-sessions" / repair_id,
    )
    raw = _load_session(root / "report.json")
    target = _required_string(raw, "target", parent="plan")
    chapter = _required_string(raw, "chapter_id", parent="plan")
    plan_raw = raw["plan"]
    if not isinstance(plan_raw, dict):
        raise ValidationError("Repair session plan is invalid")
    mode = RepairMode(str(plan_raw.get("mode")))
    chunk_values = plan_raw.get("chunks")
    if not isinstance(chunk_values, list) or not chunk_values:
        raise ValidationError("Repair session chunk list is invalid")
    session_chunks = cast(list[object], chunk_values)
    expected = _current_repair_chunks(
        manifest,
        target=target,
        chapter=chapter,
        mode=mode,
        session_chunks=session_chunks,
    )
    take_raw = _session_take(raw, take)
    if not bool(take_raw.get("accepted")):
        raise ValidationError("Cannot approve a repair take that failed QA")
    audio_values = take_raw.get("chunks")
    if not isinstance(audio_values, list) or len(audio_values) != len(expected):
        raise ValidationError("Repair take audio list is incomplete")
    installs: list[RepairInstall] = []
    for chunk_raw, audio_raw in zip(session_chunks, audio_values, strict=True):
        chunk_id = str(_required_mapping(chunk_raw, "id"))
        chunk = expected.get(chunk_id)
        if chunk is None or chunk.text_sha256 != _required_mapping(
            chunk_raw, "text_sha256"
        ):
            raise ValidationError("Repair session source changed after generation")
        if not isinstance(audio_raw, dict):
            raise ValidationError("Repair take audio entry is invalid")
        audio = safe_child(manifest.root, manifest.root / str(audio_raw.get("path")))
        if sha256_file(audio) != audio_raw.get("sha256"):
            raise ValidationError("Repair take audio digest is invalid")
        if chunk.route.profile is None:
            raise ValidationError(
                "Explicit pauses cannot be approved as speech repairs"
            )
        installs.append(
            RepairInstall(
                chunk_id=chunk.id,
                text_sha256=chunk.text_sha256,
                profile=chunk.route.profile,
                source_path=_workspace_path(chunk.source_path, manifest.root),
                source_start_line=chunk.source_start_line,
                source_end_line=chunk.source_end_line,
                repair_id=repair_id,
                take=take,
                candidate_audio=audio,
            )
        )
    return RepairBatchEntry(
        repair_id=repair_id,
        take=take,
        target=target,
        chapter_id=chapter,
        installs=tuple(installs),
    )


def _current_repair_chunks(
    manifest: AudiobookManifest,
    *,
    target: str,
    chapter: str,
    mode: RepairMode,
    session_chunks: list[object],
) -> dict[str, RepairChunk]:
    first_id = str(_required_mapping(session_chunks[0], "id"))
    current = plan_repair(
        manifest,
        target_name=target,
        chapter_selector=chapter,
        chunk_id=first_id,
        mode=(
            RepairMode.TARGET_ONLY
            if mode in {RepairMode.SENTENCE, RepairMode.CLAUSE}
            else mode
        ),
    )
    expected = {chunk.id: chunk for chunk in current.chunks}
    session_ids = {str(_required_mapping(chunk, "id")) for chunk in session_chunks}
    if session_ids != set(expected) or len(session_ids) != len(session_chunks):
        raise ValidationError("Repair session scope changed after generation")
    return expected


def _matching_chunks(
    nodes: tuple[PlanNode, ...],
    *,
    chunk_id: str | None,
    source_line: int | None,
    text_match: str | None,
    speaker: str | None,
) -> tuple[tuple[PlanNode, int], ...]:
    matches: list[tuple[PlanNode, int]] = []
    query = text_match.casefold() if text_match is not None else None
    for node in nodes:
        for index, (candidate_id, text, source, route) in enumerate(
            zip(
                node.chunk_ids,
                node.chunks,
                node.chunk_sources,
                node.chunk_routes,
                strict=True,
            )
        ):
            if route.profile is None:
                continue
            selected = (
                (chunk_id is not None and candidate_id == chunk_id)
                or (
                    source_line is not None
                    and source.start_line <= source_line <= source.end_line
                )
                or (query is not None and query in text.casefold())
            )
            if selected and (speaker is None or route.speaker == speaker):
                matches.append((node, index))
    return tuple(matches)


def _selected_indices(node: PlanNode, target: int, mode: RepairMode) -> tuple[int, ...]:
    if mode in {
        RepairMode.TARGET_ONLY,
        RepairMode.CONTEXT,
        RepairMode.SENTENCE,
        RepairMode.CLAUSE,
    }:
        return (target,)
    if mode is RepairMode.NEIGHBORS:
        return tuple(
            index
            for index in range(max(0, target - 1), min(len(node.chunks), target + 2))
            if node.chunk_routes[index].profile is not None
        )
    source = node.chunk_sources[target]
    if mode is RepairMode.PARAGRAPH:
        return tuple(
            index
            for index, candidate in enumerate(node.chunk_sources)
            if candidate == source and node.chunk_routes[index].profile is not None
        )
    start = target
    while start > 0 and node.chunk_routes[start - 1].profile is not None:
        start -= 1
    end = target
    while end + 1 < len(node.chunks) and node.chunk_routes[end + 1].profile is not None:
        end += 1
    return tuple(range(start, end + 1))


def _repair_chunk(
    node: PlanNode,
    index: int,
    *,
    mode: RepairMode = RepairMode.TARGET_ONLY,
    text_match: str | None = None,
) -> RepairChunk:
    source = node.chunk_sources[index]
    route = node.chunk_routes[index]
    start, end = _repair_text_span(node.chunks[index], text_match, mode)
    return RepairChunk(
        id=node.chunk_ids[index],
        index=index,
        text=node.chunks[index],
        source_path=source.path,
        source_start_line=source.start_line,
        source_end_line=source.end_line,
        route=route,
        boundary=node.chunk_boundaries[index],
        previous_context=_same_voice_context(node, index, -1),
        next_context=_same_voice_context(node, index, 1),
        replacement_text=node.chunks[index][start:end],
        replacement_start=start,
        replacement_end=end,
    )


def _repair_text_span(
    text: str,
    query: str | None,
    mode: RepairMode,
) -> tuple[int, int]:
    if mode not in {RepairMode.SENTENCE, RepairMode.CLAUSE}:
        return 0, len(text)
    spans = _linguistic_spans(text, mode)
    if len(spans) == 1:
        return spans[0]
    if query is None:
        raise ValidationError(
            f"{mode.value} repair needs --text when a chunk contains multiple spans"
        )
    query_start = text.casefold().find(query.casefold())
    if query_start < 0:
        raise ValidationError(
            "Repair text selector is not present in the matched chunk"
        )
    query_end = query_start + len(query)
    matches = tuple(
        span for span in spans if span[0] <= query_start and query_end <= span[1]
    )
    if len(matches) != 1:
        raise ValidationError(
            f"Repair text selector does not identify one {mode.value} span"
        )
    return matches[0]


def _linguistic_spans(text: str, mode: RepairMode) -> tuple[tuple[int, int], ...]:
    boundary = (
        r"[.!?][\"')\]]*(?:\s+|$)"
        if mode is RepairMode.SENTENCE
        else r"(?:[.!?][\"')\]]*|[,;:]|[\u2013\u2014])(?:\s+|$)"
    )
    spans: list[tuple[int, int]] = []
    start = 0
    for match in regex.finditer(boundary, text):
        end = match.end()
        left, right = _trimmed_span(text, start, end)
        if left < right:
            spans.append((left, right))
        start = end
    left, right = _trimmed_span(text, start, len(text))
    if left < right:
        spans.append((left, right))
    return tuple(spans) or ((0, len(text)),)


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _same_voice_context(node: PlanNode, index: int, step: int) -> str | None:
    candidate = index + step
    if not 0 <= candidate < len(node.chunks):
        return None
    if node.chunk_routes[candidate] != node.chunk_routes[index]:
        return None
    return node.chunks[candidate]


async def _generate_take(
    manifest: AudiobookManifest,
    plan: RepairPlan,
    *,
    number: int,
    root: Path,
    aligner: WindowSpeechAligner | None,
    services: dict[tuple[str, str | None], TextToSpeechService],
    stack: AsyncExitStack,
    api_key: str | None,
) -> RepairTake:
    directory = root / f"take-{number:02d}"
    paths: list[Path] = []
    reasons: list[str] = []
    cache_events: list[RepairCacheEvent] = []
    stage_cache = RepairStageCache(
        manifest.root / ".yakbox" / "cache" / "repair-pipeline"
    )
    for chunk in plan.chunks:
        profile = manifest.profile(_required_profile(chunk))
        service = await _repair_service(
            manifest,
            profile,
            services=services,
            stack=stack,
            api_key=api_key,
            log_path=root / "logs" / "local-worker.log",
        )
        destination = directory / f"{chunk.index + 1:04d}-{chunk.id}.wav"
        request = _repair_request(
            manifest,
            profile,
            chunk,
            take=number,
        )
        final_key = _repair_candidate_fingerprint(
            manifest,
            plan,
            chunk,
            request=request,
            aligner=aligner,
        )
        final_event = stage_cache.restore_audio(
            chunk_id=chunk.id,
            stage="verified-candidate",
            key=final_key,
            destination=destination,
        )
        cache_events.append(final_event)
        if final_event.hit:
            paths.append(destination.resolve())
            continue
        cached_service = CachedRepairSpeechService(
            service,
            stage_cache,
            chunk_id=chunk.id,
            events=cache_events,
        )
        prior_reason_count = len(reasons)
        try:
            if plan.mode is RepairMode.CONTEXT:
                if aligner is None:
                    raise BuildError("Context repair requires Whisper alignment")
                await _synthesize_context_candidate(
                    manifest,
                    chunk,
                    request=request,
                    service=cached_service,
                    aligner=aligner,
                    destination=destination,
                    take=number,
                )
            elif plan.mode in {RepairMode.SENTENCE, RepairMode.CLAUSE}:
                if aligner is None:
                    raise BuildError("Sub-chunk repair requires Whisper alignment")
                await _synthesize_subchunk_candidate(
                    manifest,
                    plan,
                    chunk,
                    request=request,
                    service=cached_service,
                    aligner=aligner,
                    destination=destination,
                    stage_cache=stage_cache,
                    cache_events=cache_events,
                )
                reasons.extend(
                    await _qa_direct_candidate(manifest, chunk, destination, aligner)
                )
            else:
                await cached_service.synthesize_to_file(
                    request, destination, overwrite=True
                )
                reasons.extend(
                    await _qa_direct_candidate(manifest, chunk, destination, aligner)
                )
        except BuildError:
            reasons.append(f"chunk_{chunk.id}_generation_failed")
            continue
        if len(reasons) == prior_reason_count:
            stage_cache.store_audio(
                stage="verified-candidate",
                key=final_key,
                source=destination,
            )
        paths.append(destination.resolve())
    accepted = len(paths) == len(plan.chunks) and not reasons
    audition = _write_audition_playlist(directory, paths)
    return RepairTake(
        number=number,
        accepted=accepted,
        reason_codes=tuple(dict.fromkeys(reasons)),
        audition_path=audition,
        chunk_paths=tuple(paths),
        chunk_sha256s=tuple(sha256_file(path) for path in paths),
        cache_events=tuple(cache_events),
    )


async def _repair_service(
    manifest: AudiobookManifest,
    profile: BackendProfile,
    *,
    services: dict[tuple[str, str | None], TextToSpeechService],
    stack: AsyncExitStack,
    api_key: str | None,
    log_path: Path,
) -> TextToSpeechService:
    from yakbox.audiobook.build import (  # noqa: PLC0415 - avoids build cycle
        _is_local_backend,
        _profile_device,
        _profile_threads,
        _profile_worker_timeout,
    )

    key = (profile.backend.casefold(), _profile_device(profile))
    service = services.get(key)
    if service is not None:
        return service
    service = await stack.enter_async_context(
        open_speech_backend(
            profile.backend,
            api_key=api_key,
            isolated_local=_is_local_backend(profile.backend),
            device=_profile_device(profile),
            local_worker_timeout_seconds=_profile_worker_timeout(profile),
            local_threads_per_process=_profile_threads(profile),
            local_worker_log_path=log_path,
            local_runtime_workspace=manifest.root if manifest.runtime.enabled else None,
            local_runtime_idle_timeout_seconds=manifest.runtime.idle_timeout_seconds,
            local_conditioning_cache_size=manifest.runtime.conditioning_cache_size,
            local_runtime_maximum_memory_bytes=manifest.runtime.maximum_memory_bytes,
        )
    )
    services[key] = service
    return service


def _repair_request(
    manifest: AudiobookManifest,
    profile: BackendProfile,
    chunk: RepairChunk,
    *,
    take: int,
) -> SpeechSynthesisRequest:
    from yakbox.audiobook.build import (  # noqa: PLC0415 - avoids build cycle
        _resolved_speech,
    )

    voice, rate, project, use_hd, reference, chatterbox = _resolved_speech(
        profile, manifest
    )
    varied = (
        replace(chatterbox, seed=_repair_seed(chunk.text_sha256, chunk.id, take))
        if chatterbox is not None
        else None
    )
    return SpeechSynthesisRequest(
        text=chunk.replacement_text or chunk.text,
        voice=voice,
        backend=profile.backend,
        profile=profile.name,
        output_format=AudioFormat.WAV,
        sample_rate=rate,
        project=project,
        use_hd=use_hd,
        reference_audio=reference,
        chatterbox=varied,
    )


def _repair_candidate_fingerprint(
    manifest: AudiobookManifest,
    plan: RepairPlan,
    chunk: RepairChunk,
    *,
    request: SpeechSynthesisRequest,
    aligner: WindowSpeechAligner | None,
) -> str:
    """Identify the complete verified repair pipeline without session IDs."""
    policy = manifest.short_utterances
    original_sha256 = None
    if plan.mode in {RepairMode.SENTENCE, RepairMode.CLAUSE}:
        try:
            original_sha256 = sha256_file(_current_chunk_audio(manifest, plan, chunk))
        except BuildError:
            original_sha256 = "unavailable"
    payload = {
        "version": "repair-pipeline-v2",
        "mode": plan.mode.value,
        "chunk_id": chunk.id,
        "text_sha256": chunk.text_sha256,
        "replacement_sha256": hashlib.sha256(
            (chunk.replacement_text or chunk.text).encode()
        ).hexdigest(),
        "replacement_start": chunk.replacement_start,
        "replacement_end": chunk.replacement_end,
        "speech_request": speech_request_fingerprint(request),
        "aligner": aligner.fingerprint if aligner is not None else None,
        "generation_policy": policy.generation_fingerprint,
        "extraction_policy": policy.extraction_fingerprint,
        "evaluation_policy": policy.evaluation_fingerprint,
        "join_window_seconds": policy.join_inspection_window_seconds,
        "original_audio_sha256": original_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def _synthesize_subchunk_candidate(
    manifest: AudiobookManifest,
    plan: RepairPlan,
    chunk: RepairChunk,
    *,
    request: SpeechSynthesisRequest,
    service: TextToSpeechService,
    aligner: WindowSpeechAligner,
    destination: Path,
    stage_cache: RepairStageCache,
    cache_events: list[RepairCacheEvent],
) -> None:
    """Replace one aligned sentence or clause inside an existing chunk."""
    replacement_text = chunk.replacement_text or chunk.text
    if replacement_text == chunk.text:
        await service.synthesize_to_file(request, destination, overwrite=True)
        replacement_reasons = await _qa_direct_candidate(
            manifest,
            replace(chunk, text=replacement_text),
            destination,
            aligner,
        )
        if replacement_reasons:
            raise BuildError(
                "Sub-chunk candidate failed QA: " + ", ".join(replacement_reasons)
            )
        return
    original = _current_chunk_audio(manifest, plan, chunk)
    original_alignment = await aligner.align(
        original,
        chunk.text,
        language=manifest.book.language,
    )
    policy = manifest.short_utterances
    decision = validate_carrier_alignment(
        original_alignment,
        expected_text=chunk.text,
        target_text=replacement_text,
        minimum_confidence=policy.minimum_alignment_confidence,
        token_aliases=policy.alignment_alias_map,
        maximum_internal_token_gap_ms=policy.maximum_internal_token_gap_ms,
        maximum_token_duration_ms=policy.maximum_token_duration_ms,
    )
    if not decision.accepted:
        raise BuildError(
            "Cannot locate a safe sub-chunk splice: " + ", ".join(decision.reason_codes)
        )
    start = _required_time(decision.start_seconds)
    end = _required_time(decision.end_seconds)
    replacement = destination.with_name(f"{destination.stem}-replacement.wav")
    carrier = destination.with_name(f"{destination.stem}-carrier.wav")
    try:
        extraction = await _synthesize_subchunk_with_context(
            manifest,
            chunk,
            request=request,
            service=service,
            aligner=aligner,
            carrier=carrier,
            replacement=replacement,
            stage_cache=stage_cache,
            cache_events=cache_events,
        )
        replacement_reasons = await _qa_direct_candidate(
            manifest,
            replace(chunk, text=replacement_text),
            replacement,
            aligner,
        )
        if replacement_reasons:
            raise BuildError(
                "Sub-chunk candidate failed QA: " + ", ".join(replacement_reasons)
            )
        splice_key = stage_cache.key(
            "dsp-splice",
            {
                "version": "adaptive-splice-v1",
                "original_sha256": sha256_file(original),
                "replacement_sha256": sha256_file(replacement),
                "start_seconds": start,
                "end_seconds": end,
            },
        )
        splice_event = stage_cache.restore_audio(
            chunk_id=chunk.id,
            stage="dsp-splice",
            key=splice_key,
            destination=destination,
        )
        cache_events.append(splice_event)
        if splice_event.hit:
            output_duration = _wav_duration(destination)
            splice_evidence: dict[str, object] = {
                "cache": "content_match",
                "output_duration_seconds": output_duration,
            }
        else:
            splice = splice_wav_region(
                original,
                replacement,
                destination,
                start_seconds=start,
                end_seconds=end,
                overwrite=True,
            )
            output_duration = splice.output_duration_seconds
            splice_evidence = splice.to_dict()
            stage_cache.store_audio(
                stage="dsp-splice",
                key=splice_key,
                source=destination,
            )
        reconstructed_reasons = await _qa_direct_candidate(
            manifest,
            chunk,
            destination,
            aligner,
        )
        if reconstructed_reasons:
            raise BuildError(
                "Reconstructed sub-chunk failed QA: " + ", ".join(reconstructed_reasons)
            )
        replacement_duration = output_duration - (
            _wav_duration(original) - (end - start)
        )
        second_join = max(start, start + replacement_duration)
        report = await inspect_joins(
            destination,
            tuple(
                JoinSpecification(
                    at_seconds,
                    boundary=f"repair_{plan.mode.value}",
                )
                for at_seconds in (start, second_join)
                if 0 < at_seconds < output_duration
            ),
            language=manifest.book.language,
            model=policy.alignment_model,
            revision=policy.alignment_revision,
            window_seconds=policy.join_inspection_window_seconds,
            aligner=aligner,
        )
        if not report.accepted:
            raise BuildError("Sub-chunk repair introduced an unsafe audio join")
        atomic_write_json(
            destination.with_name(f"{destination.stem}-splice.json"),
            {
                **runtime_metadata("audiobook-repair-splice"),
                "chunk_id": chunk.id,
                "mode": plan.mode.value,
                "carrier_extraction": extraction,
                "splice": splice_evidence,
                "reconstructed_qa": {"accepted": True, "reason_codes": []},
                "join_qa": {"accepted": report.accepted},
            },
        )
    finally:
        for path in (replacement, carrier):
            path.unlink(missing_ok=True)


async def _synthesize_subchunk_with_context(
    manifest: AudiobookManifest,
    chunk: RepairChunk,
    *,
    request: SpeechSynthesisRequest,
    service: TextToSpeechService,
    aligner: WindowSpeechAligner,
    carrier: Path,
    replacement: Path,
    stage_cache: RepairStageCache,
    cache_events: list[RepairCacheEvent],
) -> dict[str, object]:
    """Speak hidden adjacent prose and extract only the requested linguistic span."""
    before = chunk.text[: chunk.replacement_start].strip()
    after = chunk.text[chunk.replacement_end :].strip()
    if not before:
        before = chunk.previous_context or "A quiet moment passed."
    if not after:
        after = chunk.next_context or "Then the conversation continued."
    target = chunk.replacement_text or chunk.text
    carrier_text = " ".join(value for value in (before, target, after) if value)
    carrier_request = replace(request, text=carrier_text)
    await service.synthesize_to_file(carrier_request, carrier, overwrite=True)
    policy = manifest.short_utterances
    extraction_key = stage_cache.key(
        "context-extraction",
        {
            "carrier_sha256": sha256_file(carrier),
            "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
            "aligner": aligner.fingerprint,
            "extraction_policy": policy.extraction_fingerprint,
        },
    )
    extraction_event = stage_cache.restore_audio(
        chunk_id=chunk.id,
        stage="context-extraction",
        key=extraction_key,
        destination=replacement,
    )
    cache_events.append(extraction_event)
    if extraction_event.hit:
        return {
            "carrier_sha256": sha256_file(carrier),
            "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
            "cache": "content_match",
        }
    alignment = await aligner.align(
        carrier,
        carrier_text,
        language=manifest.book.language,
    )
    decision = validate_carrier_alignment(
        alignment,
        expected_text=carrier_text,
        target_text=target,
        minimum_confidence=policy.minimum_alignment_confidence,
        token_aliases=policy.alignment_alias_map,
        minimum_average_log_probability=policy.minimum_segment_average_log_probability,
        maximum_compression_ratio=policy.maximum_segment_compression_ratio,
        maximum_no_speech_probability=policy.maximum_segment_no_speech_probability,
        maximum_temperature=policy.maximum_segment_temperature,
        maximum_internal_token_gap_ms=policy.maximum_internal_token_gap_ms,
        maximum_token_duration_ms=policy.maximum_token_duration_ms,
    )
    if not decision.accepted:
        raise BuildError(
            "Cannot extract context-rendered sub-chunk: "
            + ", ".join(decision.reason_codes)
        )
    crop = crop_aligned_wav(
        carrier,
        replacement,
        start_seconds=_required_time(decision.start_seconds),
        end_seconds=_required_time(decision.end_seconds),
        pre_roll_ms=policy.pre_roll_ms,
        post_roll_ms=policy.post_roll_ms,
        fade_ms=0,
        speech_regions=alignment.speech_regions,
        overwrite=True,
    )
    stage_cache.store_audio(
        stage="context-extraction",
        key=extraction_key,
        source=replacement,
    )
    return {
        "carrier_sha256": sha256_file(carrier),
        "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
        "crop": {
            "source_start_seconds": crop.source_start_seconds,
            "source_end_seconds": crop.source_end_seconds,
            "crop_start_seconds": crop.crop_start_seconds,
            "crop_end_seconds": crop.crop_end_seconds,
            "pre_roll_ms": crop.pre_roll_ms,
            "post_roll_ms": crop.post_roll_ms,
        },
    }


def _current_chunk_audio(
    manifest: AudiobookManifest,
    plan: RepairPlan,
    chunk: RepairChunk,
) -> Path:
    path = assembly_manifest_path(
        manifest.root,
        target=plan.target,
        chapter_id=plan.chapter_id,
    )
    raw = load_assembly_manifest(path, workspace=manifest.root)
    chunks = raw.get("chunks")
    if not isinstance(chunks, list):
        raise BuildError("Assembly timeline has no chunks")
    entry = next(
        (
            cast(dict[str, object], item)
            for item in chunks
            if isinstance(item, dict) and item.get("id") == chunk.id
        ),
        None,
    )
    if entry is None or entry.get("text_sha256") != chunk.text_sha256:
        raise BuildError("Assembly timeline is stale for the repair chunk")
    fingerprint = entry.get("cache_fingerprint")
    if not isinstance(fingerprint, str):
        raise BuildError("Repair chunk has no reusable synthesis cache entry")
    from yakbox.audiobook.build import (  # noqa: PLC0415 - avoids build cycle
        _cached_chunk,
    )

    cached = _cached_chunk(manifest.root, fingerprint)
    if cached is None:
        raise BuildError("Original repair chunk is missing from synthesis cache")
    return cached


def _wav_duration(path: Path) -> float:
    return wav_duration_seconds(path)


def _required_time(value: float | None) -> float:
    if value is None:
        raise BuildError("Alignment did not provide a required splice boundary")
    return value


async def _synthesize_context_candidate(
    manifest: AudiobookManifest,
    chunk: RepairChunk,
    *,
    request: SpeechSynthesisRequest,
    service: TextToSpeechService,
    aligner: WindowSpeechAligner,
    destination: Path,
    take: int,
) -> None:
    policy = replace(
        manifest.short_utterances,
        candidate_count=max(8, manifest.short_utterances.candidate_count),
        require_review_for_one_word=False,
        keep_candidates=True,
    )
    recipes = carrier_recipes(
        chunk.text,
        policy,
        seed_material=f"repair:{chunk.id}:{take}",
        previous_context=chunk.previous_context,
        next_context=chunk.next_context,
    )
    preferred = next((recipe for recipe in recipes if recipe.natural), None)
    if preferred is None:
        preferred = next(
            (recipe for recipe in recipes if recipe.position.value == "middle"),
            recipes[0],
        )
    recipe = replace(
        preferred,
        candidate_index=take,
        seed=_repair_seed("context", chunk.id, take),
    )
    await synthesize_short_utterance(
        service=service,
        aligner=aligner,
        request=request,
        destination=destination,
        recipes=(recipe,),
        policy=policy,
        language=manifest.book.language,
        qa_directory=destination.parent / f"{destination.stem}-qa",
        candidate_cache_directory=(
            manifest.root / ".yakbox" / "cache" / "short-utterance-candidates"
        ),
    )


async def _qa_direct_candidate(
    manifest: AudiobookManifest,
    chunk: RepairChunk,
    path: Path,
    aligner: WindowSpeechAligner | None,
) -> tuple[str, ...]:
    _trim_direct_candidate_edges(manifest, path)
    reasons = list(
        _signal_reasons(
            manifest,
            path,
            enforce_single_utterance=(
                len(lexical_tokens(chunk.text)) <= _MAXIMUM_SINGLE_UTTERANCE_WORDS
            ),
        )
    )
    if aligner is not None:
        result = await aligner.align(path, chunk.text, language=manifest.book.language)
        reasons.extend(
            evaluate_alignment(
                result,
                clip_type=classify_clip_type(chunk.text),
                expected_text=chunk.text,
            ).reason_codes
        )
    return tuple(dict.fromkeys(reasons))


def _trim_direct_candidate_edges(manifest: AudiobookManifest, path: Path) -> None:
    """Remove excessive model-added edge silence without touching speech."""
    policy = manifest.short_utterances
    evidence = inspect_speech_islands(
        path,
        threshold_dbfs=policy.acoustic_threshold_dbfs,
        island_gap_ms=policy.speech_island_gap_ms,
    )
    if not evidence.regions:
        return
    first, last = evidence.regions[0], evidence.regions[-1]
    leading_ms = first.start_seconds * 1_000
    trailing_ms = max(0.0, evidence.duration_seconds - last.end_seconds) * 1_000
    if (
        leading_ms <= policy.maximum_edge_silence_ms
        and trailing_ms <= policy.maximum_edge_silence_ms
    ):
        return
    trimmed = path.with_name(f"{path.stem}-edge-trimmed{path.suffix}")
    try:
        crop_aligned_wav(
            path,
            trimmed,
            start_seconds=first.start_seconds,
            end_seconds=last.end_seconds,
            pre_roll_ms=policy.pre_roll_ms,
            post_roll_ms=policy.post_roll_ms,
            fade_ms=policy.fade_ms,
            speech_regions=evidence.regions,
            overwrite=True,
        )
        trimmed.replace(path)
    finally:
        trimmed.unlink(missing_ok=True)


def _signal_reasons(
    manifest: AudiobookManifest,
    path: Path,
    *,
    enforce_single_utterance: bool = True,
) -> tuple[str, ...]:
    evidence = inspect_signal_quality(path)
    islands = inspect_speech_islands(
        path,
        threshold_dbfs=manifest.short_utterances.acoustic_threshold_dbfs,
        island_gap_ms=manifest.short_utterances.speech_island_gap_ms,
    )
    policy = manifest.short_utterances
    reasons: list[str] = []
    if evidence.clipped_sample_ratio > policy.maximum_clipped_sample_ratio:
        reasons.append("clipping")
    if (
        max(evidence.leading_boundary_jump_ratio, evidence.trailing_boundary_jump_ratio)
        > policy.maximum_boundary_jump_ratio
    ):
        reasons.append("boundary_click")
    if evidence.longest_stationary_voiced_ms > policy.maximum_stationary_voiced_ms:
        reasons.append("stationary_voicing")
    if enforce_single_utterance:
        if islands.detached_prefix:
            reasons.append("unexpected_prefix_speech")
        if islands.detached_suffix:
            reasons.append("unexpected_suffix_speech")
        if islands.leading_silence_ms > policy.maximum_edge_silence_ms:
            reasons.append("excessive_leading_silence")
        if islands.trailing_silence_ms > policy.maximum_edge_silence_ms:
            reasons.append("excessive_trailing_silence")
    return tuple(reasons)


def _repair_aligner(manifest: AudiobookManifest) -> WindowSpeechAligner:
    policy = manifest.short_utterances
    if manifest.runtime.enabled:
        from yakbox.local_runtime import (  # noqa: PLC0415 - optional backend
            LocalRuntimeOptions,
            PersistentMlxWhisperAligner,
        )

        return PersistentMlxWhisperAligner(
            manifest.root,
            model=policy.alignment_model,
            revision=policy.alignment_revision,
            timeout_seconds=policy.alignment_timeout_seconds,
            prompted_timing=policy.prompted_timing,
            decode_consensus=policy.decode_consensus,
            prompt_sensitivity=policy.prompt_sensitivity,
            maximum_consensus_timing_delta_ms=(
                policy.maximum_consensus_timing_delta_ms
            ),
            hallucination_silence_threshold=policy.hallucination_silence_threshold,
            runtime_options=LocalRuntimeOptions(
                idle_timeout_seconds=manifest.runtime.idle_timeout_seconds,
                conditioning_cache_size=manifest.runtime.conditioning_cache_size,
                maximum_memory_bytes=manifest.runtime.maximum_memory_bytes,
            ),
        )
    return open_local_aligner(
        policy.alignment_backend,
        model=policy.alignment_model,
        revision=policy.alignment_revision,
        timeout_seconds=policy.alignment_timeout_seconds,
        prompted_timing=policy.prompted_timing,
        decode_consensus=policy.decode_consensus,
        prompt_sensitivity=policy.prompt_sensitivity,
        maximum_consensus_timing_delta_ms=policy.maximum_consensus_timing_delta_ms,
        hallucination_silence_threshold=policy.hallucination_silence_threshold,
    )


def _write_audition_playlist(directory: Path, paths: list[Path]) -> Path:
    if len(paths) == 1:
        return paths[0]
    destination = directory / "audition.m3u"
    content = "#EXTM3U\n" + "\n".join(path.name for path in paths) + "\n"
    atomic_write_bytes(destination, content.encode(), overwrite=True)
    return destination.resolve()


def _write_batch_review_playlist(
    directory: Path,
    sessions: tuple[RepairSession, ...],
) -> Path:
    destination = directory / "review.m3u"
    entries = [
        Path(os.path.relpath(take.audition_path, start=directory)).as_posix()
        for session in sessions
        for take in session.takes
    ]
    content = "#EXTM3U\n" + "\n".join(entries) + "\n"
    atomic_write_bytes(destination, content.encode(), overwrite=True)
    return destination.resolve()


def _load_session(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"Cannot read repair session {path}: {error}") from error
    if not isinstance(raw, dict) or raw.get("$schema") != schema_uri(
        "audiobook-repair-session"
    ):
        raise ValidationError(f"Unsupported repair session: {path}")
    return raw


def _session_take(raw: dict[str, object], number: int) -> dict[str, object]:
    values = raw.get("takes")
    if not isinstance(values, list):
        raise ValidationError("Repair session takes are invalid")
    selected = next(
        (
            value
            for value in values
            if isinstance(value, dict) and value.get("number") == number
        ),
        None,
    )
    if not isinstance(selected, dict):
        raise ValidationError(f"Repair session has no take {number}")
    return cast(dict[str, object], selected)


def _required_string(
    raw: dict[str, object], key: str, *, parent: str | None = None
) -> str:
    value: object = raw
    if parent is not None:
        value = raw.get(parent)
    if not isinstance(value, dict) or not isinstance(value.get(key), str):
        raise ValidationError(f"Repair session field {key!r} is invalid")
    return str(cast(dict[str, object], value)[key])


def _required_mapping(value: object, key: str) -> object:
    if not isinstance(value, dict) or key not in value:
        raise ValidationError(f"Repair session field {key!r} is invalid")
    return cast(dict[str, object], value)[key]


def _required_profile(chunk: RepairChunk) -> str:
    if chunk.route.profile is None:
        raise ValidationError("Repair selected an explicit pause")
    return chunk.route.profile


def _plan_profiles(plan: RepairPlan) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_required_profile(chunk) for chunk in plan.chunks))


def _workspace_path(path: Path, workspace: Path) -> str:
    resolved = path.resolve()
    return (
        resolved.relative_to(workspace).as_posix()
        if resolved.is_relative_to(workspace)
        else resolved.as_posix()
    )


def _repair_seed(text_sha256: str, chunk_id: str, take: int) -> int:
    material = f"repair-v2\0{text_sha256}\0{chunk_id}\0{take}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
