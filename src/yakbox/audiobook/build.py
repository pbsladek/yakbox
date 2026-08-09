from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import wave
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_json, sha256_file
from yakbox.audio import (
    AudioQualityPolicy,
    assemble_m4b,
    encode_mp3,
    inspect_audio,
    master_wav,
)
from yakbox.audio.master import copy_audio
from yakbox.audiobook.artifacts import (
    ArtifactKind,
    ArtifactRecord,
    verify_artifact,
    write_artifact_record,
)
from yakbox.audiobook.journal import RunJournal, new_run_id, target_lock
from yakbox.audiobook.manifest import (
    AudiobookManifest,
    BackendProfile,
    BuildTarget,
    ChatterboxOptions,
    FakeOptions,
    LogicalVoice,
    ResembleOptions,
)
from yakbox.audiobook.planner import (
    BuildPlan,
    BuildStage,
    ChunkRoute,
    PlanNode,
    ShortUtteranceMarker,
    plan_audiobook,
)
from yakbox.audiobook.sources import (
    Chapter,
    NormalizedDocument,
    SpeechSegment,
    apply_pronunciations,
    normalize_sources,
)
from yakbox.contracts import runtime_metadata, schema_uri
from yakbox.errors import (
    ArtifactError,
    BackendUnavailableError,
    BuildError,
    ValidationError,
)
from yakbox.fingerprints import (
    backend_fingerprint,
    backend_runtime_fingerprint,
    backend_versions,
    media_tool_fingerprint,
    media_tool_versions,
)
from yakbox.local_alignment import open_local_aligner
from yakbox.local_phoneme_alignment import open_phoneme_aligner
from yakbox.speech import (
    AudioFormat,
    BatchTextToSpeechService,
    ChatterboxSynthesisOptions,
    CurrencyCode,
    HostedUsageBudget,
    HostedUsageJournalingService,
    HostedUsageReportingService,
    HostedUsageSnapshot,
    HostedWorkEstimate,
    PricingSourceId,
    SpeechSynthesisRequest,
    TextToSpeechService,
    estimate_hosted_work,
    open_speech_backend,
    validate_hosted_preflight,
)
from yakbox.speech.alignment import SpeechAligner, WindowSpeechAligner
from yakbox.speech.chunking import CHATTERBOX_CHUNK_CHARACTERS, chunk_text
from yakbox.speech.phonemes import PhonemeAligner
from yakbox.speech.short_synthesis import synthesize_short_utterance
from yakbox.speech.short_utterances import (
    CarrierRecipe,
    ShortUtteranceFailure,
    ShortUtteranceStrategy,
    carrier_recipes,
)
from yakbox.speech.waves import (
    WavJoinBoundary,
    WavJoinPart,
    concatenate_wavs,
    wav_join_boundaries,
    write_silence,
)
from yakbox.whisper_cache import CachedWhisperAligner
from yakbox.whisper_qa import JoinSpecification, inspect_joins, verify_manuscript

_PAUSE = re.compile(r"__YAKBOX_PAUSE_MS=(\d+)__")
_STORAGE_BYTES_PER_CHARACTER = 16_000
_STORAGE_BYTES_PER_CHAPTER_FLOOR = 256 * 1024
_EXECUTION_STAGES = (
    BuildStage.SYNTHESIZE,
    BuildStage.MASTER,
    BuildStage.VERIFY_MANUSCRIPT,
    BuildStage.ENCODE_MP3,
    BuildStage.INSPECT,
)


@dataclass(frozen=True, slots=True)
class BuildChangeSummary:
    """Differences between a planned build and its most recent prior run."""

    previous_run_id: str | None
    previous_plan_fingerprint: str | None
    added_nodes: tuple[str, ...]
    removed_nodes: tuple[str, ...]
    changed_nodes: tuple[str, ...]
    unchanged_nodes: tuple[str, ...]
    reasons: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize plan changes and their reasons."""
        return {
            "previous_run_id": self.previous_run_id,
            "previous_plan_fingerprint": self.previous_plan_fingerprint,
            "added_nodes": list(self.added_nodes),
            "removed_nodes": list(self.removed_nodes),
            "changed_nodes": list(self.changed_nodes),
            "unchanged_nodes": list(self.unchanged_nodes),
            "reasons": dict(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class BuildPreflight:
    """Reuse, storage, and hosted-work estimates computed before a build."""

    target: str
    profile: str
    planned_nodes: int
    reusable_nodes: int
    pending_nodes: int
    pending_chapters: int
    estimated_output_bytes: int
    available_bytes: int
    storage_budget_bytes: int | None
    hosted_work: HostedWorkEstimate | None
    reusable_node_ids: tuple[str, ...]
    change_summary: BuildChangeSummary
    from_stage: BuildStage = BuildStage.SYNTHESIZE
    through_stage: BuildStage = BuildStage.INSPECT
    short_utterance_chunks: int = 0
    maximum_short_utterance_generations: int = 0

    @property
    def storage_sufficient(self) -> bool:
        """Return whether free space and configured storage budget are sufficient."""
        within_free_space = self.estimated_output_bytes <= self.available_bytes
        within_budget = (
            self.storage_budget_bytes is None
            or self.estimated_output_bytes <= self.storage_budget_bytes
        )
        return within_free_space and within_budget

    def to_dict(self) -> dict[str, object]:
        """Serialize the preflight estimate and reuse summary."""
        return {
            "target": self.target,
            "profile": self.profile,
            "planned_nodes": self.planned_nodes,
            "reusable_nodes": self.reusable_nodes,
            "pending_nodes": self.pending_nodes,
            "pending_chapters": self.pending_chapters,
            "estimated_output_bytes": self.estimated_output_bytes,
            "available_bytes": self.available_bytes,
            "storage_budget_bytes": self.storage_budget_bytes,
            "storage_sufficient": self.storage_sufficient,
            "reusable_node_ids": list(self.reusable_node_ids),
            "hosted_work": (
                self.hosted_work.to_dict() if self.hosted_work is not None else None
            ),
            "change_summary": self.change_summary.to_dict(),
            "from_stage": self.from_stage.value,
            "through_stage": self.through_stage.value,
            "short_utterance_chunks": self.short_utterance_chunks,
            "maximum_short_utterance_generations": (
                self.maximum_short_utterance_generations
            ),
        }


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Terminal outcome, artifacts, and usage evidence for an audiobook build."""

    schema_version: int
    run_id: str
    target: str
    status: BuildStatus
    plan_fingerprint: str
    artifacts: tuple[ArtifactRecord, ...]
    reused_nodes: tuple[str, ...]
    failed_nodes: tuple[str, ...]
    run_directory: Path
    hosted_usage: HostedUsageSnapshot | None
    preflight: BuildPreflight
    resumed: bool


@dataclass(frozen=True, slots=True)
class BuildProgress:
    """One stable progress event emitted while executing a build plan."""

    event: BuildProgressEvent
    node_id: str
    stage: BuildStage
    completed: int
    total: int
    reused: bool = False
    error: str | None = None


type BuildProgressCallback = Callable[[BuildProgress], None]


class BuildStatus(StrEnum):
    """Terminal or planned state of an audiobook build."""

    PLANNED = "planned"
    COMPLETE = "complete"
    FAILED = "failed"


class BuildProgressEvent(StrEnum):
    """Stable event names delivered to build progress callbacks."""

    STARTED = "started"
    REUSED = "reused"
    FAILED = "failed"
    COMPLETED = "completed"
    NOT_RUN = "not_run"


@dataclass(frozen=True, slots=True)
class BuildRequest:
    """Typed options for one audiobook build operation."""

    target_name: str = "default"
    profile_override: str | None = None
    chapter_selector: str | None = None
    dry_run: bool = False
    resume: bool = True
    api_key: str | None = None
    max_submitted_characters: int | None = None
    max_provider_requests: int | None = None
    max_estimated_spend: Decimal | None = None
    currency: str | None = None
    pricing_source: str | None = None
    price_per_character: Decimal | None = None
    confirm_above_characters: int | None = None
    confirm_above_requests: int | None = None
    from_stage: BuildStage | str | None = None
    through_stage: BuildStage | str | None = None


@dataclass(slots=True)
class _BuildProgressTracker:
    callback: BuildProgressCallback | None
    total: int
    completed: int = 0

    def emit(
        self,
        event: BuildProgressEvent,
        node: PlanNode,
        *,
        terminal: bool = False,
        reused: bool = False,
        error: str | None = None,
    ) -> None:
        if terminal:
            self.completed += 1
        if self.callback is None:
            return
        try:
            self.callback(
                BuildProgress(
                    event=event,
                    node_id=node.id,
                    stage=node.stage,
                    completed=self.completed,
                    total=self.total,
                    reused=reused,
                    error=error,
                )
            )
        except Exception:  # noqa: BLE001 - progress callbacks cannot fail a build
            return


@dataclass(frozen=True, slots=True)
class ReleaseCheck:
    """Release-readiness result with issues and expected delivery artifacts."""

    complete: bool
    issues: tuple[str, ...]
    master_wavs: tuple[Path, ...]
    delivery_mp3s: tuple[Path, ...]
    release_manifest: Path | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize release completeness and immutable evidence paths."""
        return {
            **runtime_metadata("audiobook-release-check"),
            "complete": self.complete,
            "issues": list(self.issues),
            "master_wavs": [str(path) for path in self.master_wavs],
            "delivery_mp3s": [str(path) for path in self.delivery_mp3s],
            "release_manifest": (
                str(self.release_manifest) if self.release_manifest else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ReleaseDiff:
    """Artifact and metadata differences between two immutable releases."""

    left_release_id: str
    right_release_id: str
    added_artifacts: tuple[str, ...]
    removed_artifacts: tuple[str, ...]
    changed_artifacts: tuple[str, ...]
    metadata_changes: tuple[str, ...]

    @property
    def identical(self) -> bool:
        """Return whether the two releases have no reported differences."""
        return not (
            self.added_artifacts
            or self.removed_artifacts
            or self.changed_artifacts
            or self.metadata_changes
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize added, removed, changed, and metadata differences."""
        return {
            "schema_version": 1,
            "left_release_id": self.left_release_id,
            "right_release_id": self.right_release_id,
            "identical": self.identical,
            "added_artifacts": list(self.added_artifacts),
            "removed_artifacts": list(self.removed_artifacts),
            "changed_artifacts": list(self.changed_artifacts),
            "metadata_changes": list(self.metadata_changes),
        }


def select_build_chapters(
    manifest: AudiobookManifest,
    *,
    selection: str,
    target_name: str = "default",
    profile_override: str | None = None,
    chapter_selector: str | None = None,
    since_release: Path | None = None,
    from_stage: BuildStage | str | None = None,
    through_stage: BuildStage | str | None = None,
) -> tuple[str, ...]:
    """Resolve changed, failed, or missing chapter selectors without mutation."""
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
    )
    plan = plan_audiobook(
        manifest,
        document,
        target_name=target_name,
        profile_override=profile_override,
        chapter_selector=chapter_selector,
    )
    target = manifest.target(target_name)
    _, nodes, _, _ = _select_stages(
        plan,
        from_stage=from_stage or target.from_stage,
        through_stage=through_stage or target.through_stage,
    )
    node_ids: set[str]
    selection_preflight: BuildPreflight | None = None
    if selection == "missing":
        selection_preflight = preflight_audiobook_build(
            manifest,
            target_name=target_name,
            profile_override=profile_override,
            chapter_selector=chapter_selector,
            from_stage=from_stage,
            through_stage=through_stage,
        )
        node_ids = {
            node.id
            for node in nodes
            if node.id not in set(selection_preflight.reusable_node_ids)
        }
    elif selection == "changed":
        if since_release is not None:
            release = _load_release_document(since_release)
            prior = _release_node_fingerprints(release)
            node_ids = {
                node.id for node in nodes if prior.get(node.id) != node.fingerprint
            }
        else:
            changes = _compare_to_previous_success(manifest, plan)
            node_ids = set(changes.added_nodes) | set(changes.changed_nodes)
    elif selection == "failed":
        node_ids = _latest_failed_node_ids(manifest.root, target_name)
    else:
        raise ValidationError(f"Unsupported build selection: {selection}")
    current_ids = {node.id for node in nodes}
    if selection_preflight is None:
        selection_preflight = preflight_audiobook_build(
            manifest,
            target_name=target_name,
            profile_override=profile_override,
            chapter_selector=chapter_selector,
            from_stage=from_stage,
            through_stage=through_stage,
        )
    pending_ids = current_ids - set(selection_preflight.reusable_node_ids)
    node_ids &= pending_ids
    return tuple(
        dict.fromkeys(node.chapter_id for node in nodes if node.id in node_ids)
    )


@dataclass(frozen=True, slots=True)
class _ExecutionOutcome:
    records: tuple[ArtifactRecord, ...]
    reused: tuple[str, ...]
    failed: tuple[str, ...]


def _execution_collections() -> tuple[list[ArtifactRecord], list[str], list[str]]:
    return [], [], []


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    plan: BuildPlan
    document: NormalizedDocument
    manifest: AudiobookManifest
    run_id: str
    service: TextToSpeechService
    journal: RunJournal
    media_semaphore: asyncio.Semaphore
    progress: _BuildProgressTracker


@dataclass(frozen=True, slots=True)
class _PendingShortUtterance:
    request: SpeechSynthesisRequest
    destination: Path
    fingerprint: str
    chunk_index: int
    recipes: tuple[CarrierRecipe, ...]
    qa_directory: Path | None


async def build_audiobook(
    manifest: AudiobookManifest,
    *,
    target_name: str = "default",
    profile_override: str | None = None,
    chapter_selector: str | None = None,
    dry_run: bool = False,
    resume: bool = True,
    api_key: str | None = None,
    max_submitted_characters: int | None = None,
    max_provider_requests: int | None = None,
    max_estimated_spend: Decimal | None = None,
    currency: str | None = None,
    pricing_source: str | None = None,
    price_per_character: Decimal | None = None,
    confirm_above_characters: int | None = None,
    confirm_above_requests: int | None = None,
    from_stage: BuildStage | str | None = None,
    through_stage: BuildStage | str | None = None,
    progress: BuildProgressCallback | None = None,
) -> BuildResult:
    """Execute a guarded, resumable audiobook build and return its durable result."""
    (
        document,
        plan,
        execution_nodes,
        resolved_from,
        resolved_through,
        profile,
        target,
    ) = _prepare_execution(
        manifest,
        target_name=target_name,
        profile_override=profile_override,
        chapter_selector=chapter_selector,
        from_stage=from_stage,
        through_stage=through_stage,
    )
    resolved_price = (
        price_per_character
        if price_per_character is not None
        else target.price_per_character
    )
    budget = _hosted_budget(
        target,
        max_submitted_characters=max_submitted_characters,
        max_provider_requests=max_provider_requests,
        max_estimated_spend=max_estimated_spend,
        currency=currency,
        pricing_source=pricing_source,
        confirm_above_characters=confirm_above_characters,
        confirm_above_requests=confirm_above_requests,
    )
    preflight = _preflight_for_plan(
        manifest,
        plan,
        profile,
        target,
        price_per_character=resolved_price,
        execution_nodes=execution_nodes,
        from_stage=resolved_from,
        through_stage=resolved_through,
    )
    if preflight.hosted_work is not None:
        validate_hosted_preflight(budget, preflight.hosted_work)
    _validate_storage_preflight(preflight)
    _validate_stage_prerequisites(
        plan,
        execution_nodes,
        target=target,
    )
    resumed_run = (
        _find_resumable_run(manifest.root, target_name, plan.fingerprint)
        if resume
        else None
    )
    run_id = resumed_run.name if resumed_run is not None else new_run_id()
    run_directory = (
        resumed_run
        if resumed_run is not None
        else manifest.root / ".yakbox" / "runs" / run_id
    )
    if dry_run:
        return BuildResult(
            schema_version=1,
            run_id=run_id,
            target=target_name,
            status=BuildStatus.PLANNED,
            plan_fingerprint=plan.fingerprint,
            artifacts=(),
            reused_nodes=(),
            failed_nodes=(),
            run_directory=run_directory,
            hosted_usage=None,
            preflight=preflight,
            resumed=False,
        )
    artifacts, reused, failed = _execution_collections()
    hosted_usage: HostedUsageSnapshot | None = None
    progress_tracker = _BuildProgressTracker(progress, len(execution_nodes))
    with target_lock(manifest.root / ".yakbox", target_name):
        run_directory.mkdir(parents=True, exist_ok=resumed_run is not None)
        if resumed_run is None:
            atomic_write_json(
                run_directory / "plan.json",
                {
                    **plan.to_dict(root=manifest.root),
                    "execution": {
                        "from_stage": resolved_from.value,
                        "through_stage": resolved_through.value,
                        "node_ids": [node.id for node in execution_nodes],
                    },
                    "change_summary": preflight.change_summary.to_dict(),
                    "preflight": preflight.to_dict(),
                },
                overwrite=False,
            )
        journal = RunJournal(run_directory / "journal.ndjson", run_id)
        prior_events: tuple[dict[str, object], ...] = ()
        if resumed_run is not None:
            prior_events = journal.events()
            journal.append("run_resumed", fingerprint=plan.fingerprint)
        else:
            journal.append("run_started", fingerprint=plan.fingerprint)
        try:
            async with (
                open_speech_backend(
                    profile.backend,
                    api_key=api_key,
                    isolated_local=_is_local_backend(profile.backend),
                    hosted_budget=budget,
                    price_per_character=resolved_price,
                    max_connections=target.provider_concurrency,
                    device=_profile_device(profile),
                    local_worker_timeout_seconds=_profile_worker_timeout(profile),
                    local_threads_per_process=_profile_threads(profile),
                    local_worker_log_path=run_directory / "logs" / "local-worker.log",
                ) as service,
                _journal_hosted_usage(service, journal, prior_events),
            ):
                if service.capabilities.hosted:
                    outcome = await _execute_hosted_plan(
                        plan,
                        execution_nodes=execution_nodes,
                        document=document,
                        manifest=manifest,
                        run_id=run_id,
                        service=service,
                        journal=journal,
                        concurrency=target.provider_concurrency,
                        media_semaphore=asyncio.Semaphore(target.media_concurrency),
                        progress=progress_tracker,
                    )
                else:
                    outcome = await _execute_node_chain(
                        execution_nodes,
                        plan=plan,
                        document=document,
                        manifest=manifest,
                        run_id=run_id,
                        service=service,
                        journal=journal,
                        media_semaphore=asyncio.Semaphore(target.media_concurrency),
                        progress=progress_tracker,
                    )
                artifacts.extend(outcome.records)
                reused.extend(outcome.reused)
                failed.extend(outcome.failed)
                if isinstance(service, HostedUsageReportingService):
                    hosted_usage = await service.usage_snapshot()
        except asyncio.CancelledError:
            journal.append(
                "run_interrupted",
                fingerprint=plan.fingerprint,
                error="build cancelled",
                usage=_usage_dict(_latest_journaled_usage(journal.events())),
            )
            raise
        status = BuildStatus.FAILED if failed else BuildStatus.COMPLETE
        journal.append(
            f"run_{status.value}",
            fingerprint=plan.fingerprint,
            usage=_usage_dict(hosted_usage),
        )
        result = BuildResult(
            schema_version=1,
            run_id=run_id,
            target=target_name,
            status=status,
            plan_fingerprint=plan.fingerprint,
            artifacts=tuple(artifacts),
            reused_nodes=tuple(reused),
            failed_nodes=tuple(failed),
            run_directory=run_directory,
            hosted_usage=hosted_usage,
            preflight=preflight,
            resumed=resumed_run is not None,
        )
        atomic_write_json(run_directory / "run.json", _result_dict(result))
    if failed:
        detail = _failed_node_detail(journal, failed[0])
        suffix = f": {detail}" if detail else ""
        raise BuildError(f"Build failed at {failed[0]}{suffix}; see {run_directory}")
    return result


async def run_audiobook_build(
    manifest: AudiobookManifest,
    request: BuildRequest,
    *,
    progress: BuildProgressCallback | None = None,
) -> BuildResult:
    """Run a build from a typed request while preserving the legacy call surface."""
    return await build_audiobook(
        manifest,
        target_name=request.target_name,
        profile_override=request.profile_override,
        chapter_selector=request.chapter_selector,
        dry_run=request.dry_run,
        resume=request.resume,
        api_key=request.api_key,
        max_submitted_characters=request.max_submitted_characters,
        max_provider_requests=request.max_provider_requests,
        max_estimated_spend=request.max_estimated_spend,
        currency=request.currency,
        pricing_source=request.pricing_source,
        price_per_character=request.price_per_character,
        confirm_above_characters=request.confirm_above_characters,
        confirm_above_requests=request.confirm_above_requests,
        from_stage=request.from_stage,
        through_stage=request.through_stage,
        progress=progress,
    )


def _prepare_execution(
    manifest: AudiobookManifest,
    *,
    target_name: str,
    profile_override: str | None,
    chapter_selector: str | None,
    from_stage: BuildStage | str | None,
    through_stage: BuildStage | str | None,
) -> tuple[
    NormalizedDocument,
    BuildPlan,
    tuple[PlanNode, ...],
    BuildStage,
    BuildStage,
    BackendProfile,
    BuildTarget,
]:
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
    )
    full_plan = plan_audiobook(
        manifest,
        document,
        target_name=target_name,
        profile_override=profile_override,
        chapter_selector=chapter_selector,
    )
    target = manifest.target(target_name)
    plan, nodes, start, end = _select_stages(
        full_plan,
        from_stage=from_stage or target.from_stage,
        through_stage=through_stage or target.through_stage,
    )
    return (
        document,
        plan,
        nodes,
        start,
        end,
        manifest.profile(plan.profile),
        target,
    )


@asynccontextmanager
async def _journal_hosted_usage(
    service: TextToSpeechService,
    journal: RunJournal,
    prior_events: tuple[dict[str, object], ...],
) -> AsyncIterator[None]:
    if not isinstance(service, HostedUsageJournalingService):
        yield
        return
    previous = _latest_journaled_usage(prior_events)
    if previous is not None:
        await service.restore_usage(previous)

    async def record(
        snapshot: HostedUsageSnapshot,
        submitted_characters: int,
    ) -> None:
        usage = _usage_dict(snapshot)
        if usage is None:
            raise BuildError("Hosted usage recorder received no snapshot")
        usage["submitted_characters_this_attempt"] = submitted_characters
        journal.append("usage_reserved", usage=usage)

    service.set_usage_recorder(record)
    try:
        yield
    finally:
        service.set_usage_recorder(None)


def _latest_journaled_usage(
    events: tuple[dict[str, object], ...],
) -> HostedUsageSnapshot | None:
    for event in reversed(events):
        value = event.get("usage")
        if not isinstance(value, dict):
            continue
        usage = cast(dict[str, object], value)
        logical_items = _journal_counter(usage.get("logical_items"), "logical_items")
        provider_attempts = _journal_counter(
            usage.get("provider_attempts"), "provider_attempts"
        )
        submitted_characters = _journal_counter(
            usage.get("submitted_characters"), "submitted_characters"
        )
        ambiguous_attempts = _journal_counter(
            usage.get("ambiguous_attempts", 0), "ambiguous_attempts"
        )
        spend_value = usage.get("estimated_spend")
        try:
            estimated_spend = (
                Decimal(str(spend_value)) if spend_value is not None else None
            )
        except InvalidOperation as error:
            raise BuildError("Invalid hosted spending value in run journal") from error
        currency_value = usage.get("currency")
        return HostedUsageSnapshot(
            logical_items=logical_items,
            provider_attempts=provider_attempts,
            submitted_characters=submitted_characters,
            estimated_spend=estimated_spend,
            currency=(
                CurrencyCode(currency_value)
                if isinstance(currency_value, str)
                else None
            ),
            ambiguous_attempts=ambiguous_attempts,
        )
    return None


def _journal_counter(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BuildError(f"Invalid hosted usage counter {name!r} in run journal")
    return value


def preflight_audiobook_build(
    manifest: AudiobookManifest,
    *,
    target_name: str = "default",
    profile_override: str | None = None,
    chapter_selector: str | None = None,
    price_per_character: Decimal | None = None,
    from_stage: BuildStage | str | None = None,
    through_stage: BuildStage | str | None = None,
) -> BuildPreflight:
    """Resolve pending work and storage without loading a model or using a network."""

    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
    )
    full_plan = plan_audiobook(
        manifest,
        document,
        target_name=target_name,
        profile_override=profile_override,
        chapter_selector=chapter_selector,
    )
    target = manifest.target(target_name)
    plan, execution_nodes, resolved_from, resolved_through = _select_stages(
        full_plan,
        from_stage=from_stage or target.from_stage,
        through_stage=through_stage or target.through_stage,
    )
    profile = manifest.profile(plan.profile)
    return _preflight_for_plan(
        manifest,
        plan,
        profile,
        target,
        price_per_character=(
            price_per_character
            if price_per_character is not None
            else target.price_per_character
        ),
        execution_nodes=execution_nodes,
        from_stage=resolved_from,
        through_stage=resolved_through,
    )


async def _execute_hosted_plan(
    plan: BuildPlan,
    *,
    execution_nodes: tuple[PlanNode, ...],
    document: NormalizedDocument,
    manifest: AudiobookManifest,
    run_id: str,
    service: TextToSpeechService,
    journal: RunJournal,
    concurrency: int,
    media_semaphore: asyncio.Semaphore,
    progress: _BuildProgressTracker,
) -> _ExecutionOutcome:
    chapter_ids = tuple(dict.fromkeys(node.chapter_id for node in execution_nodes))
    chains = tuple(
        tuple(node for node in execution_nodes if node.chapter_id == chapter_id)
        for chapter_id in chapter_ids
    )
    worker_count = min(concurrency, len(chains))
    queue: asyncio.Queue[tuple[int, tuple[PlanNode, ...]] | None] = asyncio.Queue(
        maxsize=max(1, 2 * worker_count)
    )
    stop = asyncio.Event()
    outcomes: dict[int, _ExecutionOutcome] = {}

    context = _ExecutionContext(
        plan=plan,
        document=document,
        manifest=manifest,
        run_id=run_id,
        service=service,
        journal=journal,
        media_semaphore=media_semaphore,
        progress=progress,
    )
    async with asyncio.TaskGroup() as group:
        group.create_task(_produce_chains(queue, chains, worker_count))
        for _ in range(worker_count):
            group.create_task(_hosted_worker(queue, outcomes, stop, context))
    ordered = tuple(outcomes[index] for index in sorted(outcomes))
    return _ExecutionOutcome(
        records=tuple(record for outcome in ordered for record in outcome.records),
        reused=tuple(node for outcome in ordered for node in outcome.reused),
        failed=tuple(node for outcome in ordered for node in outcome.failed),
    )


async def _produce_chains(
    queue: asyncio.Queue[tuple[int, tuple[PlanNode, ...]] | None],
    chains: tuple[tuple[PlanNode, ...], ...],
    worker_count: int,
) -> None:
    for index, chain in enumerate(chains):
        await queue.put((index, chain))
    for _ in range(worker_count):
        await queue.put(None)


async def _hosted_worker(
    queue: asyncio.Queue[tuple[int, tuple[PlanNode, ...]] | None],
    outcomes: dict[int, _ExecutionOutcome],
    stop: asyncio.Event,
    context: _ExecutionContext,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            return
        index, chain = item
        if stop.is_set():
            for node in chain:
                context.journal.append(
                    "node_not_run",
                    node_id=node.id,
                    fingerprint=node.fingerprint,
                    error="build stopped after another chapter failed",
                )
                context.progress.emit(
                    BuildProgressEvent.NOT_RUN,
                    node,
                    terminal=True,
                    error="build stopped after another chapter failed",
                )
            outcomes[index] = _ExecutionOutcome((), (), ())
            continue
        outcome = await _execute_node_chain(
            chain,
            plan=context.plan,
            document=context.document,
            manifest=context.manifest,
            run_id=context.run_id,
            service=context.service,
            journal=context.journal,
            media_semaphore=context.media_semaphore,
            progress=context.progress,
        )
        outcomes[index] = outcome
        if outcome.failed:
            stop.set()


async def _execute_node_chain(
    nodes: tuple[PlanNode, ...],
    *,
    plan: BuildPlan,
    document: NormalizedDocument,
    manifest: AudiobookManifest,
    run_id: str,
    service: TextToSpeechService,
    journal: RunJournal,
    media_semaphore: asyncio.Semaphore,
    progress: _BuildProgressTracker,
) -> _ExecutionOutcome:
    target = manifest.target(plan.target)
    records: list[ArtifactRecord] = []
    reused: list[str] = []
    for node in nodes:
        existing = _matching_artifact(
            node,
            target=plan.target,
            artifact_root=target.output_root,
        )
        if existing is not None:
            expected_voices = _node_logical_voice_names(plan, node, manifest)
            if existing.logical_voices != expected_voices:
                existing = replace(existing, logical_voices=expected_voices)
                write_artifact_record(existing, root=target.output_root)
            records.append(existing)
            reused.append(node.id)
            journal.append(
                "node_reused",
                node_id=node.id,
                fingerprint=node.fingerprint,
                artifact_path=existing.path.relative_to(manifest.root).as_posix(),
                artifact_sha256=existing.sha256,
            )
            progress.emit(
                BuildProgressEvent.REUSED,
                node,
                terminal=True,
                reused=True,
            )
            continue
        journal.append("node_started", node_id=node.id, fingerprint=node.fingerprint)
        progress.emit(BuildProgressEvent.STARTED, node)
        try:
            record = await _execute_node(
                node,
                plan=plan,
                document=document,
                manifest=manifest,
                run_id=run_id,
                service=service,
                media_semaphore=media_semaphore,
            )
        except Exception as error:  # noqa: BLE001 - node boundary records safe failure
            journal.append(
                "node_failed",
                node_id=node.id,
                fingerprint=node.fingerprint,
                error=_safe_error(error),
            )
            progress.emit(
                BuildProgressEvent.FAILED,
                node,
                terminal=True,
                error=_safe_error(error),
            )
            return _ExecutionOutcome(tuple(records), tuple(reused), (node.id,))
        records.append(record)
        journal.append(
            "node_completed",
            node_id=node.id,
            fingerprint=node.fingerprint,
            artifact_path=record.path.relative_to(manifest.root).as_posix(),
            artifact_sha256=record.sha256,
        )
        progress.emit(BuildProgressEvent.COMPLETED, node, terminal=True)
    return _ExecutionOutcome(tuple(records), tuple(reused), ())


async def audition_audiobook(
    manifest: AudiobookManifest,
    *,
    profiles: tuple[str, ...],
    target_name: str = "default",
    text: str | None = None,
    chapter_selector: str | None = None,
    api_key: str | None = None,
    matrix: tuple[str, ...] = (),
) -> tuple[ArtifactRecord, ...]:
    """Render bounded comparison samples for one or more backend profiles."""
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
    )
    sample = (
        _audition_text(document, chapter_selector)
        if text is None
        else apply_pronunciations(text, manifest.pronunciations)
    )
    run_id = new_run_id()
    target = manifest.target(target_name)
    root = target.output_root / "auditions" / run_id
    records: list[ArtifactRecord] = []
    comparisons: list[dict[str, object]] = []
    variants = _audition_variants(manifest, profiles, matrix)
    services: dict[tuple[str, str | None], TextToSpeechService] = {}
    async with AsyncExitStack() as stack:
        for variant_name, profile, overrides in variants:
            (
                voice,
                sample_rate,
                project,
                use_hd,
                reference_audio,
                chatterbox,
            ) = _resolved_speech(profile, manifest)
            destination = root / f"{variant_name}.wav"
            service_key = (profile.backend.casefold(), _profile_device(profile))
            service = services.get(service_key)
            if service is None:
                service = await stack.enter_async_context(
                    open_speech_backend(
                        profile.backend,
                        api_key=api_key,
                        isolated_local=_is_local_backend(profile.backend),
                        device=_profile_device(profile),
                        local_worker_timeout_seconds=_profile_worker_timeout(profile),
                        local_threads_per_process=_profile_threads(profile),
                        local_worker_log_path=root / "logs" / "local-worker.log",
                    )
                )
                services[service_key] = service
            artifact = await service.synthesize_to_file(
                SpeechSynthesisRequest(
                    text=_preview_sample(sample, profile),
                    voice=voice,
                    backend=profile.backend,
                    profile=profile.name,
                    output_format=AudioFormat.WAV,
                    sample_rate=sample_rate,
                    project=project,
                    use_hd=use_hd,
                    reference_audio=reference_audio,
                    chatterbox=chatterbox,
                ),
                destination,
            )
            record = ArtifactRecord(
                schema_version=1,
                id=f"audition:{run_id}:{variant_name}",
                kind=ArtifactKind.AUDITION,
                path=artifact.path,
                sha256=artifact.sha256,
                size=artifact.bytes_written,
                fingerprint=artifact.sha256,
                target=target_name,
                run_id=run_id,
                protected=False,
                media_type="audio/wav",
                logical_voice=profile.voice,
                reference_audio_sha256=(
                    sha256_file(reference_audio) if reference_audio else None
                ),
                reference_rights_basis=manifest.voice(profile.voice).rights_basis,
                watermark_disclosure=_watermark_disclosure(profile),
            )
            write_artifact_record(record, root=target.output_root)
            records.append(record)
            comparisons.append(
                {
                    "variant": variant_name,
                    "profile": profile.name,
                    "backend": profile.backend,
                    "logical_voice": profile.voice,
                    "resolved_settings": {
                        "sample_rate": sample_rate,
                        "project": project,
                        "use_hd": use_hd,
                        "chatterbox": (
                            {
                                "cfg_weight": chatterbox.cfg_weight,
                                "exaggeration": chatterbox.exaggeration,
                                "seed": chatterbox.seed,
                            }
                            if chatterbox is not None
                            else None
                        ),
                        "matrix_overrides": overrides,
                        "reference_audio_sha256": (
                            sha256_file(reference_audio) if reference_audio else None
                        ),
                    },
                    "duration_seconds": artifact.duration_seconds,
                    "attempts": artifact.attempts,
                    "artifact": record.to_dict(root=manifest.root),
                    "persist_as": _audition_persist_snippet(
                        profile,
                        variant_name,
                        target_name,
                        overrides,
                    ),
                }
            )
    report_path = root / "audition.json"
    atomic_write_json(
        report_path,
        {
            **runtime_metadata("audiobook-audition"),
            "created_at": datetime.now(UTC).isoformat(),
            "matrix": list(matrix),
            "profiles": [record.to_dict(root=manifest.root) for record in records],
            "comparisons": comparisons,
        },
    )
    write_artifact_record(
        ArtifactRecord(
            schema_version=1,
            id=f"audition:{run_id}:report",
            kind=ArtifactKind.REPORT,
            path=report_path.resolve(),
            sha256=sha256_file(report_path),
            size=report_path.stat().st_size,
            fingerprint=sha256_file(report_path),
            target=target_name,
            run_id=run_id,
            protected=False,
            dependencies=tuple(record.id for record in records),
            media_type="application/json",
        ),
        root=target.output_root,
    )
    return tuple(records)


def _audition_variants(
    manifest: AudiobookManifest,
    profiles: tuple[str, ...],
    matrix: tuple[str, ...],
) -> tuple[tuple[str, BackendProfile, dict[str, object]], ...]:
    if not profiles:
        raise ValidationError("At least one audition profile is required")
    axes = _matrix_axes(matrix)
    combinations = (
        tuple(itertools.product(*(values for _, values in axes))) if axes else ((),)
    )
    variants: list[tuple[str, BackendProfile, dict[str, object]]] = []
    names: set[str] = set()
    for profile_name in profiles:
        base = manifest.profile(profile_name)
        for combination in combinations:
            profile = base
            overrides: dict[str, object] = {}
            for (key, _), raw in zip(axes, combination, strict=True):
                profile, value = _override_audition_profile(profile, key, raw)
                overrides[key] = value
            suffix = "--".join(
                f"{_safe_name(key)}-{_safe_name(str(value))}"
                for key, value in sorted(overrides.items())
            )
            name = _safe_name(f"{profile_name}--{suffix}" if suffix else profile_name)
            if name in names:
                raise ValidationError(f"Audition matrix creates duplicate name: {name}")
            names.add(name)
            variants.append((name, profile, overrides))
    return tuple(variants)


def _matrix_axes(
    entries: tuple[str, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    axes: dict[str, tuple[str, ...]] = {}
    for entry in entries:
        key, separator, raw_values = entry.partition("=")
        key = key.strip()
        values = tuple(value.strip() for value in raw_values.split(","))
        if not separator or not key or not values or any(not value for value in values):
            raise ValidationError(
                f"Invalid audition matrix {entry!r}; use KEY=VALUE[,VALUE...]"
            )
        if key in axes:
            raise ValidationError(f"Audition matrix axis {key!r} is repeated")
        axes[key] = values
    return tuple(sorted(axes.items()))


def _override_audition_profile(
    profile: BackendProfile,
    key: str,
    raw: str,
) -> tuple[BackendProfile, object]:
    options = profile.options
    if isinstance(options, FakeOptions) and key == "sample_rate":
        value = _positive_matrix_int(raw, key)
        return replace(profile, options=replace(options, sample_rate=value)), value
    if isinstance(options, ResembleOptions):
        if key == "sample_rate":
            value = _positive_matrix_int(raw, key)
            return replace(profile, options=replace(options, sample_rate=value)), value
        if key == "use_hd":
            value = _matrix_bool(raw, key)
            return replace(profile, options=replace(options, use_hd=value)), value
    if isinstance(options, ChatterboxOptions):
        if key in {"cfg_weight", "exaggeration"}:
            value = _finite_matrix_float(raw, key)
            updated = (
                replace(options, cfg_weight=value)
                if key == "cfg_weight"
                else replace(options, exaggeration=value)
            )
            return replace(profile, options=updated), value
        if key == "seed":
            value = _matrix_int(raw, key)
            return replace(profile, options=replace(options, seed=value)), value
        if key == "device":
            value = raw.casefold()
            if value not in {"auto", "cpu", "cuda", "mps"}:
                raise ValidationError(
                    "Audition matrix device must be auto, cpu, cuda, or mps"
                )
            return replace(profile, options=replace(options, device=value)), value
    raise ValidationError(
        f"Audition matrix setting {key!r} is not supported by "
        f"profile {profile.name!r} ({profile.backend})"
    )


def _positive_matrix_int(raw: str, key: str) -> int:
    value = _matrix_int(raw, key)
    if value <= 0:
        raise ValidationError(f"Audition matrix {key} must be positive")
    return value


def _matrix_int(raw: str, key: str) -> int:
    try:
        return int(raw)
    except ValueError as error:
        raise ValidationError(f"Audition matrix {key} must be an integer") from error


def _finite_matrix_float(raw: str, key: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise ValidationError(f"Audition matrix {key} must be a number") from error
    if not math.isfinite(value):
        raise ValidationError(f"Audition matrix {key} must be finite")
    return value


def _matrix_bool(raw: str, key: str) -> bool:
    normalized = raw.casefold()
    if normalized not in {"true", "false"}:
        raise ValidationError(f"Audition matrix {key} must be true or false")
    return normalized == "true"


def _audition_persist_snippet(
    profile: BackendProfile,
    variant_name: str,
    target_name: str,
    overrides: Mapping[str, object],
) -> str:
    if not overrides:
        return f'targets.{target_name}.profile = "{profile.name}"'
    values = "\n".join(
        f"{key} = {_toml_scalar(value)}" for key, value in sorted(overrides.items())
    )
    return (
        f"# Copy profiles.{profile.name} to profiles.{variant_name}, then apply:\n"
        f"{values}\n"
        f'targets.{target_name}.profile = "{variant_name}"'
    )


def _toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return str(value).casefold()
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


async def preview_audiobook(
    manifest: AudiobookManifest,
    *,
    target_name: str = "default",
    profile_override: str | None = None,
    text: str | None = None,
    chapter_selector: str | None = None,
    api_key: str | None = None,
) -> ArtifactRecord:
    """Render a bounded chapter/selection preview outside the production DAG."""

    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
    )
    sample = (
        _audition_text(document, chapter_selector)
        if text is None
        else apply_pronunciations(text, manifest.pronunciations)
    )
    if not sample.strip():
        raise BuildError("Preview text must not be empty")
    target = manifest.target(target_name)
    profile = manifest.profile(profile_override or target.profile)
    (
        voice,
        sample_rate,
        project,
        use_hd,
        reference_audio,
        chatterbox,
    ) = _resolved_speech(profile, manifest)
    run_id = new_run_id()
    root = target.output_root / "previews" / run_id
    destination = root / f"{profile.name}.wav"
    async with open_speech_backend(
        profile.backend,
        api_key=api_key,
        isolated_local=_is_local_backend(profile.backend),
        device=_profile_device(profile),
        local_worker_timeout_seconds=_profile_worker_timeout(profile),
        local_threads_per_process=_profile_threads(profile),
        local_worker_log_path=root / "logs" / "local-worker.log",
    ) as service:
        artifact = await service.synthesize_to_file(
            SpeechSynthesisRequest(
                text=_preview_sample(sample, profile),
                voice=voice,
                backend=profile.backend,
                profile=profile.name,
                output_format=AudioFormat.WAV,
                sample_rate=sample_rate,
                project=project,
                use_hd=use_hd,
                reference_audio=reference_audio,
                chatterbox=chatterbox,
            ),
            destination,
        )
    record = ArtifactRecord(
        schema_version=1,
        id=f"preview:{run_id}:{profile.name}",
        kind=ArtifactKind.PREVIEW,
        path=artifact.path,
        sha256=artifact.sha256,
        size=artifact.bytes_written,
        fingerprint=artifact.sha256,
        target=target_name,
        run_id=run_id,
        protected=False,
        media_type="audio/wav",
        logical_voice=profile.voice,
        reference_audio_sha256=(
            sha256_file(reference_audio) if reference_audio else None
        ),
        reference_rights_basis=manifest.voice(profile.voice).rights_basis,
        watermark_disclosure=_watermark_disclosure(profile),
    )
    write_artifact_record(record, root=target.output_root)
    return record


def check_release(
    manifest: AudiobookManifest,
    *,
    target_name: str = "default",
    write_manifest: bool = False,
) -> ReleaseCheck:
    """Validate release completeness and optionally publish immutable evidence."""
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
    )
    target = manifest.target(target_name)
    plan = plan_audiobook(manifest, document, target_name=target_name)
    expected = {node.output.resolve(): node for node in plan.nodes}
    release_voice = manifest.voice(manifest.profile(plan.profile).voice)
    release_voices = _plan_logical_voices(plan, manifest)
    masters = tuple(
        target.output_root / "mastered" / f"{chapter.id}.wav"
        for chapter in document.chapters
    )
    mp3s = tuple(
        target.output_root / "release" / "mp3" / f"{chapter.id}.mp3"
        for chapter in document.chapters
    )
    issues = [
        f"logical voice {voice.name!r} has release-ineligible "
        f"rights_basis {voice.rights_basis!r}"
        for voice in release_voices
        if voice.reference_audio is not None
        and voice.rights_basis
        not in {"owned", "licensed", "consented", "public_domain"}
    ]
    for path in (*masters, *mp3s):
        issues.extend(
            _release_artifact_issues(
                path,
                expected.get(path.resolve()),
                target_name=target_name,
                expected_voices=_node_logical_voice_names(
                    plan,
                    expected.get(path.resolve()),
                    manifest,
                ),
                quality=_quality_policy(target),
            )
        )
    for node in plan.nodes:
        if node.stage is BuildStage.VERIFY_MANUSCRIPT:
            issues.extend(
                _release_verification_issues(
                    node,
                    target_name=target_name,
                    expected_voices=_node_logical_voice_names(plan, node, manifest),
                )
            )
    release_manifest: Path | None = None
    if not issues and write_manifest:
        release_id = new_run_id()
        masters, mp3s, release_manifest = _snapshot_release(
            manifest,
            document=document,
            plan=plan,
            target=target,
            target_name=target_name,
            release_id=release_id,
            masters=masters,
            mp3s=mp3s,
            release_voice=release_voice,
            release_voices=release_voices,
        )
    return ReleaseCheck(
        complete=not issues,
        issues=tuple(issues),
        master_wavs=masters,
        delivery_mp3s=mp3s,
        release_manifest=release_manifest,
    )


def _plan_logical_voices(
    plan: BuildPlan,
    manifest: AudiobookManifest,
) -> tuple[LogicalVoice, ...]:
    names: list[str] = []
    for node in plan.nodes:
        if node.stage is not BuildStage.SYNTHESIZE:
            continue
        for name in _node_logical_voice_names(plan, node, manifest):
            if name not in names:
                names.append(name)
    if not names:
        names.append(manifest.profile(plan.profile).voice)
    return tuple(manifest.voice(name) for name in names)


def _node_logical_voice_names(
    plan: BuildPlan,
    node: PlanNode | None,
    manifest: AudiobookManifest,
) -> tuple[str, ...]:
    if node is None:
        return ()
    synthesis = next(
        (
            candidate
            for candidate in plan.nodes
            if candidate.chapter_id == node.chapter_id
            and candidate.stage is BuildStage.SYNTHESIZE
        ),
        None,
    )
    names: list[str] = []
    for route in synthesis.chunk_routes if synthesis is not None else ():
        if route.profile is None:
            continue
        name = manifest.profile(route.profile).voice
        if name not in names:
            names.append(name)
    if not names:
        names.append(manifest.profile(plan.profile).voice)
    return tuple(names)


def diff_releases(left: Path, right: Path) -> ReleaseDiff:
    """Compare two immutable release manifests by artifact and metadata identity."""
    left_document = _load_release_document(left)
    right_document = _load_release_document(right)
    left_artifacts = _release_artifact_digests(left_document)
    right_artifacts = _release_artifact_digests(right_document)
    left_keys = set(left_artifacts)
    right_keys = set(right_artifacts)
    metadata_keys = (
        "book",
        "chapters",
        "voices",
        "target",
        "profile",
        "document_sha256",
        "plan_fingerprint",
        "runtime_fingerprints",
        "runtime_versions",
    )
    return ReleaseDiff(
        left_release_id=str(left_document.get("release_id", left.parent.name)),
        right_release_id=str(right_document.get("release_id", right.parent.name)),
        added_artifacts=tuple(sorted(right_keys - left_keys)),
        removed_artifacts=tuple(sorted(left_keys - right_keys)),
        changed_artifacts=tuple(
            sorted(
                key
                for key in left_keys & right_keys
                if left_artifacts[key] != right_artifacts[key]
            )
        ),
        metadata_changes=tuple(
            key
            for key in metadata_keys
            if left_document.get(key) != right_document.get(key)
        ),
    )


def _load_release_document(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Cannot read release manifest {path}: {error}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("release_id"), str):
        raise ArtifactError(f"Invalid release manifest: {path}")
    return cast(dict[str, object], raw)


def _release_artifact_digests(document: Mapping[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in ("master_wavs", "delivery_mp3s"):
        values = document.get(group, [])
        if not isinstance(values, list):
            raise ArtifactError(f"Release manifest {group} must be an array")
        for value in values:
            if not isinstance(value, dict):
                raise ArtifactError(f"Release manifest {group} entry is invalid")
            path = value.get("path")
            digest = value.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str):
                raise ArtifactError(f"Release manifest {group} entry is incomplete")
            result[f"{group}/{Path(path).name}"] = digest
    return result


def _release_node_fingerprints(document: Mapping[str, object]) -> dict[str, str]:
    values = document.get("nodes")
    if not isinstance(values, list):
        raise ArtifactError(
            "Release manifest predates node fingerprints and cannot be used "
            "with --since"
        )
    result: dict[str, str] = {}
    for value in values:
        if not isinstance(value, dict):
            raise ArtifactError("Release manifest nodes entry is invalid")
        node_id = value.get("id")
        fingerprint = value.get("fingerprint")
        if not isinstance(node_id, str) or not isinstance(fingerprint, str):
            raise ArtifactError("Release manifest node fingerprint is incomplete")
        result[node_id] = fingerprint
    return result


def _snapshot_release(
    manifest: AudiobookManifest,
    *,
    document: NormalizedDocument,
    plan: BuildPlan,
    target: BuildTarget,
    target_name: str,
    release_id: str,
    masters: tuple[Path, ...],
    mp3s: tuple[Path, ...],
    release_voice: LogicalVoice,
    release_voices: tuple[LogicalVoice, ...],
) -> tuple[tuple[Path, ...], tuple[Path, ...], Path]:
    release_root = target.output_root / "release" / release_id
    _validate_release_storage(target, (*masters, *mp3s))
    snapshot_masters = tuple(release_root / "wav" / source.name for source in masters)
    snapshot_mp3s = tuple(release_root / "mp3" / source.name for source in mp3s)
    release_manifest = release_root / "release.json"
    profile = manifest.profile(plan.profile)
    try:
        for source, destination in zip(masters, snapshot_masters, strict=True):
            _copy_release_artifact(
                source,
                destination,
                target=target,
                target_name=target_name,
                release_id=release_id,
                media_type="audio/wav",
                dependency=f"{source.stem}:master",
                release_voice=release_voice,
                release_voices=release_voices,
                profile=profile,
            )
        for source, destination in zip(mp3s, snapshot_mp3s, strict=True):
            _copy_release_artifact(
                source,
                destination,
                target=target,
                target_name=target_name,
                release_id=release_id,
                media_type="audio/mpeg",
                dependency=f"{source.stem}:encode_mp3",
                release_voice=release_voice,
                release_voices=release_voices,
                profile=profile,
            )
        atomic_write_json(
            release_manifest,
            {
                **runtime_metadata("audiobook-release"),
                "release_id": release_id,
                "target": target_name,
                "profile": plan.profile,
                "plan_fingerprint": plan.fingerprint,
                "document_sha256": plan.document_sha256,
                "runtime_fingerprints": {
                    "backend": backend_fingerprint(profile),
                    "media_tools": media_tool_fingerprint(),
                },
                "runtime_versions": {
                    "backend": backend_versions(profile),
                    "media_tools": media_tool_versions(),
                },
                "book": {
                    "title": manifest.book.title,
                    "subtitle": manifest.book.subtitle,
                    "author": manifest.book.author,
                    "narrator": manifest.book.narrator,
                    "language": manifest.book.language,
                    "copyright": manifest.book.copyright,
                    "publisher": manifest.book.publisher,
                    "genre": manifest.book.genre,
                    "series": manifest.book.series,
                    "series_position": manifest.book.series_position,
                    "isbn": manifest.book.isbn,
                    "publication_date": manifest.book.publication_date,
                    "cover_sha256": (
                        sha256_file(manifest.book.cover)
                        if manifest.book.cover is not None
                        else None
                    ),
                },
                "chapters": [
                    {
                        "id": chapter.id,
                        "title": chapter.title,
                        "order": chapter.order,
                    }
                    for chapter in document.chapters
                ],
                "nodes": [
                    {
                        "id": node.id,
                        "stage": node.stage.value,
                        "chapter_id": node.chapter_id,
                        "fingerprint": node.fingerprint,
                    }
                    for node in plan.nodes
                ],
                "voices": [
                    {
                        "name": voice.name,
                        "rights_basis": voice.rights_basis,
                        "reference_audio_sha256": (
                            sha256_file(voice.reference_audio)
                            if voice.reference_audio
                            else None
                        ),
                    }
                    for voice in manifest.voices
                ],
                "master_wavs": [
                    {
                        "path": path.relative_to(target.output_root).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for path in snapshot_masters
                ],
                "delivery_mp3s": [
                    {
                        "path": path.relative_to(target.output_root).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for path in snapshot_mp3s
                ],
            },
            overwrite=False,
        )
        release_digest = sha256_file(release_manifest)
        write_artifact_record(
            ArtifactRecord(
                schema_version=1,
                id=f"{target_name}:release:{release_id}:manifest",
                kind=ArtifactKind.RELEASE,
                path=release_manifest.resolve(),
                sha256=release_digest,
                size=release_manifest.stat().st_size,
                fingerprint=release_digest,
                target=target_name,
                run_id=release_id,
                protected=True,
                dependencies=(
                    *(f"{path.stem}:master" for path in masters),
                    *(f"{path.stem}:encode_mp3" for path in mp3s),
                ),
                media_type="application/json",
                logical_voice=release_voice.name,
                logical_voices=tuple(voice.name for voice in release_voices),
                reference_audio_sha256=(
                    sha256_file(release_voice.reference_audio)
                    if release_voice.reference_audio
                    else None
                ),
                reference_rights_basis=release_voice.rights_basis,
                watermark_disclosure=_watermark_disclosure(profile),
            ),
            root=target.output_root,
        )
    except Exception:
        if release_root.exists():
            shutil.rmtree(release_root)
        raise
    return snapshot_masters, snapshot_mp3s, release_manifest


def _validate_release_storage(
    target: BuildTarget,
    sources: tuple[Path, ...],
) -> None:
    required = sum(path.stat().st_size for path in sources)
    available = shutil.disk_usage(target.output_root).free
    if required > available:
        raise ArtifactError(
            f"Release snapshot requires {required} bytes but only {available} are free"
        )
    if target.storage_budget_bytes is None:
        return
    existing = sum(
        path.stat().st_size for path in target.output_root.rglob("*") if path.is_file()
    )
    if existing + required > target.storage_budget_bytes:
        raise ArtifactError(
            "Release snapshot would exceed target storage_budget_bytes "
            f"({target.storage_budget_bytes})"
        )


def _copy_release_artifact(
    source: Path,
    destination: Path,
    *,
    target: BuildTarget,
    target_name: str,
    release_id: str,
    media_type: str,
    dependency: str,
    release_voice: LogicalVoice,
    release_voices: tuple[LogicalVoice, ...],
    profile: BackendProfile,
) -> None:
    copy_audio(source, destination)
    digest = sha256_file(destination)
    write_artifact_record(
        ArtifactRecord(
            schema_version=1,
            id=f"{target_name}:release:{release_id}:{destination.parent.name}:"
            f"{destination.stem}",
            kind=ArtifactKind.RELEASE,
            path=destination.resolve(),
            sha256=digest,
            size=destination.stat().st_size,
            fingerprint=digest,
            target=target_name,
            run_id=release_id,
            protected=True,
            dependencies=(dependency,),
            media_type=media_type,
            logical_voice=release_voice.name,
            logical_voices=tuple(voice.name for voice in release_voices),
            reference_audio_sha256=(
                sha256_file(release_voice.reference_audio)
                if release_voice.reference_audio
                else None
            ),
            reference_rights_basis=release_voice.rights_basis,
            watermark_disclosure=_watermark_disclosure(profile),
        ),
        root=target.output_root,
    )


def _release_artifact_issues(
    path: Path,
    planned: PlanNode | None,
    *,
    target_name: str,
    expected_voices: tuple[str, ...],
    quality: AudioQualityPolicy,
) -> tuple[str, ...]:
    metadata = path.with_suffix(f"{path.suffix}.artifact.json")
    if not path.is_file() or not metadata.is_file():
        return (f"missing release artifact or manifest: {path}",)
    try:
        raw = json.loads(metadata.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return (f"invalid artifact manifest: {metadata}",)

    issues: list[str] = []
    if planned is None or raw.get("fingerprint") != planned.fingerprint:
        issues.append(f"stale or unplanned release artifact: {path}")
    if raw.get("target") != target_name:
        issues.append(f"release artifact target differs from {target_name}: {path}")
    actual_voices = tuple(
        str(item)
        for item in raw.get("logical_voices", [raw.get("logical_voice")])
        if item is not None
    )
    if actual_voices != expected_voices:
        issues.append(f"mixed logical voice in release artifact: {path}")
    if sha256_file(path) != raw.get("sha256"):
        issues.append(f"digest mismatch: {path}")
        return tuple(issues)
    try:
        inspection = inspect_audio(path, quality=quality)
    except (ArtifactError, BackendUnavailableError) as error:
        issues.append(f"cannot inspect release artifact {path}: {error}")
        return tuple(issues)
    expected_format = "wav" if path.suffix.casefold() == ".wav" else "mp3"
    if not inspection.valid or expected_format not in inspection.format_name:
        issues.append(f"release artifact has invalid {expected_format} media: {path}")
    return tuple(issues)


def _release_verification_issues(
    node: PlanNode,
    *,
    target_name: str,
    expected_voices: tuple[str, ...],
) -> tuple[str, ...]:
    path = node.output
    metadata = path.with_suffix(f"{path.suffix}.artifact.json")
    if not path.is_file() or not metadata.is_file():
        return (f"missing manuscript verification artifact: {path}",)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        raw = json.loads(metadata.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return (f"invalid manuscript verification artifact: {path}",)
    issues: list[str] = []
    if report.get("accepted") is not True:
        issues.append(f"manuscript verification did not pass: {path}")
    if raw.get("fingerprint") != node.fingerprint:
        issues.append(f"stale manuscript verification artifact: {path}")
    if raw.get("target") != target_name:
        issues.append(
            f"verification artifact target differs from {target_name}: {path}"
        )
    actual_voices = tuple(
        str(item)
        for item in raw.get("logical_voices", [raw.get("logical_voice")])
        if item is not None
    )
    if actual_voices != expected_voices:
        issues.append(f"mixed logical voice in verification artifact: {path}")
    if sha256_file(path) != raw.get("sha256"):
        issues.append(f"verification digest mismatch: {path}")
    return tuple(issues)


def assemble_release(
    manifest: AudiobookManifest, *, target_name: str = "default"
) -> Path:
    """Assemble the configured M4B output for a completed build target."""
    check = check_release(manifest, target_name=target_name)
    if not check.complete:
        raise BuildError(
            "Cannot assemble incomplete release: " + "; ".join(check.issues)
        )
    target = manifest.target(target_name)
    if not target.m4b:
        raise BuildError("M4B assembly is not enabled for this target")
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
    )
    plan = plan_audiobook(manifest, document, target_name=target_name)
    release_voices = _plan_logical_voices(plan, manifest)
    assembly_id = new_run_id()
    destination = (
        target.output_root
        / "release"
        / assembly_id
        / f"{_safe_name(manifest.book.title)}.m4b"
    )
    assemble_m4b(
        check.master_wavs,
        destination,
        title=manifest.book.title,
        author=manifest.book.author,
        narrator=manifest.book.narrator,
        subtitle=manifest.book.subtitle,
        genre=manifest.book.genre,
        publisher=manifest.book.publisher,
        copyright=manifest.book.copyright,
        language=manifest.book.language,
        date=manifest.book.publication_date,
        series=manifest.book.series,
        series_position=manifest.book.series_position,
        cover=manifest.book.cover,
        chapter_titles=tuple(chapter.title for chapter in document.chapters),
        bitrate=target.m4b_bitrate,
    )
    record = ArtifactRecord(
        schema_version=1,
        id=f"{target_name}:release:{assembly_id}:m4b",
        kind=ArtifactKind.RELEASE,
        path=destination.resolve(),
        sha256=sha256_file(destination),
        size=destination.stat().st_size,
        fingerprint=_assembly_fingerprint(check.master_wavs, manifest, target),
        target=target_name,
        run_id=assembly_id,
        protected=True,
        dependencies=tuple(f"{path.stem}:master" for path in check.master_wavs),
        media_type="audio/mp4",
        logical_voice=manifest.profile(plan.profile).voice,
        logical_voices=tuple(voice.name for voice in release_voices),
        reference_rights_basis=manifest.voice(
            manifest.profile(plan.profile).voice
        ).rights_basis,
        watermark_disclosure=_watermark_disclosure(manifest.profile(plan.profile)),
    )
    write_artifact_record(record, root=target.output_root)
    return destination


async def _execute_node(
    node: PlanNode,
    *,
    plan: BuildPlan,
    document: NormalizedDocument,
    manifest: AudiobookManifest,
    run_id: str,
    service: TextToSpeechService,
    media_semaphore: asyncio.Semaphore,
) -> ArtifactRecord:
    target = manifest.target(plan.target)
    profile = manifest.profile(plan.profile)
    logical_voice = manifest.voice(profile.voice)
    logical_voices = _node_logical_voice_names(plan, node, manifest)
    if node.stage is BuildStage.SYNTHESIZE:
        await _synthesize_node(node, profile, target, manifest, service)
        kind = ArtifactKind.RAW
        protected = False
        media_type = "audio/wav"
    elif node.stage is BuildStage.MASTER:
        source = _dependency_output(plan, node, BuildStage.SYNTHESIZE)
        async with media_semaphore:
            if target.mastering:
                await asyncio.to_thread(
                    master_wav,
                    source,
                    node.output,
                    sample_rate=target.wav_sample_rate,
                    overwrite=True,
                )
            else:
                await asyncio.to_thread(
                    copy_audio,
                    source,
                    node.output,
                    overwrite=True,
                )
        kind = ArtifactKind.MASTER
        protected = False
        media_type = "audio/wav"
    elif node.stage is BuildStage.VERIFY_MANUSCRIPT:
        source = _dependency_output(plan, node, BuildStage.MASTER)
        chapter = _chapter(document, node.chapter_id)
        expected_text = " ".join(
            segment.text
            for segment in chapter.segments
            if isinstance(segment, SpeechSegment)
        )
        aligner = open_local_aligner(
            manifest.short_utterances.alignment_backend,
            model=manifest.short_utterances.alignment_model,
            revision=manifest.short_utterances.alignment_revision,
            timeout_seconds=manifest.short_utterances.alignment_timeout_seconds,
            prompted_timing=False,
            decode_consensus=manifest.short_utterances.decode_consensus,
            prompt_sensitivity=False,
            maximum_consensus_timing_delta_ms=(
                manifest.short_utterances.maximum_consensus_timing_delta_ms
            ),
            hallucination_silence_threshold=(
                manifest.short_utterances.hallucination_silence_threshold
            ),
        )
        report = await verify_manuscript(
            source,
            chapter.source_path,
            expected_text,
            language=manifest.book.language,
            model=manifest.short_utterances.alignment_model,
            revision=manifest.short_utterances.alignment_revision,
            aligner=aligner,
            cache_root=(
                manifest.whisper_qa.cache_directory
                if manifest.whisper_qa.cache_enabled
                else None
            ),
            token_aliases=_chapter_token_aliases(manifest),
        )
        atomic_write_json(node.output, report.to_dict())
        if not report.accepted:
            raise ArtifactError(
                f"Chapter manuscript verification rejected {node.chapter_id}; "
                f"review {node.output}"
            )
        kind = ArtifactKind.REPORT
        protected = False
        media_type = "application/json"
    elif node.stage is BuildStage.ENCODE_MP3:
        source = _dependency_output(plan, node, BuildStage.MASTER)
        chapter = _chapter(document, node.chapter_id)
        async with media_semaphore:
            await asyncio.to_thread(
                encode_mp3,
                source,
                node.output,
                bitrate=target.mp3_bitrate,
                title=chapter.title,
                album=manifest.book.title,
                artist=manifest.book.author or manifest.book.narrator,
                album_artist=manifest.book.author,
                composer=manifest.book.narrator,
                genre=manifest.book.genre,
                publisher=manifest.book.publisher,
                copyright=manifest.book.copyright,
                language=manifest.book.language,
                date=manifest.book.publication_date,
                cover=manifest.book.cover,
                track=chapter.order,
                overwrite=True,
            )
        kind = ArtifactKind.DELIVERY
        protected = False
        media_type = "audio/mpeg"
    elif node.stage is BuildStage.INSPECT:
        async with media_semaphore:
            await asyncio.to_thread(
                _write_inspection_report,
                node,
                plan,
                target,
            )
        kind = ArtifactKind.REPORT
        protected = False
        media_type = "application/json"
    else:
        raise BuildError(f"Unsupported build stage: {node.stage}")
    record = ArtifactRecord(
        schema_version=1,
        id=node.id,
        kind=kind,
        path=node.output.resolve(),
        sha256=sha256_file(node.output),
        size=node.output.stat().st_size,
        fingerprint=node.fingerprint,
        target=plan.target,
        run_id=run_id,
        protected=protected,
        dependencies=node.dependencies,
        media_type=media_type,
        logical_voice=profile.voice,
        logical_voices=logical_voices,
        reference_audio_sha256=(
            sha256_file(logical_voice.reference_audio)
            if logical_voice.reference_audio
            else None
        ),
        reference_rights_basis=logical_voice.rights_basis,
        watermark_disclosure=_watermark_disclosure(profile),
    )
    write_artifact_record(record, root=target.output_root)
    return record


async def _synthesize_node(
    node: PlanNode,
    profile: BackendProfile,
    target: BuildTarget,
    manifest: AudiobookManifest,
    service: TextToSpeechService,
) -> None:
    primary_sample_rate = _resolved_speech(profile, manifest)[1]
    chunk_paths = [
        node.output.with_name(f".{node.output.stem}.chunk-{index:04d}.wav")
        for index in range(1, len(node.chunks) + 1)
    ]
    try:
        for path in chunk_paths:
            path.unlink(missing_ok=True)
        aligner: WindowSpeechAligner | None = (
            open_local_aligner(
                manifest.short_utterances.alignment_backend,
                model=manifest.short_utterances.alignment_model,
                revision=manifest.short_utterances.alignment_revision,
                timeout_seconds=manifest.short_utterances.alignment_timeout_seconds,
                prompted_timing=manifest.short_utterances.prompted_timing,
                decode_consensus=manifest.short_utterances.decode_consensus,
                prompt_sensitivity=manifest.short_utterances.prompt_sensitivity,
                maximum_consensus_timing_delta_ms=(
                    manifest.short_utterances.maximum_consensus_timing_delta_ms
                ),
                hallucination_silence_threshold=(
                    manifest.short_utterances.hallucination_silence_threshold
                ),
            )
            if _uses_context_extraction(node, manifest)
            else None
        )
        if aligner is not None and manifest.whisper_qa.cache_enabled:
            aligner = CachedWhisperAligner(
                aligner,
                manifest.whisper_qa.cache_directory,
            )
        phoneme_aligner: PhonemeAligner | None = (
            open_phoneme_aligner(
                manifest.whisper_qa.phoneme_backend,
                model=manifest.whisper_qa.phoneme_model,
                revision=manifest.whisper_qa.phoneme_revision,
                timeout_seconds=manifest.whisper_qa.phoneme_timeout_seconds,
            )
            if aligner is not None and manifest.whisper_qa.phoneme_alignment
            else None
        )
        pending, short_pending = _prepare_synthesis_chunks(
            node,
            chunk_paths,
            default_profile=profile,
            target=target,
            manifest=manifest,
            workspace=manifest.root,
            aligner_fingerprint=(
                ":".join(
                    (
                        aligner.fingerprint,
                        phoneme_aligner.fingerprint,
                    )
                )
                if aligner is not None and phoneme_aligner is not None
                else (aligner.fingerprint if aligner else None)
            ),
        )
        try:
            await _render_pending_chunks(pending, service, workspace=manifest.root)
            if short_pending:
                if aligner is None:
                    raise BuildError("Short-utterance alignment was not initialized")
                await _render_short_utterances(
                    short_pending,
                    service=service,
                    aligner=aligner,
                    phoneme_aligner=phoneme_aligner,
                    manifest=manifest,
                )
        except Exception as error:
            raise _chunk_failure(node, chunk_paths, manifest.root, error) from error
        wav_params = _normalize_synthesis_chunks(
            node,
            chunk_paths,
            preferred_sample_rate=_synthesis_sample_rate(profile, primary_sample_rate),
        )
        _materialize_pause_chunks(node, chunk_paths, wav_params)
        join_parts = tuple(
            WavJoinPart(
                path,
                boundary,
                explicit_pause=_PAUSE.fullmatch(chunk) is not None,
            )
            for path, chunk, boundary in zip(
                chunk_paths,
                node.chunks,
                _node_boundaries(node),
                strict=True,
            )
        )
        concatenate_wavs(join_parts, node.output, overwrite=True)
        if aligner is not None and manifest.short_utterances.automatic_join_inspection:
            await _inspect_synthesis_joins(
                node,
                join_parts=join_parts,
                aligner=aligner,
                manifest=manifest,
                target=target,
            )
    finally:
        for path in chunk_paths:
            path.unlink(missing_ok=True)


def _prepare_synthesis_chunks(
    node: PlanNode,
    chunk_paths: list[Path],
    *,
    default_profile: BackendProfile,
    target: BuildTarget,
    manifest: AudiobookManifest,
    workspace: Path,
    aligner_fingerprint: str | None,
) -> tuple[
    list[tuple[SpeechSynthesisRequest, Path, str]],
    list[_PendingShortUtterance],
]:
    pending: list[tuple[SpeechSynthesisRequest, Path, str]] = []
    short_pending: list[_PendingShortUtterance] = []
    routes = _node_routes(node, default_profile)
    markers = _node_short_utterances(node)
    for index, (chunk, chunk_path, route, marker) in enumerate(
        zip(node.chunks, chunk_paths, routes, markers, strict=True), start=1
    ):
        pause = _PAUSE.fullmatch(chunk)
        if pause:
            continue
        profile = _profile_from_route(manifest, route, default_profile)
        (
            voice,
            sample_rate,
            project,
            use_hd,
            reference_audio,
            chatterbox,
        ) = _resolved_speech(profile, manifest)
        request = _new_speech_request(
            chunk,
            profile=profile,
            voice=voice,
            sample_rate=sample_rate,
            project=project,
            use_hd=use_hd,
            reference_audio=reference_audio,
            chatterbox=_chunk_chatterbox(
                chatterbox,
                chapter_id=node.chapter_id,
                chunk_index=index,
                text=chunk,
            ),
        )
        if (
            marker is not None
            and manifest.short_utterances.strategy
            is ShortUtteranceStrategy.CONTEXT_EXTRACT
        ):
            if aligner_fingerprint is None:
                raise BuildError("Short-utterance aligner fingerprint is missing")
            recipes = carrier_recipes(
                chunk,
                manifest.short_utterances,
                seed_material=f"{node.chapter_id}:{index}",
                previous_context=_neighbor_context(node, index - 1, route, step=-1),
                next_context=_neighbor_context(node, index - 1, route, step=1),
            )
            fingerprint = _short_utterance_fingerprint(
                request,
                recipes=recipes,
                policy_fingerprint=manifest.short_utterances.fingerprint,
                aligner_fingerprint=aligner_fingerprint,
            )
            cached = _cached_chunk(workspace, fingerprint)
            if cached is not None:
                _materialize_cached_chunk(cached, chunk_path)
            else:
                short_pending.append(
                    _PendingShortUtterance(
                        request=request,
                        destination=chunk_path,
                        fingerprint=fingerprint,
                        chunk_index=index,
                        recipes=recipes,
                        qa_directory=_short_utterance_qa_directory(
                            target,
                            node,
                            index=index,
                            fingerprint=fingerprint,
                            word_count=marker.word_count,
                            manifest=manifest,
                        ),
                    )
                )
            continue
        fingerprint = _speech_request_fingerprint(request)
        cached = _cached_chunk(workspace, fingerprint)
        if cached is not None:
            _materialize_cached_chunk(cached, chunk_path)
        else:
            pending.append((request, chunk_path, fingerprint))
    return pending, short_pending


def _node_routes(
    node: PlanNode,
    default_profile: BackendProfile,
) -> tuple[ChunkRoute, ...]:
    if node.chunk_routes:
        return node.chunk_routes
    return tuple(ChunkRoute("narrator", default_profile.name) for _ in node.chunks)


def _profile_from_route(
    manifest: AudiobookManifest,
    route: ChunkRoute,
    default_profile: BackendProfile,
) -> BackendProfile:
    profile = (
        manifest.profile(route.profile)
        if route.profile is not None
        else default_profile
    )
    options = profile.options
    if not isinstance(options, ChatterboxOptions) or route.profile is None:
        return profile
    return replace(
        profile,
        options=replace(
            options,
            cfg_weight=route.cfg_weight,
            exaggeration=route.exaggeration,
            seed=route.seed,
        ),
    )


def _chunk_failure(
    node: PlanNode,
    chunk_paths: list[Path],
    workspace: Path,
    error: Exception,
) -> BuildError:
    failed_index = next(
        (
            index
            for index, (chunk, path) in enumerate(
                zip(node.chunks, chunk_paths, strict=True)
            )
            if _PAUSE.fullmatch(chunk) is None and not _is_readable_wav(path)
        ),
        0,
    )
    source = node.chunk_sources[failed_index]
    source_path = (
        source.path.relative_to(workspace).as_posix()
        if source.path.is_relative_to(workspace)
        else str(source.path)
    )
    return BuildError(
        f"{node.chapter_id} chunk {failed_index + 1} "
        f"({source_path}:{source.start_line}-{source.end_line}): {error}"
    )


def _new_speech_request(
    text: str,
    *,
    profile: BackendProfile,
    voice: str,
    sample_rate: int | None,
    project: str | None,
    use_hd: bool,
    reference_audio: Path | None,
    chatterbox: ChatterboxSynthesisOptions | None,
) -> SpeechSynthesisRequest:
    return SpeechSynthesisRequest(
        text=text,
        voice=voice,
        backend=profile.backend,
        profile=profile.name,
        output_format=AudioFormat.WAV,
        sample_rate=sample_rate,
        project=project,
        use_hd=use_hd,
        reference_audio=reference_audio,
        chatterbox=chatterbox,
    )


def _chunk_chatterbox(
    options: ChatterboxSynthesisOptions | None,
    *,
    chapter_id: str,
    chunk_index: int,
    text: str,
) -> ChatterboxSynthesisOptions | None:
    if options is None:
        return None
    material = json.dumps(
        {
            "base_seed": options.seed if options.seed is not None else 0,
            "chapter_id": chapter_id,
            "chunk_index": chunk_index,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
        sort_keys=True,
    ).encode()
    seed = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
    return replace(options, seed=seed)


async def _render_pending_chunks(
    pending: list[tuple[SpeechSynthesisRequest, Path, str]],
    service: TextToSpeechService,
    *,
    workspace: Path,
) -> None:
    try:
        if pending and isinstance(service, BatchTextToSpeechService):
            await service.synthesize_many_to_files(
                tuple((request, destination) for request, destination, _ in pending),
                overwrite=True,
            )
        else:
            for request, destination, _ in pending:
                await service.synthesize_to_file(
                    request,
                    destination,
                    overwrite=True,
                )
    finally:
        for _, destination, fingerprint in pending:
            if _is_readable_wav(destination):
                _store_cached_chunk(workspace, fingerprint, destination)


async def _render_short_utterances(
    pending: list[_PendingShortUtterance],
    *,
    service: TextToSpeechService,
    aligner: SpeechAligner,
    phoneme_aligner: PhonemeAligner | None,
    manifest: AudiobookManifest,
) -> None:
    for item in pending:
        await synthesize_short_utterance(
            service=service,
            aligner=aligner,
            request=item.request,
            destination=item.destination,
            recipes=item.recipes,
            policy=manifest.short_utterances,
            language=manifest.book.language,
            qa_directory=item.qa_directory,
            phoneme_aligner=phoneme_aligner,
            phoneme_language=manifest.whisper_qa.phoneme_language,
            minimum_phoneme_confidence=(manifest.whisper_qa.minimum_phoneme_confidence),
        )
        if _is_readable_wav(item.destination):
            _store_cached_chunk(
                manifest.root,
                item.fingerprint,
                item.destination,
            )


def _uses_context_extraction(node: PlanNode, manifest: AudiobookManifest) -> bool:
    return (
        manifest.short_utterances.strategy is ShortUtteranceStrategy.CONTEXT_EXTRACT
        and any(marker is not None for marker in _node_short_utterances(node))
    )


async def _inspect_synthesis_joins(
    node: PlanNode,
    *,
    join_parts: tuple[WavJoinPart, ...],
    aligner: WindowSpeechAligner,
    manifest: AudiobookManifest,
    target: BuildTarget,
) -> None:
    boundaries = wav_join_boundaries(join_parts)
    if not boundaries:
        return
    report = await inspect_joins(
        node.output,
        tuple(
            specification
            for boundary in boundaries
            for specification in _boundary_join_specifications(boundary)
        ),
        language=manifest.book.language,
        model=manifest.short_utterances.alignment_model,
        revision=manifest.short_utterances.alignment_revision,
        window_seconds=manifest.short_utterances.join_inspection_window_seconds,
        coalesce_gap_seconds=manifest.whisper_qa.join_coalesce_gap_ms / 1_000,
        aligner=aligner,
    )
    destination = target.output_root / "qa" / "joins" / f"{node.chapter_id}.joins.json"
    atomic_write_json(destination, report.to_dict())
    if not report.accepted:
        failed = sum(not item.accepted for item in report.joins)
        raise BuildError(
            f"Automatic join inspection rejected {failed} join(s) in "
            f"{node.chapter_id}; review {destination}"
        )


def _boundary_join_specifications(
    boundary: WavJoinBoundary,
) -> tuple[JoinSpecification, ...]:
    kind = (
        "explicit_pause" if boundary.adjacent_to_explicit_pause else boundary.boundary
    )
    previous = JoinSpecification(
        at_seconds=boundary.previous_end_seconds,
        boundary=f"{kind}:previous_end",
    )
    if boundary.at_seconds == boundary.previous_end_seconds:
        return (previous,)
    return (
        previous,
        JoinSpecification(
            at_seconds=boundary.at_seconds,
            boundary=f"{kind}:next_start",
        ),
    )


def _node_short_utterances(
    node: PlanNode,
) -> tuple[ShortUtteranceMarker | None, ...]:
    if node.chunk_short_utterances:
        return node.chunk_short_utterances
    return tuple(None for _ in node.chunks)


def _neighbor_context(
    node: PlanNode,
    chunk_index: int,
    route: ChunkRoute,
    *,
    step: int,
) -> str | None:
    routes = node.chunk_routes
    candidate = chunk_index + step
    while 0 <= candidate < len(node.chunks):
        text = node.chunks[candidate]
        if _PAUSE.fullmatch(text) is not None:
            return None
        candidate_route = routes[candidate] if routes else route
        if (
            candidate_route.speaker == route.speaker
            and candidate_route.profile == route.profile
        ):
            return text
        candidate += step
    return None


def _short_utterance_fingerprint(
    request: SpeechSynthesisRequest,
    *,
    recipes: tuple[CarrierRecipe, ...],
    policy_fingerprint: str,
    aligner_fingerprint: str,
) -> str:
    payload = {
        "version": 1,
        "request": _speech_request_fingerprint(request),
        "policy": policy_fingerprint,
        "aligner": aligner_fingerprint,
        "recipes": [
            {
                "index": recipe.candidate_index,
                "text_sha256": hashlib.sha256(recipe.text.encode()).hexdigest(),
                "template": recipe.template_id,
                "position": recipe.position.value,
                "natural": recipe.natural,
                "seed": recipe.seed,
            }
            for recipe in recipes
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _short_utterance_qa_directory(
    target: BuildTarget,
    node: PlanNode,
    *,
    index: int,
    fingerprint: str,
    word_count: int,
    manifest: AudiobookManifest,
) -> Path | None:
    policy = manifest.short_utterances
    requires_review = policy.require_review_for_one_word and word_count == 1
    if not (
        policy.keep_candidates
        or policy.failure is ShortUtteranceFailure.REVIEW
        or requires_review
    ):
        return None
    return (
        target.output_root
        / "qa"
        / "short-utterances"
        / node.chapter_id
        / f"chunk-{index:04d}-{fingerprint[:12]}"
    )


def _write_inspection_report(
    node: PlanNode,
    plan: BuildPlan,
    target: BuildTarget,
) -> None:
    master = _dependency_output(plan, node, BuildStage.MASTER)
    mp3 = _dependency_output(plan, node, BuildStage.ENCODE_MP3)
    policy = _quality_policy(target)
    inspected = (
        inspect_audio(master, quality=policy),
        inspect_audio(mp3, quality=policy),
    )
    invalid = tuple(
        issue
        for inspection in inspected
        if not inspection.valid
        for issue in inspection.issues
    )
    if invalid:
        raise ArtifactError(
            f"Audio inspection failed for {node.chapter_id}: {'; '.join(invalid)}"
        )
    atomic_write_json(
        node.output,
        {
            **runtime_metadata("audiobook-chapter-inspection"),
            "chapter_id": node.chapter_id,
            "inspections": [
                inspection.to_dict(root=target.output_root) for inspection in inspected
            ],
        },
    )


def _quality_policy(target: BuildTarget) -> AudioQualityPolicy:
    return AudioQualityPolicy(
        minimum_loudness_lufs=target.quality_min_lufs,
        maximum_loudness_lufs=target.quality_max_lufs,
        maximum_true_peak_dbfs=target.quality_max_true_peak_dbfs,
        maximum_leading_silence_seconds=target.quality_max_leading_silence_seconds,
        maximum_trailing_silence_seconds=target.quality_max_trailing_silence_seconds,
    )


def _matching_artifact(
    node: PlanNode, *, target: str, artifact_root: Path
) -> ArtifactRecord | None:
    metadata = node.output.with_suffix(f"{node.output.suffix}.artifact.json")
    if not metadata.is_file():
        return None
    try:
        raw = json.loads(metadata.read_text(encoding="utf-8"))
        if raw.get("fingerprint") != node.fingerprint or raw.get("target") != target:
            return None
        record = ArtifactRecord(
            schema_version=int(raw["schema_version"]),
            id=str(raw["id"]),
            kind=ArtifactKind(raw["kind"]),
            path=(artifact_root / raw["path"]).resolve(),
            sha256=str(raw["sha256"]),
            size=int(raw["size"]),
            fingerprint=str(raw["fingerprint"]),
            target=str(raw["target"]),
            run_id=str(raw["run_id"]),
            protected=bool(raw["protected"]),
            dependencies=tuple(raw.get("dependencies", [])),
            media_type=raw.get("media_type"),
            logical_voice=raw.get("logical_voice"),
            logical_voices=tuple(raw.get("logical_voices", [])),
            reference_audio_sha256=raw.get("reference_audio_sha256"),
            reference_rights_basis=raw.get("reference_rights_basis"),
            watermark_disclosure=raw.get("watermark_disclosure"),
        )
    except OSError, KeyError, TypeError, ValueError, json.JSONDecodeError:
        return None
    valid, _ = verify_artifact(record)
    return record if valid else None


def _dependency_output(plan: BuildPlan, node: PlanNode, stage: BuildStage) -> Path:
    by_id = {candidate.id: candidate for candidate in plan.nodes}
    pending = list(node.dependencies)
    visited: set[str] = set()
    while pending:
        dependency = pending.pop(0)
        if dependency in visited:
            continue
        visited.add(dependency)
        candidate = by_id.get(dependency)
        if candidate is None:
            continue
        if candidate.stage is stage:
            return candidate.output
        pending.extend(candidate.dependencies)
    raise BuildError(f"{node.id} lacks {stage.value} dependency")


def _resolved_speech(
    profile: BackendProfile,
    manifest: AudiobookManifest,
) -> tuple[
    str,
    int | None,
    str | None,
    bool,
    Path | None,
    ChatterboxSynthesisOptions | None,
]:
    options = profile.options
    if isinstance(options, ResembleOptions):
        return (
            options.voice_uuid,
            options.sample_rate,
            options.project_uuid,
            options.use_hd,
            None,
            None,
        )
    if isinstance(options, FakeOptions):
        return profile.voice, options.sample_rate, None, False, None, None
    if isinstance(options, ChatterboxOptions):
        return (
            profile.voice,
            None,
            None,
            False,
            manifest.voice(profile.voice).reference_audio,
            ChatterboxSynthesisOptions(
                cfg_weight=options.cfg_weight,
                exaggeration=options.exaggeration,
                seed=options.seed,
            ),
        )
    return (
        profile.voice,
        None,
        None,
        False,
        manifest.voice(profile.voice).reference_audio,
        None,
    )


def _profile_device(profile: BackendProfile) -> str | None:
    return (
        profile.options.device
        if isinstance(profile.options, ChatterboxOptions)
        else None
    )


def _is_local_backend(backend: str) -> bool:
    return backend.casefold() in {"local", "chatterbox", "chatterbox-local"}


def _profile_worker_timeout(profile: BackendProfile) -> float:
    return (
        profile.options.worker_timeout_seconds
        if isinstance(profile.options, ChatterboxOptions)
        else 3_600
    )


def _profile_threads(profile: BackendProfile) -> int:
    return (
        profile.options.threads_per_process
        if isinstance(profile.options, ChatterboxOptions)
        else 1
    )


def _synthesis_sample_rate(
    profile: BackendProfile, requested_sample_rate: int | None
) -> int:
    if isinstance(profile.options, ChatterboxOptions):
        return 24_000
    return requested_sample_rate or 16_000


def _watermark_disclosure(profile: BackendProfile) -> str:
    if isinstance(profile.options, ResembleOptions):
        return "Provider watermarking behavior is governed by the Resemble account/API"
    if isinstance(profile.options, ChatterboxOptions):
        return (
            "Yakbox adds no watermark; local Chatterbox embeds its upstream "
            "PerTh watermark"
        )
    return "No Yakbox watermark is added"


def _normalize_synthesis_chunks(
    node: PlanNode,
    chunk_paths: list[Path],
    *,
    preferred_sample_rate: int,
) -> tuple[int, int, int]:
    speech_paths = tuple(
        path
        for chunk, path in zip(node.chunks, chunk_paths, strict=True)
        if _PAUSE.fullmatch(chunk) is None
    )
    if not speech_paths:
        return 1, 2, preferred_sample_rate
    params = tuple(_wav_params(path) for path in speech_paths)
    if all(item == params[0] for item in params[1:]):
        return params[0]
    for path in speech_paths:
        normalized = path.with_name(f"{path.stem}.normalized.wav")
        try:
            master_wav(
                path,
                normalized,
                sample_rate=preferred_sample_rate,
                normalize=False,
                overwrite=True,
            )
            normalized.replace(path)
        finally:
            normalized.unlink(missing_ok=True)
    normalized_params = tuple(_wav_params(path) for path in speech_paths)
    if any(item != normalized_params[0] for item in normalized_params[1:]):
        raise ArtifactError("Could not normalize synthesized WAV chunk formats")
    return normalized_params[0]


def _materialize_pause_chunks(
    node: PlanNode,
    chunk_paths: list[Path],
    params: tuple[int, int, int],
) -> None:
    channels, sample_width, sample_rate = params
    for chunk, path in zip(node.chunks, chunk_paths, strict=True):
        pause = _PAUSE.fullmatch(chunk)
        if pause is not None:
            write_silence(
                path,
                int(pause.group(1)),
                sample_rate=sample_rate,
                channels=channels,
                sample_width=sample_width,
            )


def _wav_params(path: Path) -> tuple[int, int, int]:
    try:
        with wave.open(str(path), "rb") as source:
            if source.getnframes() < 1:
                raise ArtifactError(f"Synthesized WAV contains no frames: {path}")
            return (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
            )
    except (OSError, EOFError, wave.Error) as error:
        raise ArtifactError(
            f"Synthesized chunk is not a readable WAV: {path}"
        ) from error


def _node_boundaries(node: PlanNode) -> tuple[str, ...]:
    if node.chunk_boundaries:
        return node.chunk_boundaries
    return tuple("end" for _ in node.chunks)


def _speech_request_fingerprint(request: SpeechSynthesisRequest) -> str:
    chatterbox = request.chatterbox
    payload = {
        "version": 1,
        "text": request.text,
        "voice": request.voice,
        "backend": request.backend,
        "backend_runtime": backend_runtime_fingerprint(request.backend),
        "output_format": request.output_format.value,
        "sample_rate": request.sample_rate,
        "use_hd": request.use_hd,
        "precision": request.precision,
        "apply_custom_pronunciations": request.apply_custom_pronunciations,
        "project": request.project,
        "reference_audio_sha256": (
            sha256_file(request.reference_audio) if request.reference_audio else None
        ),
        "chatterbox": (
            {
                "cfg_weight": chatterbox.cfg_weight,
                "exaggeration": chatterbox.exaggeration,
                "seed": chatterbox.seed,
            }
            if chatterbox is not None
            else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cached_chunk(workspace: Path, fingerprint: str) -> Path | None:
    audio, metadata = _chunk_cache_paths(workspace, fingerprint)
    if not audio.is_file() or not metadata.is_file():
        return None
    try:
        raw = json.loads(metadata.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != 1
            or raw.get("fingerprint") != fingerprint
            or raw.get("size") != audio.stat().st_size
            or raw.get("sha256") != sha256_file(audio)
            or not _is_readable_wav(audio)
        ):
            return None
    except OSError, json.JSONDecodeError:
        return None
    return audio


def _store_cached_chunk(workspace: Path, fingerprint: str, source: Path) -> None:
    audio, metadata = _chunk_cache_paths(workspace, fingerprint)
    audio.parent.mkdir(parents=True, exist_ok=True)
    copy_audio(source, audio, overwrite=True)
    atomic_write_json(
        metadata,
        {
            "schema_version": 1,
            "kind": "synthesis_chunk",
            "fingerprint": fingerprint,
            "sha256": sha256_file(audio),
            "size": audio.stat().st_size,
        },
    )


def _materialize_cached_chunk(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        copy_audio(source, destination, overwrite=True)


def _chunk_cache_paths(workspace: Path, fingerprint: str) -> tuple[Path, Path]:
    root = workspace / ".yakbox" / "cache" / "synthesis" / fingerprint[:2]
    return root / f"{fingerprint}.wav", root / f"{fingerprint}.json"


def _is_readable_wav(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with wave.open(str(path), "rb") as source:
            return (
                source.getnchannels() > 0
                and source.getsampwidth() > 0
                and source.getframerate() > 0
                and source.getnframes() > 0
            )
    except OSError, EOFError, wave.Error:
        return False


def _chapter(document: NormalizedDocument, chapter_id: str) -> Chapter:
    for chapter in document.chapters:
        if chapter.id == chapter_id:
            return chapter
    raise BuildError(f"Unknown planned chapter: {chapter_id}")


def _audition_text(document: NormalizedDocument, chapter_selector: str | None) -> str:
    for chapter in document.chapters:
        if chapter_selector and (
            chapter_selector not in {chapter.id, str(chapter.order)}
            and chapter_selector.casefold() not in chapter.title.casefold()
        ):
            continue
        return " ".join(
            segment.text
            for segment in chapter.segments
            if isinstance(segment, SpeechSegment)
        )
    raise BuildError("No chapter is available for audition")


def _preview_sample(text: str, profile: BackendProfile) -> str:
    maximum = (
        CHATTERBOX_CHUNK_CHARACTERS
        if isinstance(profile.options, ChatterboxOptions)
        else 1_000
    )
    chunks = chunk_text(text, maximum)
    if not chunks:
        raise BuildError("Preview text must not be empty")
    return chunks[0]


def _preflight_for_plan(
    manifest: AudiobookManifest,
    plan: BuildPlan,
    profile: BackendProfile,
    target: BuildTarget,
    *,
    price_per_character: Decimal | None,
    execution_nodes: tuple[PlanNode, ...] | None = None,
    from_stage: BuildStage = BuildStage.SYNTHESIZE,
    through_stage: BuildStage = BuildStage.INSPECT,
) -> BuildPreflight:
    selected = execution_nodes if execution_nodes is not None else plan.nodes
    reusable = {
        node.id
        for node in selected
        if _matching_artifact(
            node,
            target=plan.target,
            artifact_root=target.output_root,
        )
        is not None
    }
    pending = tuple(node for node in selected if node.id not in reusable)
    pending_chapter_ids = {node.chapter_id for node in pending}
    synthesis_nodes = tuple(
        node for node in pending if node.stage is BuildStage.SYNTHESIZE
    )
    texts = _uncached_synthesis_texts(
        manifest,
        profile,
        synthesis_nodes,
    )
    hosted_work = (
        estimate_hosted_work(texts, price_per_character=price_per_character)
        if profile.backend in {"resemble", "cloud"}
        else None
    )
    chapter_characters = {
        node.chapter_id: sum(
            len(chunk) for chunk in node.chunks if _PAUSE.fullmatch(chunk) is None
        )
        for node in plan.nodes
        if node.stage is BuildStage.SYNTHESIZE
        and node.chapter_id in pending_chapter_ids
    }
    estimated_output_bytes = sum(
        max(
            _STORAGE_BYTES_PER_CHAPTER_FLOOR,
            characters * _STORAGE_BYTES_PER_CHARACTER,
        )
        for characters in chapter_characters.values()
    )
    short_utterance_chunks = (
        sum(
            marker is not None
            for node in synthesis_nodes
            for marker in _node_short_utterances(node)
        )
        if manifest.short_utterances.strategy is ShortUtteranceStrategy.CONTEXT_EXTRACT
        else 0
    )
    storage_path = _nearest_existing_parent(target.output_root)
    return BuildPreflight(
        target=plan.target,
        profile=plan.profile,
        planned_nodes=len(selected),
        reusable_nodes=len(reusable),
        pending_nodes=len(pending),
        pending_chapters=len(pending_chapter_ids),
        estimated_output_bytes=estimated_output_bytes,
        available_bytes=shutil.disk_usage(storage_path).free,
        storage_budget_bytes=target.storage_budget_bytes,
        hosted_work=hosted_work,
        reusable_node_ids=tuple(sorted(reusable)),
        change_summary=_compare_to_previous_success(manifest, plan),
        from_stage=from_stage,
        through_stage=through_stage,
        short_utterance_chunks=short_utterance_chunks,
        maximum_short_utterance_generations=(
            short_utterance_chunks * manifest.short_utterances.candidate_count
        ),
    )


def _uncached_synthesis_texts(
    manifest: AudiobookManifest,
    profile: BackendProfile,
    nodes: tuple[PlanNode, ...],
) -> tuple[str, ...]:
    texts: list[str] = []
    for node in nodes:
        routes = _node_routes(node, profile)
        for index, (chunk, route) in enumerate(
            zip(node.chunks, routes, strict=True), start=1
        ):
            if _PAUSE.fullmatch(chunk) is not None:
                continue
            routed_profile = _profile_from_route(manifest, route, profile)
            (
                voice,
                sample_rate,
                project,
                use_hd,
                reference_audio,
                chatterbox,
            ) = _resolved_speech(routed_profile, manifest)
            request = _new_speech_request(
                chunk,
                profile=routed_profile,
                voice=voice,
                sample_rate=sample_rate,
                project=project,
                use_hd=use_hd,
                reference_audio=reference_audio,
                chatterbox=_chunk_chatterbox(
                    chatterbox,
                    chapter_id=node.chapter_id,
                    chunk_index=index,
                    text=chunk,
                ),
            )
            if (
                _cached_chunk(
                    manifest.root,
                    _speech_request_fingerprint(request),
                )
                is None
            ):
                texts.append(chunk)
    return tuple(texts)


def _select_stages(
    plan: BuildPlan,
    *,
    from_stage: BuildStage | str | None,
    through_stage: BuildStage | str | None,
) -> tuple[BuildPlan, tuple[PlanNode, ...], BuildStage, BuildStage]:
    start = _build_stage(from_stage, BuildStage.SYNTHESIZE)
    end = _build_stage(through_stage, BuildStage.INSPECT)
    start_index = _EXECUTION_STAGES.index(start)
    end_index = _EXECUTION_STAGES.index(end)
    if start_index > end_index:
        raise ValidationError(f"Build stage {start.value!r} comes after {end.value!r}")
    stages = frozenset(_EXECUTION_STAGES[start_index : end_index + 1])
    selected = tuple(node for node in plan.nodes if node.stage in stages)
    if start is BuildStage.SYNTHESIZE and end is BuildStage.INSPECT:
        return plan, selected, start, end
    fingerprint = hashlib.sha256(
        f"{plan.fingerprint}\0{start.value}\0{end.value}".encode()
    ).hexdigest()
    effective = BuildPlan(
        schema_version=plan.schema_version,
        target=plan.target,
        profile=plan.profile,
        document_sha256=plan.document_sha256,
        fingerprint=fingerprint,
        nodes=plan.nodes,
        complete_document=plan.complete_document,
        attribution_findings=plan.attribution_findings,
    )
    return effective, selected, start, end


def _build_stage(
    value: BuildStage | str | None,
    default: BuildStage,
) -> BuildStage:
    if value is None:
        return default
    try:
        stage = value if isinstance(value, BuildStage) else BuildStage(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in _EXECUTION_STAGES)
        raise ValidationError(
            f"Unknown build stage {value!r}; choose {choices}"
        ) from error
    if stage not in _EXECUTION_STAGES:
        raise ValidationError(
            f"Build stage {stage.value!r} cannot be executed directly"
        )
    return stage


def _validate_stage_prerequisites(
    plan: BuildPlan,
    execution_nodes: tuple[PlanNode, ...],
    *,
    target: BuildTarget,
) -> None:
    selected_ids = {node.id for node in execution_nodes}
    planned = {node.id: node for node in plan.nodes}
    missing: list[str] = []
    for node in execution_nodes:
        for dependency in node.dependencies:
            if dependency in selected_ids:
                continue
            prerequisite = planned.get(dependency)
            if (
                prerequisite is None
                or _matching_artifact(
                    prerequisite,
                    target=plan.target,
                    artifact_root=target.output_root,
                )
                is None
            ):
                missing.append(f"{node.id} requires verified {dependency}")
    if missing:
        raise BuildError(
            "Stage-bounded build prerequisites are unavailable: " + "; ".join(missing)
        )


def _nearest_existing_parent(path: Path) -> Path:
    current = path.resolve()
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise BuildError(f"Cannot locate a filesystem for output path: {path}")
        current = parent
    return current


def _validate_storage_preflight(preflight: BuildPreflight) -> None:
    if preflight.estimated_output_bytes > preflight.available_bytes:
        raise BuildError(
            "Insufficient free disk space before synthesis: approximately "
            f"{preflight.estimated_output_bytes} byte(s) are required but "
            f"{preflight.available_bytes} are available"
        )
    if (
        preflight.storage_budget_bytes is not None
        and preflight.estimated_output_bytes > preflight.storage_budget_bytes
    ):
        raise BuildError(
            "Planned output exceeds the target storage budget: approximately "
            f"{preflight.estimated_output_bytes} byte(s) are required but the "
            f"budget is {preflight.storage_budget_bytes}"
        )


def _compare_to_previous_success(
    manifest: AudiobookManifest,
    plan: BuildPlan,
) -> BuildChangeSummary:
    current = {
        node.id: {
            "fingerprint": node.fingerprint,
            "stage": node.stage.value,
        }
        for node in plan.nodes
    }
    previous = _latest_successful_plan(
        manifest.root,
        plan.target,
        complete_document=plan.complete_document,
        current_node_ids=frozenset(current),
    )
    if previous is None:
        added = tuple(sorted(current))
        return BuildChangeSummary(
            previous_run_id=None,
            previous_plan_fingerprint=None,
            added_nodes=added,
            removed_nodes=(),
            changed_nodes=(),
            unchanged_nodes=(),
            reasons=tuple((node, "no previous successful run") for node in added),
        )
    previous_run_id, previous_plan = previous
    prior_nodes = _serialized_plan_nodes(previous_plan)
    current_ids = set(current)
    prior_ids = set(prior_nodes)
    added = tuple(sorted(current_ids - prior_ids))
    removed = tuple(sorted(prior_ids - current_ids))
    shared = current_ids & prior_ids
    changed = tuple(
        sorted(
            node
            for node in shared
            if current[node]["fingerprint"] != prior_nodes[node]["fingerprint"]
        )
    )
    unchanged = tuple(sorted(shared - set(changed)))
    reasons = [(node, "new planned node") for node in added]
    reasons.extend(
        (
            node,
            _change_reason(
                current[node],
                plan,
                previous_plan,
            ),
        )
        for node in changed
    )
    return BuildChangeSummary(
        previous_run_id=previous_run_id,
        previous_plan_fingerprint=_string_value(previous_plan.get("fingerprint")),
        added_nodes=added,
        removed_nodes=removed,
        changed_nodes=changed,
        unchanged_nodes=unchanged,
        reasons=tuple(reasons),
    )


def _latest_successful_plan(
    workspace: Path,
    target: str,
    *,
    complete_document: bool,
    current_node_ids: frozenset[str],
) -> tuple[str, dict[str, object]] | None:
    runs = workspace.resolve() / ".yakbox" / "runs"
    if not runs.exists():
        return None
    directories = sorted(
        (path for path in runs.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for directory in directories:
        summary_path = directory / "run.json"
        plan_path = directory / "plan.json"
        if not summary_path.is_file() or not plan_path.is_file():
            continue
        summary = _load_json_object(summary_path, "prior run")
        if summary.get("schema_version") != 1:
            raise BuildError(f"Unsupported prior run summary: {summary_path}")
        if summary.get("status") != "complete" or summary.get("target") != target:
            continue
        previous_plan = _load_json_object(plan_path, "prior run plan")
        if (
            previous_plan.get("$schema") != schema_uri("audiobook-plan")
            or previous_plan.get("schema_version") != 1
        ):
            raise BuildError(f"Unsupported prior run plan: {plan_path}")
        prior_complete = previous_plan.get("complete_document", True)
        if complete_document and prior_complete is not True:
            continue
        if not complete_document:
            prior_ids = set(_serialized_plan_nodes(previous_plan))
            if not current_node_ids.issubset(prior_ids):
                continue
        return directory.name, previous_plan
    return None


def _serialized_plan_nodes(
    plan: dict[str, object],
) -> dict[str, dict[str, object]]:
    raw_nodes = plan.get("nodes")
    if not isinstance(raw_nodes, list):
        raise BuildError("Prior run plan has no valid node list")
    result: dict[str, dict[str, object]] = {}
    for item in raw_nodes:
        if not isinstance(item, dict):
            raise BuildError("Prior run plan contains an invalid node")
        node_id = item.get("id")
        fingerprint = item.get("fingerprint")
        if not isinstance(node_id, str) or not isinstance(fingerprint, str):
            raise BuildError("Prior run plan contains an invalid node identity")
        result[node_id] = {
            "fingerprint": fingerprint,
            "stage": item.get("stage"),
        }
    return result


def _change_reason(
    node: Mapping[str, object],
    current_plan: BuildPlan,
    previous_plan: dict[str, object],
) -> str:
    if previous_plan.get("profile") != current_plan.profile:
        return "resolved backend profile changed"
    if previous_plan.get("document_sha256") != current_plan.document_sha256:
        return "normalized source or pronunciation input changed"
    return f"{node.get('stage', 'stage')} settings or dependency changed"


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _find_resumable_run(
    workspace: Path,
    target: str,
    plan_fingerprint: str,
) -> Path | None:
    runs = workspace.resolve() / ".yakbox" / "runs"
    if not runs.exists():
        return None
    for directory in sorted(
        (path for path in runs.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    ):
        plan_path = directory / "plan.json"
        if not plan_path.is_file():
            continue
        prior_plan = _load_json_object(plan_path, "prior run plan")
        if (
            prior_plan.get("$schema") != schema_uri("audiobook-plan")
            or prior_plan.get("schema_version") != 1
        ):
            raise BuildError(f"Unsupported prior run plan: {plan_path}")
        if (
            prior_plan.get("target") != target
            or prior_plan.get("fingerprint") != plan_fingerprint
        ):
            continue
        summary = directory / "run.json"
        if not summary.exists():
            return directory
        result = _load_json_object(summary, "prior run")
        if result.get("schema_version") != 1:
            raise BuildError(f"Unsupported prior run summary: {summary}")
        if result.get("status") != "complete":
            return directory
    return None


def _latest_failed_node_ids(
    workspace: Path,
    target: str,
) -> set[str]:
    runs = workspace.resolve() / ".yakbox" / "runs"
    if not runs.exists():
        return set()
    for directory in sorted(
        (path for path in runs.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    ):
        plan_path = directory / "plan.json"
        journal_path = directory / "journal.ndjson"
        if not plan_path.is_file() or not journal_path.is_file():
            continue
        plan = _load_json_object(plan_path, "prior run plan")
        if plan.get("target") != target:
            continue
        failed = _failed_node_ids(journal_path)
        if failed:
            return failed
    return set()


def _failed_node_ids(journal_path: Path) -> set[str]:
    try:
        lines = journal_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BuildError(f"Cannot read run journal {journal_path}: {error}") from error
    failed: set[str] = set()
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            break
        if not isinstance(event, dict) or event.get("event") != "node_failed":
            continue
        node_id = event.get("node_id")
        if isinstance(node_id, str):
            failed.add(node_id)
    return failed


def _load_json_object(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BuildError(f"Cannot inspect {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise BuildError(f"Invalid {description}: {path}")
    return value


def _result_dict(result: BuildResult) -> dict[str, object]:
    return {
        **runtime_metadata("audiobook-run"),
        "run_id": result.run_id,
        "target": result.target,
        "status": result.status,
        "plan_fingerprint": result.plan_fingerprint,
        "artifacts": [str(item.path) for item in result.artifacts],
        "reused_nodes": list(result.reused_nodes),
        "failed_nodes": list(result.failed_nodes),
        "hosted_usage": _usage_dict(result.hosted_usage),
        "preflight": result.preflight.to_dict(),
        "resumed": result.resumed,
    }


def _hosted_budget(
    target: BuildTarget,
    *,
    max_submitted_characters: int | None,
    max_provider_requests: int | None,
    max_estimated_spend: Decimal | None,
    currency: str | None,
    pricing_source: str | None,
    confirm_above_characters: int | None,
    confirm_above_requests: int | None,
) -> HostedUsageBudget:
    resolved_currency = currency or target.currency
    resolved_pricing_source = pricing_source or target.pricing_source
    return HostedUsageBudget(
        max_submitted_characters=(
            max_submitted_characters
            if max_submitted_characters is not None
            else target.max_submitted_characters
        ),
        max_provider_requests=(
            max_provider_requests
            if max_provider_requests is not None
            else target.max_provider_requests
        ),
        max_estimated_spend=(
            max_estimated_spend
            if max_estimated_spend is not None
            else target.max_estimated_spend
        ),
        currency=CurrencyCode(resolved_currency) if resolved_currency else None,
        pricing_source=PricingSourceId(resolved_pricing_source)
        if resolved_pricing_source
        else None,
        confirm_above_characters=(
            confirm_above_characters
            if confirm_above_characters is not None
            else target.confirm_above_characters
        ),
        confirm_above_requests=(
            confirm_above_requests
            if confirm_above_requests is not None
            else target.confirm_above_requests
        ),
    )


def _usage_dict(snapshot: HostedUsageSnapshot | None) -> dict[str, object] | None:
    if snapshot is None:
        return None
    return {
        "logical_items": snapshot.logical_items,
        "provider_attempts": snapshot.provider_attempts,
        "submitted_characters": snapshot.submitted_characters,
        "estimated_spend": (
            str(snapshot.estimated_spend)
            if snapshot.estimated_spend is not None
            else None
        ),
        "currency": str(snapshot.currency) if snapshot.currency is not None else None,
        "ambiguous_attempts": snapshot.ambiguous_attempts,
        "basis": "estimate" if snapshot.estimated_spend is not None else "usage_only",
    }


def _safe_error(error: Exception) -> str:
    return str(error).replace("\n", " ")[:2048]


def _failed_node_detail(journal: RunJournal, node_id: str) -> str | None:
    for event in reversed(journal.events()):
        if event.get("event") == "node_failed" and event.get("node_id") == node_id:
            detail = event.get("error")
            if isinstance(detail, str):
                return detail[:512]
    return None


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w-]+", "-", value.casefold()).strip("-") or "audiobook"


def _chapter_token_aliases(
    manifest: AudiobookManifest,
) -> dict[str, tuple[str, ...]]:
    short_aliases = manifest.short_utterances.alignment_alias_map
    chapter_aliases = manifest.whisper_qa.manuscript_alias_map
    return {
        word: tuple(
            dict.fromkeys(
                (*short_aliases.get(word, ()), *chapter_aliases.get(word, ()))
            )
        )
        for word in sorted(short_aliases.keys() | chapter_aliases.keys())
    }


def _assembly_fingerprint(
    chapters: tuple[Path, ...],
    manifest: AudiobookManifest,
    target: BuildTarget,
) -> str:
    digest = hashlib.sha256(b"audiobook-m4b-v2")
    for chapter in chapters:
        digest.update(bytes.fromhex(sha256_file(chapter)))
    metadata = {
        "book": {
            "title": manifest.book.title,
            "subtitle": manifest.book.subtitle,
            "author": manifest.book.author,
            "narrator": manifest.book.narrator,
            "language": manifest.book.language,
            "copyright": manifest.book.copyright,
            "publisher": manifest.book.publisher,
            "genre": manifest.book.genre,
            "series": manifest.book.series,
            "series_position": manifest.book.series_position,
            "publication_date": manifest.book.publication_date,
            "cover_sha256": (
                sha256_file(manifest.book.cover)
                if manifest.book.cover is not None
                else None
            ),
        },
        "bitrate": target.m4b_bitrate,
    }
    digest.update(json.dumps(metadata, sort_keys=True).encode())
    return digest.hexdigest()
