from __future__ import annotations

import asyncio
import hashlib
import io
import itertools
import json
import math
import re
import shutil
import wave
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_bytes, atomic_write_json, sha256_file
from yakbox.audio import assemble_m4b, encode_mp3, inspect_audio, master_wav
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
from yakbox.audiobook.planner import BuildPlan, BuildStage, PlanNode, plan_audiobook
from yakbox.audiobook.sources import (
    Chapter,
    NormalizedDocument,
    SpeechSegment,
    normalize_sources,
)
from yakbox.contracts import runtime_metadata, schema_uri
from yakbox.errors import (
    ArtifactError,
    BackendUnavailableError,
    BuildError,
    ValidationError,
)
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

_PAUSE = re.compile(r"__YAKBOX_PAUSE_MS=(\d+)__")
_STORAGE_BYTES_PER_CHARACTER = 16_000
_STORAGE_BYTES_PER_CHAPTER_FLOOR = 256 * 1024
_EXECUTION_STAGES = (
    BuildStage.SYNTHESIZE,
    BuildStage.MASTER,
    BuildStage.ENCODE_MP3,
    BuildStage.INSPECT,
)


@dataclass(frozen=True, slots=True)
class BuildChangeSummary:
    previous_run_id: str | None
    previous_plan_fingerprint: str | None
    added_nodes: tuple[str, ...]
    removed_nodes: tuple[str, ...]
    changed_nodes: tuple[str, ...]
    unchanged_nodes: tuple[str, ...]
    reasons: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
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
    from_stage: str = BuildStage.SYNTHESIZE.value
    through_stage: str = BuildStage.INSPECT.value

    @property
    def storage_sufficient(self) -> bool:
        within_free_space = self.estimated_output_bytes <= self.available_bytes
        within_budget = (
            self.storage_budget_bytes is None
            or self.estimated_output_bytes <= self.storage_budget_bytes
        )
        return within_free_space and within_budget

    def to_dict(self) -> dict[str, object]:
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
            "from_stage": self.from_stage,
            "through_stage": self.through_stage,
        }


@dataclass(frozen=True, slots=True)
class BuildResult:
    schema_version: int
    run_id: str
    target: str
    status: str
    plan_fingerprint: str
    artifacts: tuple[ArtifactRecord, ...]
    reused_nodes: tuple[str, ...]
    failed_nodes: tuple[str, ...]
    run_directory: Path
    hosted_usage: HostedUsageSnapshot | None
    preflight: BuildPreflight
    resumed: bool


@dataclass(frozen=True, slots=True)
class ReleaseCheck:
    complete: bool
    issues: tuple[str, ...]
    master_wavs: tuple[Path, ...]
    delivery_mp3s: tuple[Path, ...]
    release_manifest: Path | None = None

    def to_dict(self) -> dict[str, object]:
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
class _ExecutionOutcome:
    records: tuple[ArtifactRecord, ...]
    reused: tuple[str, ...]
    failed: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    plan: BuildPlan
    document: NormalizedDocument
    manifest: AudiobookManifest
    run_id: str
    service: TextToSpeechService
    journal: RunJournal


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
) -> BuildResult:
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
            status="planned",
            plan_fingerprint=plan.fingerprint,
            artifacts=(),
            reused_nodes=(),
            failed_nodes=(),
            run_directory=run_directory,
            hosted_usage=None,
            preflight=preflight,
            resumed=False,
        )
    state_root = manifest.root / ".yakbox"
    artifacts: list[ArtifactRecord] = []
    reused: list[str] = []
    failed: list[str] = []
    hosted_usage: HostedUsageSnapshot | None = None
    with target_lock(state_root, target_name):
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
                    isolated_local=profile.backend
                    in {"local", "chatterbox", "chatterbox-local"},
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
        status = "failed" if failed else "complete"
        journal.append(
            f"run_{status}",
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
        raise BuildError(f"Build failed at {failed[0]}; see {run_directory}")
    return result


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
    plan, nodes, start, end = _select_stages(
        full_plan,
        from_stage=from_stage,
        through_stage=through_stage,
    )
    return (
        document,
        plan,
        nodes,
        start,
        end,
        manifest.profile(plan.profile),
        manifest.target(target_name),
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
        assert usage is not None
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
    plan, execution_nodes, resolved_from, resolved_through = _select_stages(
        full_plan,
        from_stage=from_stage,
        through_stage=through_stage,
    )
    profile = manifest.profile(plan.profile)
    target = manifest.target(target_name)
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
            records.append(existing)
            reused.append(node.id)
            journal.append(
                "node_reused",
                node_id=node.id,
                fingerprint=node.fingerprint,
                artifact_path=str(existing.path.relative_to(manifest.root)),
                artifact_sha256=existing.sha256,
            )
            continue
        journal.append("node_started", node_id=node.id, fingerprint=node.fingerprint)
        try:
            record = await _execute_node(
                node,
                plan=plan,
                document=document,
                manifest=manifest,
                run_id=run_id,
                service=service,
            )
        except Exception as error:
            journal.append(
                "node_failed",
                node_id=node.id,
                fingerprint=node.fingerprint,
                error=_safe_error(error),
            )
            return _ExecutionOutcome(tuple(records), tuple(reused), (node.id,))
        records.append(record)
        journal.append(
            "node_completed",
            node_id=node.id,
            fingerprint=node.fingerprint,
            artifact_path=str(record.path.relative_to(manifest.root)),
            artifact_sha256=record.sha256,
        )
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
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
    )
    sample = text or _audition_text(document, chapter_selector)
    run_id = new_run_id()
    target = manifest.target(target_name)
    root = target.output_root / "auditions" / run_id
    records: list[ArtifactRecord] = []
    comparisons: list[dict[str, object]] = []
    variants = _audition_variants(manifest, profiles, matrix)
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
        async with open_speech_backend(
            profile.backend,
            api_key=api_key,
            device=_profile_device(profile),
        ) as service:
            artifact = await service.synthesize_to_file(
                SpeechSynthesisRequest(
                    text=sample[: min(1_000, len(sample))],
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
    sample = text or _audition_text(document, chapter_selector)
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
        device=_profile_device(profile),
    ) as service:
        artifact = await service.synthesize_to_file(
            SpeechSynthesisRequest(
                text=sample[: min(1_000, len(sample))],
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
    document = normalize_sources(
        manifest.sources,
        pronunciations=manifest.pronunciations,
        max_pause_ms=manifest.max_pause_ms,
    )
    target = manifest.target(target_name)
    plan = plan_audiobook(manifest, document, target_name=target_name)
    expected = {node.output.resolve(): node for node in plan.nodes}
    release_voice = manifest.voice(manifest.profile(target.profile).voice)
    masters = tuple(
        target.output_root / "mastered" / f"{chapter.id}.wav"
        for chapter in document.chapters
    )
    mp3s = tuple(
        target.output_root / "release" / "mp3" / f"{chapter.id}.mp3"
        for chapter in document.chapters
    )
    issues: list[str] = []
    if release_voice.reference_audio is not None and release_voice.rights_basis not in {
        "owned",
        "licensed",
        "consented",
        "public_domain",
    }:
        issues.append(
            f"logical voice {release_voice.name!r} has release-ineligible "
            f"rights_basis {release_voice.rights_basis!r}"
        )
    for path in (*masters, *mp3s):
        issues.extend(
            _release_artifact_issues(
                path,
                expected.get(path.resolve()),
                target_name=target_name,
                release_voice=release_voice,
            )
        )
    release_manifest: Path | None = None
    if not issues and write_manifest:
        release_id = new_run_id()
        masters, mp3s, release_manifest = _snapshot_release(
            manifest,
            target=target,
            target_name=target_name,
            release_id=release_id,
            masters=masters,
            mp3s=mp3s,
            release_voice=release_voice,
        )
    return ReleaseCheck(
        complete=not issues,
        issues=tuple(issues),
        master_wavs=masters,
        delivery_mp3s=mp3s,
        release_manifest=release_manifest,
    )


def _snapshot_release(
    manifest: AudiobookManifest,
    *,
    target: BuildTarget,
    target_name: str,
    release_id: str,
    masters: tuple[Path, ...],
    mp3s: tuple[Path, ...],
    release_voice: LogicalVoice,
) -> tuple[tuple[Path, ...], tuple[Path, ...], Path]:
    release_root = target.output_root / "release" / release_id
    _validate_release_storage(target, (*masters, *mp3s))
    snapshot_masters = tuple(release_root / "wav" / source.name for source in masters)
    snapshot_mp3s = tuple(release_root / "mp3" / source.name for source in mp3s)
    release_manifest = release_root / "release.json"
    profile = manifest.profile(target.profile)
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
                profile=profile,
            )
        atomic_write_json(
            release_manifest,
            {
                **runtime_metadata("audiobook-release"),
                "release_id": release_id,
                "book": {
                    "title": manifest.book.title,
                    "author": manifest.book.author,
                    "narrator": manifest.book.narrator,
                },
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
                        "path": str(path.relative_to(target.output_root)),
                        "sha256": sha256_file(path),
                    }
                    for path in snapshot_masters
                ],
                "delivery_mp3s": [
                    {
                        "path": str(path.relative_to(target.output_root)),
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
    release_voice: LogicalVoice,
) -> tuple[str, ...]:
    metadata = path.with_suffix(f"{path.suffix}.artifact.json")
    if not path.is_file() or not metadata.is_file():
        return (f"missing release artifact or manifest: {path}",)
    try:
        raw = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (f"invalid artifact manifest: {metadata}",)

    issues: list[str] = []
    if planned is None or raw.get("fingerprint") != planned.fingerprint:
        issues.append(f"stale or unplanned release artifact: {path}")
    if raw.get("target") != target_name:
        issues.append(f"release artifact target differs from {target_name}: {path}")
    if raw.get("logical_voice") != release_voice.name:
        issues.append(f"mixed logical voice in release artifact: {path}")
    if sha256_file(path) != raw.get("sha256"):
        issues.append(f"digest mismatch: {path}")
        return tuple(issues)
    try:
        inspection = inspect_audio(path)
    except (ArtifactError, BackendUnavailableError) as error:
        issues.append(f"cannot inspect release artifact {path}: {error}")
        return tuple(issues)
    expected_format = "wav" if path.suffix.casefold() == ".wav" else "mp3"
    if not inspection.valid or expected_format not in inspection.format_name:
        issues.append(f"release artifact has invalid {expected_format} media: {path}")
    return tuple(issues)


def assemble_release(
    manifest: AudiobookManifest, *, target_name: str = "default"
) -> Path:
    check = check_release(manifest, target_name=target_name)
    if not check.complete:
        raise BuildError(
            "Cannot assemble incomplete release: " + "; ".join(check.issues)
        )
    target = manifest.target(target_name)
    if not target.m4b:
        raise BuildError("M4B assembly is not enabled for this target")
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
    )
    record = ArtifactRecord(
        schema_version=1,
        id=f"{target_name}:release:{assembly_id}:m4b",
        kind=ArtifactKind.RELEASE,
        path=destination.resolve(),
        sha256=sha256_file(destination),
        size=destination.stat().st_size,
        fingerprint=_assembly_fingerprint(check.master_wavs),
        target=target_name,
        run_id=assembly_id,
        protected=True,
        dependencies=tuple(f"{path.stem}:master" for path in check.master_wavs),
        media_type="audio/mp4",
        logical_voice=manifest.profile(target.profile).voice,
        reference_rights_basis=manifest.voice(
            manifest.profile(target.profile).voice
        ).rights_basis,
        watermark_disclosure=_watermark_disclosure(manifest.profile(target.profile)),
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
) -> ArtifactRecord:
    target = manifest.target(plan.target)
    profile = manifest.profile(plan.profile)
    logical_voice = manifest.voice(profile.voice)
    if node.stage is BuildStage.SYNTHESIZE:
        await _synthesize_node(node, profile, manifest, service)
        kind = ArtifactKind.RAW
        protected = False
        media_type = "audio/wav"
    elif node.stage is BuildStage.MASTER:
        source = _dependency_output(plan, node, BuildStage.SYNTHESIZE)
        if target.mastering:
            master_wav(
                source,
                node.output,
                sample_rate=target.wav_sample_rate,
                overwrite=True,
            )
        else:
            copy_audio(source, node.output, overwrite=True)
        kind = ArtifactKind.MASTER
        protected = False
        media_type = "audio/wav"
    elif node.stage is BuildStage.ENCODE_MP3:
        source = _dependency_output(plan, node, BuildStage.MASTER)
        chapter = _chapter(document, node.chapter_id)
        encode_mp3(
            source,
            node.output,
            bitrate=target.mp3_bitrate,
            title=chapter.title,
            album=manifest.book.title,
            artist=manifest.book.author or manifest.book.narrator,
            track=chapter.order,
            overwrite=True,
        )
        kind = ArtifactKind.DELIVERY
        protected = False
        media_type = "audio/mpeg"
    elif node.stage is BuildStage.INSPECT:
        _write_inspection_report(node, plan)
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
    manifest: AudiobookManifest,
    service: TextToSpeechService,
) -> None:
    (
        voice,
        sample_rate,
        project,
        use_hd,
        reference_audio,
        chatterbox,
    ) = _resolved_speech(profile, manifest)
    chunk_paths = [
        node.output.with_name(f".{node.output.stem}.chunk-{index:04d}.wav")
        for index in range(1, len(node.chunks) + 1)
    ]
    try:
        for path in chunk_paths:
            path.unlink(missing_ok=True)
        pending: list[tuple[SpeechSynthesisRequest, Path]] = []
        for chunk, chunk_path in zip(node.chunks, chunk_paths, strict=True):
            pause = _PAUSE.fullmatch(chunk)
            if pause:
                _write_silence(
                    chunk_path,
                    int(pause.group(1)),
                    sample_rate=sample_rate or 16_000,
                )
            else:
                pending.append(
                    (
                        SpeechSynthesisRequest(
                            text=chunk,
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
                        chunk_path,
                    )
                )
        if pending and isinstance(service, BatchTextToSpeechService):
            await service.synthesize_many_to_files(tuple(pending), overwrite=True)
        else:
            for request, destination in pending:
                await service.synthesize_to_file(
                    request,
                    destination,
                    overwrite=True,
                )
        _concatenate_wav(tuple(chunk_paths), node.output)
    finally:
        for path in chunk_paths:
            path.unlink(missing_ok=True)


def _write_inspection_report(node: PlanNode, plan: BuildPlan) -> None:
    master = _dependency_output(plan, node, BuildStage.MASTER)
    mp3 = _dependency_output(plan, node, BuildStage.ENCODE_MP3)
    inspected = (inspect_audio(master), inspect_audio(mp3))
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
            "inspections": [inspection.to_dict() for inspection in inspected],
        },
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
            reference_audio_sha256=raw.get("reference_audio_sha256"),
            reference_rights_basis=raw.get("reference_rights_basis"),
            watermark_disclosure=raw.get("watermark_disclosure"),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    valid, _ = verify_artifact(record)
    return record if valid else None


def _dependency_output(plan: BuildPlan, node: PlanNode, stage: BuildStage) -> Path:
    for dependency in node.dependencies:
        for candidate in plan.nodes:
            if candidate.id == dependency and candidate.stage is stage:
                return candidate.output
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


def _watermark_disclosure(profile: BackendProfile) -> str:
    if isinstance(profile.options, ResembleOptions):
        return "Provider watermarking behavior is governed by the Resemble account/API"
    return "No Yakbox watermark is added"


def _concatenate_wav(paths: tuple[Path, ...], destination: Path) -> None:
    if not paths:
        raise ArtifactError("Synthesis produced no chunks")
    params: tuple[int, int, int] | None = None
    frames: list[bytes] = []
    for path in paths:
        with wave.open(str(path), "rb") as source:
            current = (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
            )
            if params is None:
                params = current
            elif params != current:
                raise ArtifactError("Synthesized chunks have incompatible WAV formats")
            frames.append(source.readframes(source.getnframes()))
    if params is None:
        raise ArtifactError("Synthesis produced no readable WAV chunks")
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(params[0])
        writer.setsampwidth(params[1])
        writer.setframerate(params[2])
        for frame in frames:
            writer.writeframes(frame)
    atomic_write_bytes(destination, output.getvalue(), overwrite=True)


def _write_silence(path: Path, milliseconds: int, *, sample_rate: int) -> None:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\0\0" * int(sample_rate * milliseconds / 1_000))
    atomic_write_bytes(path, output.getvalue(), overwrite=True)


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
        )[:1_000]
    raise BuildError("No chapter is available for audition")


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
    texts = tuple(
        chunk
        for node in synthesis_nodes
        for chunk in node.chunks
        if _PAUSE.fullmatch(chunk) is None
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
        from_stage=from_stage.value,
        through_stage=through_stage.value,
    )


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
    previous = _latest_successful_plan(manifest.root, plan.target)
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


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w-]+", "-", value.casefold()).strip("-") or "audiobook"


def _assembly_fingerprint(chapters: tuple[Path, ...]) -> str:
    digest = hashlib.sha256(b"audiobook-m4b-v1")
    for chapter in chapters:
        digest.update(bytes.fromhex(sha256_file(chapter)))
    return digest.hexdigest()
