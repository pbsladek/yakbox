"""Targeted, reviewable regeneration of source-addressed audiobook chunks."""

from __future__ import annotations

import hashlib
import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_bytes, atomic_write_json, safe_child, sha256_file
from yakbox.audio.crop import (
    crop_aligned_wav,
    inspect_signal_quality,
    inspect_speech_islands,
)
from yakbox.audiobook.journal import new_run_id
from yakbox.audiobook.manifest import AudiobookManifest, BackendProfile
from yakbox.audiobook.planner import BuildStage, ChunkRoute, PlanNode, plan_audiobook
from yakbox.audiobook.repairs import ApprovedRepair, install_approved_repair
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
from yakbox.speech.alignment import WindowSpeechAligner
from yakbox.speech.short_synthesis import synthesize_short_utterance
from yakbox.speech.short_utterances import carrier_recipes
from yakbox.whisper_qa import classify_clip_type, evaluate_alignment

_MAXIMUM_REPAIR_TAKES = 20


class RepairMode(StrEnum):
    """Scope and synthesis-context policy for one repair session."""

    TARGET_ONLY = "target-only"
    CONTEXT = "context"
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

    def to_dict(self, *, workspace: Path) -> dict[str, object]:
        """Serialize one take using workspace-relative audition paths."""
        return {
            "number": self.number,
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
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

    def to_dict(self, *, workspace: Path) -> dict[str, object]:
        """Serialize the versioned session for review and later approval."""
        return {
            **runtime_metadata("audiobook-repair-session"),
            "repair_id": self.id,
            "plan": self.plan.to_dict(workspace=workspace),
            "takes": [take.to_dict(workspace=workspace) for take in self.takes],
        }


@dataclass(frozen=True, slots=True)
class RepairApproval:
    """Approved replacements and the chapter that must be rebuilt."""

    target: str
    chapter_id: str
    repairs: tuple[ApprovedRepair, ...]


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
    chunks = tuple(_repair_chunk(node, index) for index in selected)
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
    whisper_qa: bool = True,
    api_key: str | None = None,
) -> RepairSession:
    """Generate reviewable takes while keeping each backend worker warm."""
    if not 1 <= takes <= _MAXIMUM_REPAIR_TAKES:
        raise ValidationError("Repair takes must be between 1 and 20")
    repair_id = new_run_id()
    root = manifest.root / ".yakbox" / "repair-sessions" / repair_id
    aligner = (
        _repair_aligner(manifest)
        if whisper_qa or plan.mode is RepairMode.CONTEXT
        else None
    )
    services: dict[tuple[str, str | None], TextToSpeechService] = {}
    async with AsyncExitStack() as stack:
        generated = [
            await _generate_take(
                manifest,
                plan,
                repair_id=repair_id,
                number=number,
                root=root,
                aligner=aligner,
                services=services,
                stack=stack,
                api_key=api_key,
            )
            for number in range(1, takes + 1)
        ]
    session = RepairSession(
        id=repair_id,
        plan=plan,
        takes=tuple(generated),
        report_path=root / "report.json",
    )
    atomic_write_json(session.report_path, session.to_dict(workspace=manifest.root))
    return session


def approve_repair_session(
    manifest: AudiobookManifest,
    *,
    repair_id: str,
    take: int,
) -> RepairApproval:
    """Approve one complete take after revalidating source and candidate bytes."""
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
    approved: list[ApprovedRepair] = []
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
        approved.append(
            install_approved_repair(
                manifest.root,
                target,
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
    return RepairApproval(target=target, chapter_id=chapter, repairs=tuple(approved))


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
        mode=mode,
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
    if mode in {RepairMode.TARGET_ONLY, RepairMode.CONTEXT}:
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


def _repair_chunk(node: PlanNode, index: int) -> RepairChunk:
    source = node.chunk_sources[index]
    route = node.chunk_routes[index]
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
    )


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
    repair_id: str,
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
    for chunk in plan.chunks:
        profile = manifest.profile(_required_profile(chunk))
        service = await _repair_service(
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
            repair_id=repair_id,
            take=number,
        )
        try:
            if plan.mode is RepairMode.CONTEXT:
                if aligner is None:
                    raise BuildError("Context repair requires Whisper alignment")
                await _synthesize_context_candidate(
                    manifest,
                    chunk,
                    request=request,
                    service=service,
                    aligner=aligner,
                    destination=destination,
                    take=number,
                )
            else:
                await service.synthesize_to_file(request, destination, overwrite=True)
                reasons.extend(
                    await _qa_direct_candidate(manifest, chunk, destination, aligner)
                )
        except BuildError:
            reasons.append(f"chunk_{chunk.id}_generation_failed")
            continue
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
    )


async def _repair_service(
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
        )
    )
    services[key] = service
    return service


def _repair_request(
    manifest: AudiobookManifest,
    profile: BackendProfile,
    chunk: RepairChunk,
    *,
    repair_id: str,
    take: int,
) -> SpeechSynthesisRequest:
    from yakbox.audiobook.build import (  # noqa: PLC0415 - avoids build cycle
        _resolved_speech,
    )

    voice, rate, project, use_hd, reference, chatterbox = _resolved_speech(
        profile, manifest
    )
    varied = (
        replace(chatterbox, seed=_repair_seed(repair_id, chunk.id, take))
        if chatterbox is not None
        else None
    )
    return SpeechSynthesisRequest(
        text=chunk.text,
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
    )


async def _qa_direct_candidate(
    manifest: AudiobookManifest,
    chunk: RepairChunk,
    path: Path,
    aligner: WindowSpeechAligner | None,
) -> tuple[str, ...]:
    _trim_direct_candidate_edges(manifest, path)
    reasons = list(_signal_reasons(manifest, path))
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


def _signal_reasons(manifest: AudiobookManifest, path: Path) -> tuple[str, ...]:
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


def _workspace_path(path: Path, workspace: Path) -> str:
    resolved = path.resolve()
    return (
        resolved.relative_to(workspace).as_posix()
        if resolved.is_relative_to(workspace)
        else resolved.as_posix()
    )


def _repair_seed(repair_id: str, chunk_id: str, take: int) -> int:
    material = f"repair-v1\0{repair_id}\0{chunk_id}\0{take}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
