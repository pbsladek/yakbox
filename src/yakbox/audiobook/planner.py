from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from yakbox._files import sha256_file
from yakbox.audiobook.manifest import (
    AudiobookManifest,
    BackendProfile,
    BuildTarget,
    ChatterboxOptions,
)
from yakbox.audiobook.sources import (
    CHATTERBOX_CHUNK_CHARACTERS,
    Chapter,
    ChunkBoundary,
    NormalizedDocument,
    Pause,
    SourceLocation,
    SpeechSegment,
    plan_text_chunks,
)
from yakbox.contracts import runtime_metadata
from yakbox.errors import ValidationError
from yakbox.fingerprints import backend_fingerprint, media_tool_fingerprint
from yakbox.speech.short_utterances import classify_short_utterance


class BuildStage(StrEnum):
    SYNTHESIZE = "synthesize"
    MASTER = "master"
    VERIFY_MANUSCRIPT = "verify_manuscript"
    ENCODE_MP3 = "encode_mp3"
    INSPECT = "inspect"
    RELEASE = "release"


@dataclass(frozen=True, slots=True)
class ChunkRoute:
    """Resolved speaker, profile, and performance settings for one chunk."""

    speaker: str | None
    profile: str | None
    cfg_weight: float | None = None
    exaggeration: float | None = None
    seed: int | None = None

    @classmethod
    def from_profile(cls, speaker: str, profile: BackendProfile) -> ChunkRoute:
        """Capture the effective synthesis controls for a routed speaker."""
        options = profile.options
        if isinstance(options, ChatterboxOptions):
            return cls(
                speaker=speaker,
                profile=profile.name,
                cfg_weight=options.cfg_weight,
                exaggeration=options.exaggeration,
                seed=options.seed,
            )
        return cls(speaker=speaker, profile=profile.name)


@dataclass(frozen=True, slots=True)
class AttributionFinding:
    """Source-located suggestion for improving spoken speaker attribution."""

    code: str
    speaker: str
    message: str
    suggestion: str
    source: SourceLocation

    def to_dict(self, *, root: Path | None = None) -> dict[str, object]:
        """Serialize one attribution suggestion without including source text."""
        path = self.source.path
        path_value = (
            path.relative_to(root).as_posix()
            if root is not None and path.is_relative_to(root)
            else path.as_posix()
        )
        return {
            "code": self.code,
            "speaker": self.speaker,
            "message": self.message,
            "suggestion": self.suggestion,
            "source": {
                "path": path_value,
                "start_line": self.source.start_line,
                "end_line": self.source.end_line,
            },
        }


@dataclass(frozen=True, slots=True)
class ShortUtteranceMarker:
    """Privacy-safe planning evidence for one risky speech chunk."""

    word_count: int
    reason: str
    policy_fingerprint: str


@dataclass(frozen=True, slots=True)
class PlanNode:
    """Fingerprint-addressed unit of work in an audiobook build graph."""

    id: str
    stage: BuildStage
    chapter_id: str
    fingerprint: str
    dependencies: tuple[str, ...]
    output: Path
    chunks: tuple[str, ...] = ()
    chunk_sources: tuple[SourceLocation, ...] = ()
    chunk_boundaries: tuple[str, ...] = ()
    chunk_routes: tuple[ChunkRoute, ...] = ()
    chunk_short_utterances: tuple[ShortUtteranceMarker | None, ...] = ()


@dataclass(frozen=True, slots=True)
class BuildPlan:
    """Deterministic dependency graph for producing one audiobook target."""

    schema_version: int
    target: str
    profile: str
    document_sha256: str
    fingerprint: str
    nodes: tuple[PlanNode, ...]
    complete_document: bool = True
    attribution_findings: tuple[AttributionFinding, ...] = ()

    def to_dict(self, *, root: Path | None = None) -> dict[str, object]:
        """Serialize the plan using its versioned JSON contract."""

        def path_value(path: Path) -> str:
            if root is not None and path.is_relative_to(root):
                return path.relative_to(root).as_posix()
            return path.as_posix()

        return {
            **runtime_metadata("audiobook-plan"),
            "target": self.target,
            "profile": self.profile,
            "document_sha256": self.document_sha256,
            "fingerprint": self.fingerprint,
            "complete_document": self.complete_document,
            "attribution_findings": [
                item.to_dict(root=root) for item in self.attribution_findings
            ],
            "nodes": [
                {
                    "id": node.id,
                    "stage": node.stage.value,
                    "chapter_id": node.chapter_id,
                    "fingerprint": node.fingerprint,
                    "dependencies": list(node.dependencies),
                    "output": path_value(node.output),
                    "chunks": [
                        {
                            "sha256": hashlib.sha256(chunk.encode()).hexdigest(),
                            "characters": len(chunk),
                            "boundary": boundary,
                            "speaker": route.speaker,
                            "profile": route.profile,
                            "performance": (
                                {
                                    "cfg_weight": route.cfg_weight,
                                    "exaggeration": route.exaggeration,
                                    "seed": route.seed,
                                }
                                if route.profile is not None
                                else None
                            ),
                            "short_utterance": (
                                asdict(short_utterance)
                                if short_utterance is not None
                                else None
                            ),
                            "source": {
                                "path": path_value(source.path),
                                "start_line": source.start_line,
                                "end_line": source.end_line,
                            },
                        }
                        for chunk, source, boundary, route, short_utterance in zip(
                            node.chunks,
                            node.chunk_sources,
                            _chunk_boundaries(node),
                            _chunk_routes(node),
                            _chunk_short_utterances(node),
                            strict=True,
                        )
                    ],
                }
                for node in self.nodes
            ],
        }


def plan_audiobook(
    manifest: AudiobookManifest,
    document: NormalizedDocument,
    *,
    target_name: str = "default",
    profile_override: str | None = None,
    chapter_selector: str | None = None,
) -> BuildPlan:
    """Create a deterministic build graph without model, network, or file writes."""
    target = manifest.target(target_name)
    profile = _primary_profile(manifest, target.profile, profile_override)
    chapters = _select_chapters(document.chapters, chapter_selector)
    attribution_findings = _attribution_findings(manifest, chapters)
    _validate_attribution_policy(manifest, attribution_findings)
    media_runtime = media_tool_fingerprint()
    book_payload = json.dumps(
        {
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
        sort_keys=True,
    )
    nodes: list[PlanNode] = []
    for chapter in chapters:
        chunk_items: list[str] = []
        chunk_sources: list[SourceLocation] = []
        chunk_boundaries: list[str] = []
        chunk_routes: list[ChunkRoute] = []
        chunk_short_utterances: list[ShortUtteranceMarker | None] = []
        routed_profiles: list[BackendProfile] = []
        routed_chunk_limits: list[int] = []
        for item in chapter.segments:
            if isinstance(item, SpeechSegment):
                routed_profile = (
                    manifest.profile(item.profile_override)
                    if item.profile_override is not None
                    else _speaker_profile(manifest, item.speaker, profile)
                )
                _validate_routed_profile(profile, routed_profile, item.speaker)
                routed_profiles.append(routed_profile)
                route = ChunkRoute.from_profile(item.speaker, routed_profile)
                chunk_limit = _synthesis_chunk_limit(
                    routed_profile.backend, target.chunk_chars
                )
                routed_chunk_limits.append(chunk_limit)
                segment_chunks = plan_text_chunks(item.text, chunk_limit)
                chunk_items.extend(chunk.text for chunk in segment_chunks)
                chunk_sources.extend(item.source for _ in segment_chunks)
                chunk_routes.extend(route for _ in segment_chunks)
                chunk_short_utterances.extend(
                    _short_utterance_marker(chunk.text, manifest)
                    for chunk in segment_chunks
                )
                chunk_boundaries.extend(
                    (
                        item.boundary_after.value
                        if index == len(segment_chunks) - 1
                        else chunk.boundary.value
                    )
                    for index, chunk in enumerate(segment_chunks)
                )
            elif isinstance(item, Pause):
                _append_pause_chunk(
                    chunk_items,
                    chunk_sources,
                    chunk_boundaries,
                    chunk_routes,
                    short_utterances=chunk_short_utterances,
                    pause=item,
                )
        chunks = tuple(chunk_items)
        boundaries = tuple(chunk_boundaries)
        routes = tuple(chunk_routes)
        short_utterances = tuple(chunk_short_utterances)
        synthesis_fingerprint = _synthesis_fingerprint(
            chapter.id,
            chunks=chunks,
            boundaries=boundaries,
            routes=routes,
            short_utterances=short_utterances,
            profiles=tuple(routed_profiles),
            configured_chunk_limit=target.chunk_chars,
            routed_chunk_limits=tuple(routed_chunk_limits),
            qa_fingerprint=_fingerprint(
                "speech-qa-v1",
                manifest.short_utterances.fingerprint,
                json.dumps(asdict(manifest.whisper_qa), sort_keys=True, default=str),
            ),
        )
        raw_output = target.output_root / "raw" / f"{chapter.id}.wav"
        synth_id = f"{chapter.id}:synthesize"
        nodes.append(
            PlanNode(
                id=synth_id,
                stage=BuildStage.SYNTHESIZE,
                chapter_id=chapter.id,
                fingerprint=synthesis_fingerprint,
                dependencies=(),
                output=raw_output,
                chunks=chunks,
                chunk_sources=tuple(chunk_sources),
                chunk_boundaries=boundaries,
                chunk_routes=routes,
                chunk_short_utterances=short_utterances,
            )
        )
        master_fingerprint = _fingerprint(
            "master-v1",
            synthesis_fingerprint,
            str(target.wav_sample_rate),
            str(target.mastering),
            media_runtime,
        )
        master_id = f"{chapter.id}:master"
        master_output = target.output_root / "mastered" / f"{chapter.id}.wav"
        nodes.append(
            PlanNode(
                id=master_id,
                stage=BuildStage.MASTER,
                chapter_id=chapter.id,
                fingerprint=master_fingerprint,
                dependencies=(synth_id,),
                output=master_output,
            )
        )
        delivery_dependency, delivery_fingerprint = _append_verification_node(
            nodes,
            manifest,
            target=target,
            chapter=chapter,
            master_id=master_id,
            master_fingerprint=master_fingerprint,
        )
        mp3_fingerprint = _fingerprint(
            "mp3-v2",
            delivery_fingerprint,
            target.mp3_bitrate,
            chapter.title,
            str(chapter.order),
            book_payload,
            media_runtime,
        )
        mp3_id = f"{chapter.id}:encode_mp3"
        mp3_output = target.output_root / "release" / "mp3" / f"{chapter.id}.mp3"
        nodes.append(
            PlanNode(
                id=mp3_id,
                stage=BuildStage.ENCODE_MP3,
                chapter_id=chapter.id,
                fingerprint=mp3_fingerprint,
                dependencies=(delivery_dependency,),
                output=mp3_output,
            )
        )
        nodes.append(
            PlanNode(
                id=f"{chapter.id}:inspect",
                stage=BuildStage.INSPECT,
                chapter_id=chapter.id,
                fingerprint=_fingerprint(
                    "inspect-v2",
                    mp3_fingerprint,
                    media_runtime,
                    str(target.quality_min_lufs),
                    str(target.quality_max_lufs),
                    str(target.quality_max_true_peak_dbfs),
                    str(target.quality_max_leading_silence_seconds),
                    str(target.quality_max_trailing_silence_seconds),
                ),
                dependencies=(master_id, mp3_id),
                output=target.output_root / "reports" / f"{chapter.id}.inspection.json",
            )
        )
    plan_fingerprint = _fingerprint(
        "audiobook-plan-v1",
        manifest.book.title,
        target_name,
        profile.name,
        document.sha256,
        *(node.fingerprint for node in nodes),
    )
    return BuildPlan(
        schema_version=1,
        target=target_name,
        profile=profile.name,
        document_sha256=document.sha256,
        fingerprint=plan_fingerprint,
        nodes=tuple(nodes),
        complete_document=chapter_selector is None,
        attribution_findings=attribution_findings,
    )


def _append_verification_node(
    nodes: list[PlanNode],
    manifest: AudiobookManifest,
    *,
    target: BuildTarget,
    chapter: Chapter,
    master_id: str,
    master_fingerprint: str,
) -> tuple[str, str]:
    if not manifest.whisper_qa.chapter_verification:
        return master_id, master_fingerprint
    verification_id = f"{chapter.id}:verify_manuscript"
    fingerprint = _fingerprint(
        "verify-manuscript-v1",
        master_fingerprint,
        manifest.short_utterances.alignment_backend,
        manifest.short_utterances.alignment_model,
        str(manifest.short_utterances.alignment_revision),
        str(manifest.short_utterances.decode_consensus),
        str(manifest.short_utterances.maximum_consensus_timing_delta_ms),
    )
    node = PlanNode(
        id=verification_id,
        stage=BuildStage.VERIFY_MANUSCRIPT,
        chapter_id=chapter.id,
        fingerprint=fingerprint,
        dependencies=(master_id,),
        output=target.output_root
        / "reports"
        / f"{chapter.id}.manuscript-verification.json",
    )
    nodes.append(node)
    return verification_id, fingerprint


def _synthesis_fingerprint(
    chapter_id: str,
    *,
    chunks: tuple[str, ...],
    boundaries: tuple[str, ...],
    routes: tuple[ChunkRoute, ...],
    short_utterances: tuple[ShortUtteranceMarker | None, ...],
    profiles: tuple[BackendProfile, ...],
    configured_chunk_limit: int,
    routed_chunk_limits: tuple[int, ...],
    qa_fingerprint: str,
) -> str:
    routing_payload = json.dumps(
        tuple(
            (chunk, boundary, asdict(route), asdict(short) if short else None)
            for chunk, boundary, route, short in zip(
                chunks, boundaries, routes, short_utterances, strict=True
            )
        ),
        sort_keys=True,
    )
    profile_payload = json.dumps(
        tuple(asdict(item) for item in profiles),
        sort_keys=True,
        default=str,
    )
    synthesis_runtime = json.dumps(
        sorted({backend_fingerprint(item) for item in profiles})
    )
    return _fingerprint(
        "synthesis-v3",
        chapter_id,
        routing_payload,
        profile_payload,
        synthesis_runtime,
        str(configured_chunk_limit),
        json.dumps(routed_chunk_limits),
        qa_fingerprint,
    )


def _append_pause_chunk(
    chunks: list[str],
    sources: list[SourceLocation],
    boundaries: list[str],
    routes: list[ChunkRoute],
    *,
    short_utterances: list[ShortUtteranceMarker | None],
    pause: Pause,
) -> None:
    chunks.append(f"__YAKBOX_PAUSE_MS={pause.milliseconds}__")
    sources.append(pause.source)
    boundaries.append(ChunkBoundary.EXPLICIT_PAUSE.value)
    routes.append(ChunkRoute(None, None))
    short_utterances.append(None)


def _synthesis_chunk_limit(backend: str, configured: int) -> int:
    if backend.casefold() in {"local", "chatterbox", "chatterbox-local"}:
        return min(configured, CHATTERBOX_CHUNK_CHARACTERS)
    return configured


def _chunk_boundaries(node: PlanNode) -> tuple[str, ...]:
    if node.chunk_boundaries:
        return node.chunk_boundaries
    return tuple(ChunkBoundary.END.value for _ in node.chunks)


def _chunk_routes(node: PlanNode) -> tuple[ChunkRoute, ...]:
    if node.chunk_routes:
        return node.chunk_routes
    return tuple(ChunkRoute(None, None) for _ in node.chunks)


def _chunk_short_utterances(
    node: PlanNode,
) -> tuple[ShortUtteranceMarker | None, ...]:
    if node.chunk_short_utterances:
        return node.chunk_short_utterances
    return tuple(None for _ in node.chunks)


def _short_utterance_marker(
    text: str, manifest: AudiobookManifest
) -> ShortUtteranceMarker | None:
    risk = classify_short_utterance(text, manifest.short_utterances)
    if not risk.risky or risk.reason is None:
        return None
    return ShortUtteranceMarker(
        word_count=risk.word_count,
        reason=risk.reason,
        policy_fingerprint=manifest.short_utterances.fingerprint,
    )


def _primary_profile(
    manifest: AudiobookManifest,
    target_profile: str,
    profile_override: str | None,
) -> BackendProfile:
    if profile_override is not None:
        return manifest.profile(profile_override)
    if manifest.characters:
        return manifest.profile_for_speaker(
            "narrator",
            fallback_profile=target_profile,
        )
    return manifest.profile(target_profile)


def _speaker_profile(
    manifest: AudiobookManifest,
    speaker: str,
    primary: BackendProfile,
) -> BackendProfile:
    if speaker == "narrator" or not manifest.characters:
        return primary
    return manifest.profile_for_speaker(speaker, fallback_profile=primary.name)


def _validate_routed_profile(
    primary: BackendProfile,
    routed: BackendProfile,
    speaker: str,
) -> None:
    if routed.backend != primary.backend or routed.executor != primary.executor:
        raise ValidationError(
            f"Speaker {speaker!r} profile must use {primary.backend}/{primary.executor}"
        )
    primary_options = primary.options
    routed_options = routed.options
    if (
        isinstance(primary_options, ChatterboxOptions)
        and isinstance(routed_options, ChatterboxOptions)
        and routed_options.device != primary_options.device
    ):
        raise ValidationError(
            f"Speaker {speaker!r} profile must use Chatterbox device "
            f"{primary_options.device!r}"
        )


def _attribution_findings(
    manifest: AudiobookManifest,
    chapters: tuple[Chapter, ...],
) -> tuple[AttributionFinding, ...]:
    if not manifest.characters or manifest.dialogue.attribution_assistance == "off":
        return ()
    findings: list[AttributionFinding] = []
    previous: SpeechSegment | None = None
    for chapter in chapters:
        previous = None
        for item in chapter.segments:
            if not isinstance(item, SpeechSegment):
                continue
            _ = manifest.character(item.speaker)
            if _looks_like_dialogue(item.text) and not item.speaker_explicit:
                findings.append(
                    AttributionFinding(
                        code="unrouted-dialogue",
                        speaker="narrator",
                        message=(
                            "Quoted speech has no character route and uses narrator"
                        ),
                        suggestion=(
                            "Add <!-- yakbox:speech:speaker name=... --> before "
                            "the paragraph or retain audible speaker attribution"
                        ),
                        source=item.source,
                    )
                )
            if (
                item.speaker_explicit
                and item.speaker != "narrator"
                and _word_count(item.text) <= manifest.dialogue.short_utterance_words
            ):
                findings.append(
                    AttributionFinding(
                        code="short-dialogue",
                        speaker=item.speaker,
                        message=(
                            "Short isolated dialogue may synthesize unnaturally or "
                            "lose speaker clarity"
                        ),
                        suggestion=(
                            "Expand the spoken turn or retain a brief natural "
                            "attribution beat"
                        ),
                        source=item.source,
                    )
                )
            if _shared_voice_transition(manifest, previous, item):
                findings.append(
                    AttributionFinding(
                        code="shared-voice-transition",
                        speaker=item.speaker,
                        message=(
                            "Adjacent characters use the same logical voice and may "
                            "be hard to distinguish"
                        ),
                        suggestion="Retain a natural speaker tag at this transition",
                        source=item.source,
                    )
                )
            previous = item
    return tuple(findings)


def _shared_voice_transition(
    manifest: AudiobookManifest,
    previous: SpeechSegment | None,
    current: SpeechSegment,
) -> bool:
    if (
        previous is None
        or previous.speaker in {current.speaker, "narrator"}
        or current.speaker == "narrator"
    ):
        return False
    fallback = manifest.character("narrator").profile
    previous_profile = manifest.profile_for_speaker(
        previous.speaker, fallback_profile=fallback
    )
    current_profile = manifest.profile_for_speaker(
        current.speaker, fallback_profile=fallback
    )
    return previous_profile.voice == current_profile.voice


def _looks_like_dialogue(text: str) -> bool:
    return text.lstrip().startswith(('"', "\u201c", "\u2018"))


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w\u2019'-]+", text, flags=re.UNICODE))


def _validate_attribution_policy(
    manifest: AudiobookManifest,
    findings: tuple[AttributionFinding, ...],
) -> None:
    if manifest.dialogue.attribution_assistance != "error" or not findings:
        return
    raise ValidationError(
        f"Attribution assistance found {len(findings)} issue(s); "
        f"first: {findings[0].message}"
    )


def shard_plan(plan: BuildPlan, count: int) -> tuple[tuple[PlanNode, ...], ...]:
    if count < 1:
        raise ValueError("Shard count must be at least 1")
    chapter_ids = list(dict.fromkeys(node.chapter_id for node in plan.nodes))
    assignments: list[list[str]] = [[] for _ in range(min(count, len(chapter_ids)))]
    for index, chapter_id in enumerate(chapter_ids):
        assignments[index % len(assignments)].append(chapter_id)
    return tuple(
        tuple(node for node in plan.nodes if node.chapter_id in chapter_set)
        for chapter_set in (set(items) for items in assignments)
    )


def _select_chapters(
    chapters: tuple[Chapter, ...], selector: str | None
) -> tuple[Chapter, ...]:
    if selector is None:
        return chapters
    terms = tuple(term.strip() for term in selector.split(",") if term.strip())
    if not terms:
        raise ValueError("Chapter selector must not be empty")
    orders: set[int] = set()
    labels: list[str] = []
    for term in terms:
        match = re.fullmatch(r"(\d+)-(\d+)", term)
        if match:
            start, end = (int(value) for value in match.groups())
            if start > end:
                raise ValueError(f"Chapter range starts after it ends: {term}")
            orders.update(range(start, end + 1))
        elif term.isdecimal():
            orders.add(int(term))
        else:
            labels.append(term)
    selected = tuple(
        chapter
        for chapter in chapters
        if chapter.order in orders
        or any(
            label == chapter.id or label.casefold() in chapter.title.casefold()
            for label in labels
        )
    )
    if not selected:
        raise ValueError(f"No chapter matches {selector!r}")
    return selected


def _fingerprint(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()
