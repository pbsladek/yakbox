"""Deterministic shared analysis-window planning in canonical audio frames."""

from __future__ import annotations

import math
from dataclasses import dataclass

from yakbox.errors import ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_models import (
    AnalysisWindow,
    AnalysisWindowPlan,
    AudioSpan,
    CanonicalAudioIdentity,
    SpokenTextPlan,
)

WINDOW_PLANNER_VERSION = 1
STITCHING_VERSION = 1


@dataclass(frozen=True, slots=True)
class SegmentAudioSpan:
    """Canonical audio occupied by one spoken-text segment."""

    segment_id: str
    span: AudioSpan

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValidationError("Segment audio span requires an identifier")


@dataclass(frozen=True, slots=True)
class WindowPlanningPolicy:
    """Engine-independent context and overlap bounds for shared windows."""

    version: int
    maximum_window_frames: int
    overlap_frames: int

    def __post_init__(self) -> None:
        if self.version != WINDOW_PLANNER_VERSION:
            raise ValidationError("Unsupported analysis-window planner version")
        if self.maximum_window_frames <= 0 or self.overlap_frames < 0:
            raise ValidationError("Analysis-window frame limits are invalid")
        if self.overlap_frames * 2 >= self.maximum_window_frames:
            raise ValidationError("Analysis-window overlap leaves no owned context")

    @property
    def owned_frame_limit(self) -> int:
        return self.maximum_window_frames - (2 * self.overlap_frames)

    @property
    def stitching_fingerprint(self) -> str:
        return semantic_fingerprint(
            "analysis-window-stitching-v1",
            {
                "planner_version": self.version,
                "stitching_version": STITCHING_VERSION,
                "maximum_window_frames": self.maximum_window_frames,
                "overlap_frames": self.overlap_frames,
                "ownership": "half_open_frame_and_lexical_token_v1",
            },
        )


@dataclass(frozen=True, slots=True)
class _ResolvedSegment:
    segment_id: str
    span: AudioSpan
    token_start: int
    token_end: int


@dataclass(frozen=True, slots=True)
class _OwnedChunk:
    frame_start: int
    frame_end: int
    token_start: int
    token_end: int
    boundary_reason: str


class AnalysisWindowPlanner:
    """Plan one immutable window set used by every recognition backend."""

    def __init__(self, policy: WindowPlanningPolicy) -> None:
        self.policy = policy

    def plan(
        self,
        *,
        canonical: CanonicalAudioIdentity,
        spoken_text: SpokenTextPlan,
        segment_audio: tuple[SegmentAudioSpan, ...],
    ) -> AnalysisWindowPlan:
        segments = _resolve_segments(canonical, spoken_text, segment_audio)
        chunks = _owned_chunks(segments, self.policy.owned_frame_limit)
        windows = tuple(
            _window_for_chunk(
                canonical=canonical,
                segments=segments,
                chunk=chunk,
                index=index,
                policy=self.policy,
            )
            for index, chunk in enumerate(chunks)
        )
        return AnalysisWindowPlan(
            version=WINDOW_PLANNER_VERSION,
            canonical_audio_fingerprint=canonical.fingerprint,
            spoken_text_plan_fingerprint=spoken_text.fingerprint,
            stitching_fingerprint=self.policy.stitching_fingerprint,
            windows=windows,
        )


def _resolve_segments(
    canonical: CanonicalAudioIdentity,
    spoken_text: SpokenTextPlan,
    segment_audio: tuple[SegmentAudioSpan, ...],
) -> tuple[_ResolvedSegment, ...]:
    by_id = {item.segment_id: item for item in segment_audio}
    if len(by_id) != len(segment_audio):
        raise ValidationError("Segment audio identifiers must be unique")
    expected_ids = tuple(segment.segment_id for segment in spoken_text.segments)
    if set(by_id) != set(expected_ids):
        raise ValidationError("Segment audio must exactly cover the spoken-text plan")
    resolved: list[_ResolvedSegment] = []
    token_cursor = 0
    previous_end = 0
    for text_segment in spoken_text.segments:
        audio = by_id[text_segment.segment_id]
        span = audio.span
        if (
            span.audio_digest != canonical.canonical_digest
            or span.sample_rate != canonical.frame_map.analysis_rate
            or span.end_frame > canonical.frame_map.analysis_frame_count
        ):
            raise ValidationError(
                "Segment audio uses a different canonical audio space"
            )
        if span.start_frame < previous_end:
            raise ValidationError("Segment audio spans must not overlap")
        token_end = token_cursor + len(text_segment.expected_lexical_tokens)
        resolved.append(
            _ResolvedSegment(
                text_segment.segment_id,
                span,
                token_cursor,
                token_end,
            )
        )
        token_cursor = token_end
        previous_end = span.end_frame
    return tuple(resolved)


def _owned_chunks(
    segments: tuple[_ResolvedSegment, ...],
    owned_frame_limit: int,
) -> tuple[_OwnedChunk, ...]:
    chunks: list[_OwnedChunk] = []
    grouped: list[_ResolvedSegment] = []
    for segment in segments:
        duration = segment.span.end_frame - segment.span.start_frame
        if duration > owned_frame_limit:
            _flush_group(grouped, chunks)
            grouped = []
            chunks.extend(_split_segment(segment, owned_frame_limit))
            continue
        if (
            grouped
            and segment.span.end_frame - grouped[0].span.start_frame > owned_frame_limit
        ):
            _flush_group(grouped, chunks)
            grouped = []
        grouped.append(segment)
    _flush_group(grouped, chunks)
    return tuple(chunks)


def _flush_group(grouped: list[_ResolvedSegment], chunks: list[_OwnedChunk]) -> None:
    if not grouped:
        return
    chunks.append(
        _OwnedChunk(
            grouped[0].span.start_frame,
            grouped[-1].span.end_frame,
            grouped[0].token_start,
            grouped[-1].token_end,
            "assembly_boundary",
        )
    )


def _split_segment(
    segment: _ResolvedSegment,
    owned_frame_limit: int,
) -> tuple[_OwnedChunk, ...]:
    duration = segment.span.end_frame - segment.span.start_frame
    chunk_count = math.ceil(duration / owned_frame_limit)
    token_count = segment.token_end - segment.token_start
    if chunk_count > token_count:
        raise ValidationError(
            "A model-limited segment needs verified token boundaries before planning"
        )
    frame_bounds = tuple(
        segment.span.start_frame + (duration * index // chunk_count)
        for index in range(chunk_count + 1)
    )
    token_bounds = tuple(
        segment.token_start + (token_count * index // chunk_count)
        for index in range(chunk_count + 1)
    )
    return tuple(
        _OwnedChunk(
            frame_bounds[index],
            frame_bounds[index + 1],
            token_bounds[index],
            token_bounds[index + 1],
            "model_limit",
        )
        for index in range(chunk_count)
    )


def _window_for_chunk(
    *,
    canonical: CanonicalAudioIdentity,
    segments: tuple[_ResolvedSegment, ...],
    chunk: _OwnedChunk,
    index: int,
    policy: WindowPlanningPolicy,
) -> AnalysisWindow:
    frame_count = canonical.frame_map.analysis_frame_count
    context_start = max(0, chunk.frame_start - policy.overlap_frames)
    context_end = min(frame_count, chunk.frame_end + policy.overlap_frames)
    segment_ids = tuple(
        segment.segment_id
        for segment in segments
        if segment.span.start_frame < context_end
        and segment.span.end_frame > context_start
    )
    identity = {
        "index": index,
        "audio_digest": canonical.canonical_digest,
        "context": (context_start, context_end),
        "owned_frames": (chunk.frame_start, chunk.frame_end),
        "owned_tokens": (chunk.token_start, chunk.token_end),
        "segments": segment_ids,
        "boundary_reason": chunk.boundary_reason,
        "stitching_fingerprint": policy.stitching_fingerprint,
    }
    return AnalysisWindow(
        window_id=semantic_fingerprint("analysis-window-v1", identity),
        span=AudioSpan(
            canonical.canonical_digest,
            context_start,
            context_end,
            canonical.frame_map.analysis_rate,
        ),
        expected_segment_ids=segment_ids,
        owned_frame_start=chunk.frame_start,
        owned_frame_end=chunk.frame_end,
        owned_token_start=chunk.token_start,
        owned_token_end=chunk.token_end,
        boundary_reason=chunk.boundary_reason,
        maximum_context_frames=policy.maximum_window_frames,
    )


__all__ = [
    "STITCHING_VERSION",
    "WINDOW_PLANNER_VERSION",
    "AnalysisWindowPlanner",
    "SegmentAudioSpan",
    "WindowPlanningPolicy",
]
