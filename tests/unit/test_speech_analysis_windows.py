from __future__ import annotations

from itertools import pairwise

import pytest

from yakbox.errors import ValidationError
from yakbox.speech.analysis_models import (
    AudioSpan,
    CanonicalAudioIdentity,
    FrameCoordinateMap,
    SourceTextSpan,
    SpokenTextPlan,
    SpokenTextSegment,
)
from yakbox.speech.analysis_windows import (
    AnalysisWindowPlanner,
    SegmentAudioSpan,
    WindowPlanningPolicy,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _canonical(frame_count: int = 1_000) -> CanonicalAudioIdentity:
    return CanonicalAudioIdentity(
        source_digest=SHA_A,
        source_format="wav",
        canonical_digest=SHA_B,
        canonical_format="wav-pcm-s16le-mono",
        preprocessing_fingerprint=SHA_C,
        frame_map=FrameCoordinateMap(48_000, 16_000, frame_count * 3, frame_count),
    )


def _spoken(token_counts: tuple[int, ...]) -> SpokenTextPlan:
    segments = tuple(
        SpokenTextSegment(
            segment_id=f"segment-{index}",
            source=SourceTextSpan(SHA_A, index + 1, 0, index + 1, 10),
            display_text_hash=SHA_B,
            synthesis_text_hash=SHA_C,
            expected_lexical_tokens=tuple(
                f"word{index}_{token}" for token in range(count)
            ),
            expected_phonemes=(),
            speaker="narrator",
            profile="narrator",
            language="en",
            boundary="sentence",
            transforms=(),
        )
        for index, count in enumerate(token_counts)
    )
    return SpokenTextPlan(1, SHA_A, SHA_B, segments)


def test_window_planner_groups_assembly_segments_and_owns_overlap_once() -> None:
    canonical = _canonical(400)
    spoken = _spoken((2, 2, 2, 2))
    audio = tuple(
        SegmentAudioSpan(
            f"segment-{index}",
            AudioSpan(SHA_B, index * 100, (index + 1) * 100, 16_000),
        )
        for index in range(4)
    )
    planner = AnalysisWindowPlanner(WindowPlanningPolicy(1, 400, 50))

    first = planner.plan(
        canonical=canonical,
        spoken_text=spoken,
        segment_audio=audio,
    )
    reordered = planner.plan(
        canonical=canonical,
        spoken_text=spoken,
        segment_audio=tuple(reversed(audio)),
    )

    assert first == reordered
    assert len(first.windows) == 2
    assert tuple(
        (window.owned_token_start, window.owned_token_end) for window in first.windows
    ) == ((0, 6), (6, 8))
    assert tuple(
        (window.owned_frame_start, window.owned_frame_end) for window in first.windows
    ) == ((0, 300), (300, 400))
    assert first.windows[0].span.end_frame == 350
    assert first.windows[1].span.start_frame == 250
    assert all(
        window.span.end_frame - window.span.start_frame <= 400
        for window in first.windows
    )


def test_window_planner_splits_model_limited_segment_deterministically() -> None:
    canonical = _canonical()
    spoken = _spoken((10,))
    audio = (SegmentAudioSpan("segment-0", AudioSpan(SHA_B, 0, 1_000, 16_000)),)
    planner = AnalysisWindowPlanner(WindowPlanningPolicy(1, 400, 50))

    result = planner.plan(
        canonical=canonical,
        spoken_text=spoken,
        segment_audio=audio,
    )

    assert len(result.windows) == 4
    assert {window.boundary_reason for window in result.windows} == {"model_limit"}
    assert result.windows[0].owned_frame_start == 0
    assert result.windows[-1].owned_frame_end == 1_000
    assert result.windows[0].owned_token_start == 0
    assert result.windows[-1].owned_token_end == 10
    assert all(
        left.owned_token_end == right.owned_token_start
        for left, right in pairwise(result.windows)
    )


def test_window_planner_refuses_unprojectable_long_short_utterance() -> None:
    canonical = _canonical()
    spoken = _spoken((1,))
    audio = (SegmentAudioSpan("segment-0", AudioSpan(SHA_B, 0, 1_000, 16_000)),)

    with pytest.raises(ValidationError, match="verified token boundaries"):
        AnalysisWindowPlanner(WindowPlanningPolicy(1, 400, 50)).plan(
            canonical=canonical,
            spoken_text=spoken,
            segment_audio=audio,
        )
