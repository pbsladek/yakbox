from __future__ import annotations

from dataclasses import replace

import pytest
from tests.unit.test_speech_analysis_contracts import (
    SHA_A,
    SHA_B,
    _alignment,
    _execution,
    _model,
    _recognition,
)

import yakbox.speech.analysis_models as models
from yakbox.errors import ValidationError


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("source", "start_line", 0),
        ("source", "end_line", 0),
        ("source", "start_character", -1),
        ("transform", "kind", ""),
        ("transform", "version", 0),
        ("transform", "input_start", -1),
        ("transform", "output_end", -1),
        ("transform", "input_start", 2),
        ("audio", "start_frame", -1),
        ("audio", "end_frame", 0),
        ("audio", "sample_rate", 0),
        ("coordinates", "source_rate", 0),
        ("coordinates", "analysis_frame_count", 0),
        ("model", "engine", ""),
        ("model", "backend_package", ""),
        ("model", "adapter_version", 0),
        ("model", "worker_protocol_version", 0),
        ("conversion", "tool", ""),
        ("conversion", "source", ""),
        ("execution", "python_version", ""),
        ("execution", "architecture", ""),
        ("token", "text", ""),
        ("token", "text", "Uppercase"),
        ("token", "start_frame", None),
        ("token", "start_frame", -1),
        ("token", "end_frame", 0),
        ("token", "score", float("nan")),
        ("token", "score", 1.1),
        ("token", "score_kind", models.ScoreKind.UNAVAILABLE),
        ("recognition", "engine", "qwen"),
        ("recognition", "requested_language", "fr"),
        ("forced", "engine", "whisper"),
        ("forced", "coverage_ratio", float("inf")),
        ("forced", "coverage_ratio", -0.1),
        ("unit", "start_frame", -1),
        ("unit", "end_frame", 0),
        ("lexical", "start", -1),
        ("lexical", "end", 0),
        ("window", "window_id", ""),
        ("window", "expected_segment_ids", ()),
        ("window", "expected_segment_ids", ("s1", "s1")),
        ("window", "owned_frame_end", 2000),
        ("window", "owned_frame_start", -1),
        ("window", "owned_token_start", -1),
        ("window", "owned_token_end", 0),
        ("window", "maximum_context_frames", 0),
        ("window", "maximum_context_frames", 100),
    ],
)
def test_domain_records_reject_invalid_provenance_timing_and_score_values(
    kind: str, field: str, value: object
) -> None:
    model = _model("whisper")
    recognition = _recognition("whisper", ("hello",))
    records = {
        "source": models.SourceTextSpan(SHA_A, 1, 0, 1, 5),
        "transform": models.TextTransform("pronunciation", 1, 0, 1, 0, 1, SHA_A),
        "audio": models.AudioSpan(SHA_A, 0, 1600, 16000),
        "coordinates": models.FrameCoordinateMap(16000, 16000, 1600, 1600),
        "model": model,
        "conversion": model.conversion,
        "execution": _execution(),
        "token": recognition.tokens[0],
        "recognition": recognition,
        "forced": _alignment(),
        "unit": models.ForcedAlignmentUnit(SHA_A, 0, 100),
        "lexical": models.LexicalSpan(0, 1),
        "window": models.AnalysisWindow(
            "window-1",
            models.AudioSpan(SHA_A, 0, 1600, 16000),
            ("s1",),
            0,
            1600,
            0,
            1,
            "sentence",
            1600,
        ),
    }
    with pytest.raises(ValidationError):
        replace(records[kind], **{field: value})


def _timing() -> models.FrameTimingMap:
    return models.FrameTimingMap(
        16000, 16000, 1600, 1600, 0, 0, 1, 1, 0, 0, True, True, SHA_A
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"source_rate": 0},
        {"source_origin_frame": -1},
        {"uncertainty_frames": -1},
        {"scale_denominator": 0},
        {"monotonic": False},
    ],
)
def test_timing_map_requires_valid_positive_shape_and_monotonic_qualification(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        replace(_timing(), **changes)


@pytest.mark.parametrize("bounds", [(-1, 100), (10, 10), (0, 1601)])
def test_timing_map_rejects_intervals_outside_its_source(
    bounds: tuple[int, int],
) -> None:
    with pytest.raises(ValidationError, match="outside"):
        _timing().map_span_outward(*bounds)


def test_unqualified_timing_maps_cannot_authorize_frame_projection() -> None:
    with pytest.raises(ValidationError, match="not qualified"):
        replace(_timing(), qualified=False).map_span_outward(0, 100)


@pytest.mark.parametrize(
    "failure", ["empty", "duplicate", "offset", "audio", "gap", "overlap"]
)
def test_window_plan_requires_complete_unique_nonoverlapping_ownership(
    failure: str,
) -> None:
    span = models.AudioSpan(SHA_A, 0, 1600, 16000)
    first = models.AnalysisWindow(
        "first", span, ("s1",), 0, 800, 0, 1, "sentence", 1600
    )
    second = models.AnalysisWindow(
        "second", span, ("s2",), 800, 1600, 1, 2, "sentence", 1600
    )
    windows = (first, second)
    if failure == "empty":
        windows = ()
    elif failure == "duplicate":
        windows = (first, first)
    elif failure == "offset":
        windows = (replace(first, owned_token_start=1, owned_token_end=2),)
    elif failure == "audio":
        windows = (first, replace(second, span=replace(span, audio_digest=SHA_B)))
    elif failure == "gap":
        windows = (first, replace(second, owned_token_start=2, owned_token_end=3))
    else:
        windows = (first, replace(second, owned_frame_start=700))
    with pytest.raises(ValidationError):
        models.AnalysisWindowPlan(1, SHA_A, SHA_A, SHA_A, windows)


def test_window_plan_has_a_stable_fingerprint_for_valid_contiguous_ownership() -> None:
    first = models.AnalysisWindow(
        "first",
        models.AudioSpan(SHA_A, 0, 1600, 16000),
        ("s1",),
        0,
        1600,
        0,
        1,
        "sentence",
        1600,
    )
    plan = models.AnalysisWindowPlan(1, SHA_A, SHA_A, SHA_A, (first,))
    assert plan.fingerprint == replace(plan).fingerprint
    assert plan.fingerprint != replace(plan, stitching_fingerprint=SHA_B).fingerprint
