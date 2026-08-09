"""Frozen short-utterance calibration metrics for Whisper safety thresholds."""

from __future__ import annotations

import hashlib
import json
import time
import tomllib
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from yakbox.contracts import runtime_metadata
from yakbox.errors import ValidationError
from yakbox.speech.alignment import (
    AlignmentResult,
    AlignmentSegment,
    AlignmentToken,
    lexical_tokens,
    validate_extracted_alignment,
)
from yakbox.speech.short_utterances import ShortUtterancePolicy

DEFAULT_CALIBRATION_CORPUS = (
    Path(__file__).parent / "data" / "whisper-calibration-v1.toml"
)


@dataclass(frozen=True, slots=True)
class CalibrationCaseResult:
    """One frozen case outcome with lexical and crop measurements."""

    id: str
    expected_accept: bool
    accepted: bool
    reason_codes: tuple[str, ...]
    token_edits: int
    expected_token_count: int
    crop_error_ms: float
    listening_score: int
    consonant_sensitive: bool


def calibrate_frozen_corpus(
    path: Path = DEFAULT_CALIBRATION_CORPUS,
) -> dict[str, object]:
    """Evaluate checked-in evidence against the current policy thresholds."""
    started = time.perf_counter()
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    raw_bytes = path.read_bytes()
    try:
        raw = tomllib.loads(raw_bytes.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValidationError(
            f"Cannot parse Whisper calibration corpus: {path}"
        ) from error
    if raw.get("schema_version") != 1 or not isinstance(raw.get("cases"), list):
        raise ValidationError("Whisper calibration corpus requires schema_version = 1")
    policy = ShortUtterancePolicy()
    cases = tuple(_evaluate_case(item, policy) for item in raw["cases"])
    false_accepts = sum(item.accepted and not item.expected_accept for item in cases)
    false_rejects = sum(not item.accepted and item.expected_accept for item in cases)
    total_expected_tokens = sum(item.expected_token_count for item in cases)
    total_edits = sum(item.token_edits for item in cases)
    consonant_cases = tuple(item for item in cases if item.consonant_sensitive)
    thresholds_fingerprint = policy.fingerprint
    corpus_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    benchmark_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "corpus_sha256": corpus_sha256,
                "thresholds_fingerprint": thresholds_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _current_memory, peak_memory = tracemalloc.get_traced_memory()
    if not tracing:
        tracemalloc.stop()
    runtime_seconds = time.perf_counter() - started
    return {
        **runtime_metadata("whisper-calibration"),
        "corpus": str(path.resolve()),
        "corpus_sha256": corpus_sha256,
        "thresholds_fingerprint": thresholds_fingerprint,
        "benchmark_fingerprint": benchmark_fingerprint,
        "case_count": len(cases),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "exact_case_accuracy": sum(
            item.accepted == item.expected_accept for item in cases
        )
        / len(cases),
        "token_accuracy": 1.0 - total_edits / max(1, total_expected_tokens),
        "mean_crop_error_ms": sum(item.crop_error_ms for item in cases) / len(cases),
        "consonant_case_accuracy": sum(
            item.accepted == item.expected_accept for item in consonant_cases
        )
        / max(1, len(consonant_cases)),
        "mean_listening_score": sum(item.listening_score for item in cases)
        / len(cases),
        "runtime_seconds": runtime_seconds,
        "peak_memory_bytes": peak_memory,
        "cases": [
            {
                "id": item.id,
                "expected_accept": item.expected_accept,
                "accepted": item.accepted,
                "reason_codes": list(item.reason_codes),
                "token_edits": item.token_edits,
                "expected_token_count": item.expected_token_count,
                "crop_error_ms": item.crop_error_ms,
                "listening_score": item.listening_score,
                "consonant_sensitive": item.consonant_sensitive,
            }
            for item in cases
        ],
    }


def _evaluate_case(raw: object, policy: ShortUtterancePolicy) -> CalibrationCaseResult:
    if not isinstance(raw, dict):
        raise ValidationError("Whisper calibration cases must be TOML tables")
    raw = cast(dict[object, object], raw)
    identifier = _string(raw, "id")
    expected = _string(raw, "expected")
    recognized = _string(raw, "recognized")
    expected_tokens = lexical_tokens(expected)
    recognized_tokens = lexical_tokens(recognized)
    starts = _number_list(raw, "word_starts")
    ends = _number_list(raw, "word_ends")
    if len(starts) != len(recognized_tokens) or len(ends) != len(recognized_tokens):
        raise ValidationError(f"Calibration case {identifier} has inconsistent words")
    confidence = _number(raw, "confidence")
    tokens = tuple(
        AlignmentToken(word, start, end, confidence)
        for word, start, end in zip(recognized_tokens, starts, ends, strict=True)
    )
    segment = AlignmentSegment(
        start_seconds=min(starts),
        end_seconds=max(ends),
        average_log_probability=-0.2,
        compression_ratio=1.0,
        no_speech_probability=0.01,
        temperature=0.0,
    )
    decision = validate_extracted_alignment(
        AlignmentResult(tokens, (), "frozen", "fixture", "fixture", (segment,)),
        target_text=expected,
        minimum_confidence=policy.minimum_confidence_for(
            max(1, len(expected_tokens)),
            extracted=True,
        ),
        maximum_extra_speech_ms=policy.maximum_extra_speech_ms,
        minimum_duration_seconds=0.06 * max(1, len(expected_tokens)),
        maximum_duration_seconds=1.6 + 0.75 * (len(expected_tokens) - 1),
        maximum_internal_token_gap_ms=policy.maximum_internal_token_gap_ms,
        maximum_temperature=policy.maximum_segment_temperature,
    )
    expected_start = _number(raw, "expected_start")
    expected_end = _number(raw, "expected_end")
    crop_error = (
        abs(min(starts) - expected_start) + abs(max(ends) - expected_end)
    ) * 500
    return CalibrationCaseResult(
        id=identifier,
        expected_accept=_boolean(raw, "expected_accept"),
        accepted=decision.accepted,
        reason_codes=decision.reason_codes,
        token_edits=_edit_distance(expected_tokens, recognized_tokens),
        expected_token_count=len(expected_tokens),
        crop_error_ms=crop_error,
        listening_score=int(_number(raw, "listening_score")),
        consonant_sensitive=_boolean(raw, "consonant_sensitive"),
    )


def _edit_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _string(raw: dict[object, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Calibration case {key} must be a string")
    return value


def _number(raw: dict[object, object], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"Calibration case {key} must be numeric")
    return float(value)


def _number_list(raw: dict[object, object], key: str) -> tuple[float, ...]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise ValidationError(f"Calibration case {key} must be an array")
    return tuple(_number({key: item}, key) for item in value)


def _boolean(raw: dict[object, object], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValidationError(f"Calibration case {key} must be boolean")
    return value
