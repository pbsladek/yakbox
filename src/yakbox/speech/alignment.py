"""Backend-neutral speech alignment contracts and lexical safety gates."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Protocol, runtime_checkable

import regex

from yakbox.audio.crop import SpeechRegion

_LEXICAL_TOKEN = regex.compile(
    r"[\p{L}\p{M}\p{N}]+(?:[\N{RIGHT SINGLE QUOTATION MARK}'-]"
    r"[\p{L}\p{M}\p{N}]+)*"
)
INTERNAL_SENTENCE_BOUNDARY_PAUSE_MS = 900
MINIMUM_CONSENSUS_DECODE_PASSES = 2


@dataclass(frozen=True, slots=True)
class AlignmentToken:
    """One recognized lexical token with audio-relative timing evidence."""

    text: str
    start_seconds: float
    end_seconds: float
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class AlignmentSegment:
    """Whisper segment-level quality evidence retained for hard gates."""

    start_seconds: float
    end_seconds: float
    average_log_probability: float | None = None
    compression_ratio: float | None = None
    no_speech_probability: float | None = None
    temperature: float | None = None


@dataclass(frozen=True, slots=True)
class DecodePassEvidence:
    """Compact evidence from one independently configured Whisper decode."""

    name: str
    transcript: str
    tokens: tuple[str, ...]
    issues: tuple[str, ...]
    minimum_confidence: float | None
    matches_expected: bool | None


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Normalized output from one local speech alignment backend."""

    tokens: tuple[AlignmentToken, ...]
    speech_regions: tuple[SpeechRegion, ...]
    backend: str
    model: str
    fingerprint: str
    segments: tuple[AlignmentSegment, ...] = ()
    issues: tuple[str, ...] = ()
    language: str | None = None
    timing_source: str = "unprompted"
    transcript: str = ""
    decode_passes: tuple[DecodePassEvidence, ...] = ()
    consensus_score: float | None = None
    maximum_timing_delta_ms: float | None = None
    consensus_reason_codes: tuple[str, ...] = ()
    prompt_sensitivity: str = "not_tested"
    clip_start_seconds: float | None = None
    clip_end_seconds: float | None = None


@runtime_checkable
class SpeechAligner(Protocol):
    """Lazy local aligner used by short-utterance candidate selection."""

    @property
    def fingerprint(self) -> str:
        """Return a stable backend and model fingerprint."""
        ...

    async def align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
    ) -> AlignmentResult:
        """Recognize speech and return word-level timing evidence."""
        ...


@runtime_checkable
class WindowSpeechAligner(SpeechAligner, Protocol):
    """Speech aligner that can inspect bounded ranges with absolute timing."""

    async def align_window(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
        start_seconds: float,
        end_seconds: float,
    ) -> AlignmentResult:
        """Recognize one bounded range while retaining file-relative times."""
        ...


@dataclass(frozen=True, slots=True)
class AlignmentDecision:
    """Hard-gate result for locating one target inside recognized speech."""

    accepted: bool
    reason_codes: tuple[str, ...]
    start_seconds: float | None = None
    end_seconds: float | None = None
    confidence: float | None = None


def lexical_tokens(text: str) -> tuple[str, ...]:
    """Return case-folded Unicode lexical tokens used by planning and QA."""
    return tuple(match.group().casefold() for match in _LEXICAL_TOKEN.finditer(text))


def alignment_fingerprint(
    backend: str,
    model: str,
    version: str,
    *,
    settings: Mapping[str, object] | None = None,
) -> str:
    """Fingerprint an aligner implementation without exposing input text."""
    payload = json.dumps(
        {
            "backend": backend,
            "model": model,
            "version": version,
            "settings": settings or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_carrier_alignment(
    result: AlignmentResult,
    *,
    expected_text: str,
    target_text: str,
    minimum_confidence: float,
    token_aliases: Mapping[str, tuple[str, ...]] | None = None,
    minimum_average_log_probability: float = -1.0,
    maximum_compression_ratio: float = 2.4,
    maximum_no_speech_probability: float = 0.6,
    maximum_temperature: float = 0.2,
    maximum_internal_token_gap_ms: int = 350,
    maximum_token_duration_ms: int = 1_200,
) -> AlignmentDecision:
    """Require an exact carrier transcript and one contiguous target occurrence."""
    expected = _canonical_tokens(lexical_tokens(expected_text), token_aliases)
    target = _canonical_tokens(lexical_tokens(target_text), token_aliases)
    recognized = _canonical_tokens(_recognized_tokens(result.tokens), token_aliases)
    reasons = list(
        _alias_resolved_decode_quality_reasons(
            alignment_quality_reason_codes(
                result,
                minimum_average_log_probability=minimum_average_log_probability,
                maximum_compression_ratio=maximum_compression_ratio,
                maximum_no_speech_probability=maximum_no_speech_probability,
                maximum_temperature=maximum_temperature,
                maximum_token_duration_ms=maximum_token_duration_ms,
            ),
            result,
            expected,
            token_aliases,
        )
    )
    if not target:
        reasons.append("empty_target")
    if recognized != expected:
        reasons.append("carrier_transcript_mismatch")
    occurrences = _occurrences(recognized, target)
    if not occurrences:
        reasons.append("target_missing")
    elif len(occurrences) > 1:
        reasons.append("target_ambiguous")
    if reasons:
        return AlignmentDecision(False, tuple(dict.fromkeys(reasons)))

    start_index = occurrences[0]
    selected = result.tokens[start_index : start_index + len(target)]
    confidences = tuple(
        token.confidence for token in selected if token.confidence is not None
    )
    if len(confidences) != len(selected):
        return AlignmentDecision(False, ("confidence_missing",))
    confidence = min(confidences)
    if confidence < minimum_confidence:
        return AlignmentDecision(False, ("low_confidence",), confidence=confidence)
    start = selected[0].start_seconds
    end = selected[-1].end_seconds
    if _maximum_internal_gap_ms(selected) > _internal_gap_limit_ms(
        target_text,
        maximum_internal_token_gap_ms,
    ):
        return AlignmentDecision(
            False,
            ("excessive_internal_pause",),
            start_seconds=start,
            end_seconds=end,
            confidence=confidence,
        )
    if start < 0 or end <= start:
        return AlignmentDecision(False, ("invalid_timing",), confidence=confidence)
    return AlignmentDecision(
        True,
        (),
        start_seconds=start,
        end_seconds=end,
        confidence=confidence,
    )


def validate_extracted_alignment(
    result: AlignmentResult,
    *,
    target_text: str,
    minimum_confidence: float,
    maximum_extra_speech_ms: int,
    minimum_duration_seconds: float,
    maximum_duration_seconds: float,
    token_aliases: Mapping[str, tuple[str, ...]] | None = None,
    minimum_average_log_probability: float = -1.0,
    maximum_compression_ratio: float = 2.4,
    maximum_no_speech_probability: float = 0.6,
    maximum_temperature: float = 0.2,
    maximum_internal_token_gap_ms: int = 350,
    maximum_token_duration_ms: int = 1_200,
) -> AlignmentDecision:
    """Reject lexical, timing, duration, and acoustic evidence outside a crop."""
    target = _canonical_tokens(lexical_tokens(target_text), token_aliases)
    recognized = _canonical_tokens(_recognized_tokens(result.tokens), token_aliases)
    reasons = list(
        _alias_resolved_decode_quality_reasons(
            alignment_quality_reason_codes(
                result,
                minimum_average_log_probability=minimum_average_log_probability,
                maximum_compression_ratio=maximum_compression_ratio,
                maximum_no_speech_probability=maximum_no_speech_probability,
                maximum_temperature=maximum_temperature,
                maximum_token_duration_ms=maximum_token_duration_ms,
            ),
            result,
            target,
            token_aliases,
        )
    )
    if recognized != target:
        reasons.extend(_transcript_reason_codes(recognized, target))
    if not result.tokens:
        reasons.append("target_missing")
    if reasons:
        return AlignmentDecision(False, tuple(dict.fromkeys(reasons)))

    confidence_values = tuple(
        token.confidence for token in result.tokens if token.confidence is not None
    )
    if len(confidence_values) != len(result.tokens):
        return AlignmentDecision(False, ("confidence_missing",))
    confidence = min(confidence_values)
    if confidence < minimum_confidence:
        reasons.append("low_confidence")
    start = result.tokens[0].start_seconds
    end = result.tokens[-1].end_seconds
    if _maximum_internal_gap_ms(result.tokens) > _internal_gap_limit_ms(
        target_text,
        maximum_internal_token_gap_ms,
    ):
        reasons.append("excessive_internal_pause")
    reasons.extend(
        _timing_reason_codes(
            result,
            start=start,
            end=end,
            minimum_duration_seconds=minimum_duration_seconds,
            maximum_duration_seconds=maximum_duration_seconds,
            maximum_extra_speech_ms=maximum_extra_speech_ms,
        )
    )
    return AlignmentDecision(
        not reasons,
        tuple(dict.fromkeys(reasons)),
        start_seconds=start,
        end_seconds=end,
        confidence=confidence,
    )


def alignment_quality_reason_codes(
    result: AlignmentResult,
    *,
    minimum_average_log_probability: float = -1.0,
    maximum_compression_ratio: float = 2.4,
    maximum_no_speech_probability: float = 0.6,
    maximum_temperature: float = 0.2,
    maximum_token_duration_ms: int = 1_200,
) -> tuple[str, ...]:
    """Return fail-closed structural and Whisper quality-gate reasons."""
    reasons = [
        *result.issues,
        *result.consensus_reason_codes,
        *(
            ("prompt_sensitive_transcript",)
            if result.prompt_sensitivity in {"prompt_destabilized", "prompt_changed"}
            else ()
        ),
        *_token_quality_reason_codes(
            result.tokens,
            maximum_token_duration_ms=maximum_token_duration_ms,
        ),
        *_repetition_reason_codes(result.tokens),
        *_segment_quality_reason_codes(
            result.segments,
            minimum_average_log_probability=minimum_average_log_probability,
            maximum_compression_ratio=maximum_compression_ratio,
            maximum_no_speech_probability=maximum_no_speech_probability,
            maximum_temperature=maximum_temperature,
        ),
    ]
    return tuple(dict.fromkeys(reasons))


def _token_quality_reason_codes(
    tokens: tuple[AlignmentToken, ...],
    *,
    maximum_token_duration_ms: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    previous_end = 0.0
    for token in tokens:
        values = (token.start_seconds, token.end_seconds)
        if not all(math.isfinite(value) for value in values):
            reasons.append("nonfinite_token_timing")
            continue
        if token.start_seconds < 0 or token.end_seconds <= token.start_seconds:
            reasons.append("invalid_token_timing")
        if token.start_seconds < previous_end:
            reasons.append("nonmonotonic_token_timing")
        if (
            token.end_seconds - token.start_seconds
        ) * 1_000 > maximum_token_duration_ms:
            reasons.append("excessive_token_duration")
        previous_end = max(previous_end, token.end_seconds)
        if token.confidence is not None and (
            not math.isfinite(token.confidence) or not 0 <= token.confidence <= 1
        ):
            reasons.append("invalid_token_confidence")
    return tuple(reasons)


def _repetition_reason_codes(
    tokens: tuple[AlignmentToken, ...],
) -> tuple[str, ...]:
    words = tuple(token.text for token in tokens)
    if any(
        words[index] == words[index + 1] == words[index + 2]
        for index in range(max(0, len(words) - 2))
    ):
        return ("probable_repetition_loop",)
    for width in range(2, min(6, len(words) // 3 + 1)):
        for start in range(len(words) - width * 3 + 1):
            phrase = words[start : start + width]
            if (
                words[start + width : start + width * 2] == phrase
                and words[start + width * 2 : start + width * 3] == phrase
            ):
                return ("probable_repetition_loop",)
    return ()


def _segment_quality_reason_codes(
    segments: tuple[AlignmentSegment, ...],
    *,
    minimum_average_log_probability: float,
    maximum_compression_ratio: float,
    maximum_no_speech_probability: float,
    maximum_temperature: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    previous_segment_end = 0.0
    for segment in segments:
        if not math.isfinite(segment.start_seconds) or not math.isfinite(
            segment.end_seconds
        ):
            reasons.append("nonfinite_segment_timing")
        elif (
            segment.start_seconds < 0
            or segment.end_seconds <= segment.start_seconds
            or segment.start_seconds < previous_segment_end
        ):
            reasons.append("invalid_segment_timing")
        previous_segment_end = max(previous_segment_end, segment.end_seconds)
        if (
            segment.average_log_probability is not None
            and segment.average_log_probability < minimum_average_log_probability
        ):
            reasons.append("low_segment_log_probability")
        if (
            segment.compression_ratio is not None
            and segment.compression_ratio > maximum_compression_ratio
        ):
            reasons.append("high_segment_compression_ratio")
        if (
            segment.no_speech_probability is not None
            and segment.no_speech_probability > maximum_no_speech_probability
        ):
            reasons.append("high_no_speech_probability")
        if (
            segment.temperature is not None
            and segment.temperature > maximum_temperature
        ):
            reasons.append("high_segment_temperature")
    return tuple(reasons)


def _occurrences(haystack: tuple[str, ...], needle: tuple[str, ...]) -> tuple[int, ...]:
    if not needle or len(needle) > len(haystack):
        return ()
    return tuple(
        index
        for index in range(len(haystack) - len(needle) + 1)
        if haystack[index : index + len(needle)] == needle
    )


def _timing_reason_codes(
    result: AlignmentResult,
    *,
    start: float,
    end: float,
    minimum_duration_seconds: float,
    maximum_duration_seconds: float,
    maximum_extra_speech_ms: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    duration = end - start
    if start < 0 or duration <= 0:
        reasons.append("invalid_timing")
    if duration < minimum_duration_seconds:
        reasons.append("duration_too_short")
    if duration > maximum_duration_seconds:
        reasons.append("duration_too_long")
    allowed = maximum_extra_speech_ms / 1_000
    if _speech_duration(result.speech_regions, 0, start) > allowed:
        reasons.append("unexpected_prefix_speech")
    tail_end = max(
        (region.end_seconds for region in result.speech_regions), default=end
    )
    if _speech_duration(result.speech_regions, end, tail_end) > allowed:
        reasons.append("unexpected_suffix_speech")
    return tuple(reasons)


def _recognized_tokens(tokens: tuple[AlignmentToken, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for token in tokens:
        normalized = lexical_tokens(token.text)
        if len(normalized) != 1:
            result.append("<invalid-token>")
        else:
            result.append(normalized[0])
    return tuple(result)


def _internal_gap_limit_ms(text: str, baseline_ms: int) -> int:
    return (
        max(baseline_ms, INTERNAL_SENTENCE_BOUNDARY_PAUSE_MS)
        if has_internal_sentence_boundary(text)
        else baseline_ms
    )


def has_internal_sentence_boundary(text: str) -> bool:
    """Return whether text contains punctuation between lexical phrases."""
    return regex.search(r"[.!?]\s+\S", text) is not None


def _canonical_tokens(
    tokens: tuple[str, ...], aliases: Mapping[str, tuple[str, ...]] | None
) -> tuple[str, ...]:
    if not aliases:
        return tokens
    reverse = {
        alias: canonical
        for canonical, accepted in aliases.items()
        for alias in (canonical, *accepted)
    }
    return tuple(_canonical_token(token, reverse) for token in tokens)


def canonical_tokens(
    tokens: tuple[str, ...], aliases: Mapping[str, tuple[str, ...]] | None = None
) -> tuple[str, ...]:
    """Apply explicit transcript aliases while preserving token boundaries."""
    return _canonical_tokens(tokens, aliases)


def _canonical_token(token: str, reverse: Mapping[str, str]) -> str:
    canonical = reverse.get(token)
    if canonical is not None:
        return canonical
    for suffix in ("'s", "\N{RIGHT SINGLE QUOTATION MARK}s"):
        if token.endswith(suffix):
            stem = token[: -len(suffix)]
            canonical_stem = reverse.get(stem)
            if canonical_stem is not None:
                return f"{canonical_stem}{suffix}"
    return token


def _alias_resolved_decode_quality_reasons(
    reasons: tuple[str, ...],
    result: AlignmentResult,
    expected: tuple[str, ...],
    aliases: Mapping[str, tuple[str, ...]] | None,
) -> tuple[str, ...]:
    if not aliases or len(result.decode_passes) < MINIMUM_CONSENSUS_DECODE_PASSES:
        return reasons
    passes = result.decode_passes
    if any(item.issues for item in passes) or any(
        _canonical_tokens(item.tokens, aliases) != expected for item in passes
    ):
        return reasons
    alias_resolved = {"decode_consensus_mismatch", "prompt_sensitive_transcript"}
    return tuple(reason for reason in reasons if reason not in alias_resolved)


def _transcript_reason_codes(
    recognized: tuple[str, ...], target: tuple[str, ...]
) -> tuple[str, ...]:
    occurrences = _occurrences(recognized, target)
    if len(occurrences) == 1:
        start = occurrences[0]
        result: list[str] = []
        if start:
            result.append("unexpected_prefix")
        if start + len(target) < len(recognized):
            result.append("unexpected_suffix")
        return tuple(result)
    if len(occurrences) > 1:
        return ("target_repeated",)
    if not recognized:
        return ("target_missing",)
    return ("target_substituted",)


def _speech_duration(
    regions: tuple[SpeechRegion, ...], start: float, end: float
) -> float:
    return sum(
        max(0.0, min(region.end_seconds, end) - max(region.start_seconds, start))
        for region in regions
    )


def _maximum_internal_gap_ms(tokens: tuple[AlignmentToken, ...]) -> float:
    return max(
        (
            max(0.0, following.start_seconds - previous.end_seconds) * 1_000
            for previous, following in pairwise(tokens)
        ),
        default=0.0,
    )
