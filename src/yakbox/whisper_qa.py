"""Whisper-backed chapter, join, and clip-specific QA services."""

from __future__ import annotations

import hashlib
import math
import wave
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from enum import StrEnum
from itertools import pairwise, product
from pathlib import Path

import regex

from yakbox.audio.crop import wav_duration_seconds
from yakbox.contracts import runtime_metadata
from yakbox.errors import ArtifactError, ValidationError
from yakbox.local_alignment import MlxWhisperAligner
from yakbox.speech.alignment import (
    INTERNAL_CLAUSE_BOUNDARY_PAUSE_MS,
    INTERNAL_SENTENCE_BOUNDARY_PAUSE_MS,
    MINIMUM_CONSENSUS_DECODE_PASSES,
    AlignmentResult,
    AlignmentToken,
    WindowSpeechAligner,
    alignment_quality_reason_codes,
    canonical_tokens,
    lexical_tokens,
)
from yakbox.whisper_cache import CachedWhisperAligner

_MAXIMUM_SHORT_PHRASE_WORDS = 3
_MAXIMUM_JOIN_JUMP_RATIO = 0.25
_JOIN_PHRASE_WORDS = 2
_COMPOUND_PART_COUNT = 2
_MAXIMUM_IGNORED_INSERT_DURATION_SECONDS = 0.2
_MAXIMUM_NEAR_ZERO_INSERT_CONFIDENCE = 0.05
_MAXIMUM_REPEATED_INSERT_CONFIDENCE = 0.35
_MANUSCRIPT_RECHECK_CONTEXT_TOKENS = 8
_MANUSCRIPT_NARROW_RECHECK_CONTEXT_TOKENS = 2
_MANUSCRIPT_WIDE_RECHECK_CONTEXT_TOKENS = 16
_MANUSCRIPT_VERY_WIDE_RECHECK_CONTEXT_TOKENS = 32
_MANUSCRIPT_RECHECK_GUARD_SECONDS = 0.25


class WhisperClipType(StrEnum):
    """Confidence-policy classes with materially different ASR behavior."""

    ONE_WORD = "one_word"
    SHORT_PHRASE = "short_phrase"
    SENTENCE = "sentence"
    JOIN = "join"
    CHAPTER = "chapter"


@dataclass(frozen=True, slots=True)
class ConfidenceProfile:
    """Calibrated Whisper gates for one class of audio."""

    minimum_word_confidence: float
    minimum_average_log_probability: float
    maximum_compression_ratio: float
    maximum_no_speech_probability: float
    maximum_temperature: float
    maximum_internal_gap_ms: int
    maximum_token_duration_ms: int


CONFIDENCE_PROFILES: dict[WhisperClipType, ConfidenceProfile] = {
    WhisperClipType.ONE_WORD: ConfidenceProfile(0.60, -0.80, 2.2, 0.45, 0.0, 0, 900),
    WhisperClipType.SHORT_PHRASE: ConfidenceProfile(
        0.50, -0.90, 2.3, 0.50, 0.2, 320, 1_100
    ),
    WhisperClipType.SENTENCE: ConfidenceProfile(
        0.35, -1.00, 2.4, 0.60, 0.2, 500, 1_300
    ),
    WhisperClipType.JOIN: ConfidenceProfile(0.45, -0.90, 2.3, 0.50, 0.2, 500, 1_200),
    WhisperClipType.CHAPTER: ConfidenceProfile(0.15, -1.00, 2.4, 0.60, 0.2, 900, 1_500),
}


@dataclass(frozen=True, slots=True)
class AlignmentEvaluation:
    """A clip-type-aware decision over one normalized alignment."""

    clip_type: WhisperClipType
    accepted: bool
    reason_codes: tuple[str, ...]
    minimum_word_confidence: float | None
    profile: ConfidenceProfile


@dataclass(frozen=True, slots=True)
class ManuscriptMismatch:
    """One bounded expected/recognized edit with an audio location when known."""

    operation: str
    expected_start: int
    expected_end: int
    recognized_start: int
    recognized_end: int
    expected_preview: tuple[str, ...]
    recognized_preview: tuple[str, ...]
    expected_tokens_omitted: int
    recognized_tokens_omitted: int
    audio_start_seconds: float | None
    audio_end_seconds: float | None


@dataclass(frozen=True, slots=True)
class ManuscriptVerification:
    """Chapter-wide lexical and decode-quality verification evidence."""

    audio: Path
    manuscript: Path
    manuscript_sha256: str
    result: AlignmentResult
    expected_token_count: int
    recognized_token_count: int
    matched_token_count: int
    token_accuracy: float
    mismatches: tuple[ManuscriptMismatch, ...]
    reason_codes: tuple[str, ...]
    diagnostic_reason_codes: tuple[str, ...]
    confidence: AlignmentEvaluation
    alignment_cache_hits: int = 0
    alignment_cache_misses: int = 0

    @property
    def accepted(self) -> bool:
        return not self.reason_codes

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("whisper-manuscript-verification"),
            "audio": str(self.audio),
            "manuscript": str(self.manuscript),
            "manuscript_sha256": self.manuscript_sha256,
            "expected_token_count": self.expected_token_count,
            "recognized_token_count": self.recognized_token_count,
            "matched_token_count": self.matched_token_count,
            "token_accuracy": self.token_accuracy,
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "diagnostic_reason_codes": list(self.diagnostic_reason_codes),
            "mismatches": [asdict(item) for item in self.mismatches],
            "alignment": alignment_evidence(self.result, include_transcripts=False),
            "confidence_profile": asdict(self.confidence.profile),
            "alignment_cache": {
                "hits": self.alignment_cache_hits,
                "misses": self.alignment_cache_misses,
            },
        }


@dataclass(frozen=True, slots=True)
class JoinSpecification:
    """One physical audio join and optional exact local transcript context."""

    at_seconds: float
    expected_before: str = ""
    expected_after: str = ""
    boundary: str = "unknown"


type _JoinWindow = tuple[int, JoinSpecification, float, float, str]


@dataclass(frozen=True, slots=True)
class JoinSignalEvidence:
    """PCM evidence measured directly around an internal splice."""

    sample_jump_ratio: float
    local_peak_delta_ratio: float
    silence_before_ms: float
    silence_after_ms: float


@dataclass(frozen=True, slots=True)
class JoinInspection:
    """ASR and PCM evidence for one targeted chapter join."""

    index: int
    specification: JoinSpecification
    window_start_seconds: float
    window_end_seconds: float
    result: AlignmentResult
    signal: JoinSignalEvidence
    confidence: AlignmentEvaluation
    reason_codes: tuple[str, ...]
    diagnostic_reason_codes: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.reason_codes

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "at_seconds": self.specification.at_seconds,
            "boundary": self.specification.boundary,
            "window_start_seconds": self.window_start_seconds,
            "window_end_seconds": self.window_end_seconds,
            "expected_before_sha256": _text_sha256(self.specification.expected_before),
            "expected_after_sha256": _text_sha256(self.specification.expected_after),
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "diagnostic_reason_codes": list(self.diagnostic_reason_codes),
            "recognized_tokens": [token.text for token in self.result.tokens],
            "signal": asdict(self.signal),
            "confidence_profile": asdict(self.confidence.profile),
            "alignment": alignment_evidence(self.result),
        }


@dataclass(frozen=True, slots=True)
class JoinInspectionReport:
    """One automatic pass over every declared physical join in an audio file."""

    audio: Path
    audio_duration_seconds: float
    window_seconds: float
    joins: tuple[JoinInspection, ...]
    alignment_window_count: int = 0
    coalesced_join_count: int = 0

    @property
    def accepted(self) -> bool:
        return all(item.accepted for item in self.joins)

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("whisper-join-inspection"),
            "audio": str(self.audio),
            "audio_duration_seconds": self.audio_duration_seconds,
            "window_seconds": self.window_seconds,
            "alignment_window_count": self.alignment_window_count,
            "coalesced_join_count": self.coalesced_join_count,
            "accepted": self.accepted,
            "failed_join_count": sum(not item.accepted for item in self.joins),
            "joins": [item.to_dict() for item in self.joins],
        }


def classify_clip_type(expected_text: str | None) -> WhisperClipType:
    """Choose a calibrated class from lexical length."""
    count = len(lexical_tokens(expected_text or ""))
    if count == 1:
        return WhisperClipType.ONE_WORD
    if 1 < count <= _MAXIMUM_SHORT_PHRASE_WORDS:
        return WhisperClipType.SHORT_PHRASE
    return WhisperClipType.SENTENCE


def evaluate_alignment(
    result: AlignmentResult,
    *,
    clip_type: WhisperClipType,
    expected_text: str | None = None,
) -> AlignmentEvaluation:
    """Apply thresholds selected for the semantic type of audio under test."""
    profile = CONFIDENCE_PROFILES[clip_type]
    reasons = list(
        alignment_quality_reason_codes(
            result,
            minimum_average_log_probability=profile.minimum_average_log_probability,
            maximum_compression_ratio=profile.maximum_compression_ratio,
            maximum_no_speech_probability=profile.maximum_no_speech_probability,
            maximum_temperature=profile.maximum_temperature,
            maximum_token_duration_ms=profile.maximum_token_duration_ms,
        )
    )
    confidences = tuple(
        token.confidence for token in result.tokens if token.confidence is not None
    )
    minimum = (
        min(confidences)
        if len(confidences) == len(result.tokens) and confidences
        else None
    )
    if result.tokens and minimum is None:
        reasons.append("confidence_missing")
    elif minimum is not None and minimum < profile.minimum_word_confidence:
        reasons.append("low_confidence")
    expected = lexical_tokens(expected_text or "")
    recognized = tuple(token.text for token in result.tokens)
    if expected_text is not None and recognized != expected:
        reasons.append("expected_transcript_mismatch")
    if not result.tokens:
        reasons.append("target_missing")
    if clip_type is not WhisperClipType.ONE_WORD and _has_unexpected_internal_pause(
        result.tokens,
        expected_text,
        baseline_ms=profile.maximum_internal_gap_ms,
    ):
        reasons.append("excessive_internal_pause")
    return AlignmentEvaluation(
        clip_type=clip_type,
        accepted=not reasons,
        reason_codes=tuple(dict.fromkeys(reasons)),
        minimum_word_confidence=minimum,
        profile=profile,
    )


def _has_unexpected_internal_pause(
    tokens: tuple[AlignmentToken, ...],
    expected_text: str | None,
    *,
    baseline_ms: int,
) -> bool:
    gaps = tuple(
        max(0.0, following.start_seconds - previous.end_seconds) * 1_000
        for previous, following in pairwise(tokens)
    )
    if not gaps:
        return False
    words = tuple(regex.finditer(r"[\p{L}\p{M}\p{N}]+", expected_text or ""))
    if len(words) != len(tokens):
        return max(gaps) > baseline_ms
    for index, gap in enumerate(gaps):
        punctuation = (expected_text or "")[
            words[index].end() : words[index + 1].start()
        ]
        limit = baseline_ms
        if regex.search(r"[.!?]", punctuation):
            limit = max(limit, INTERNAL_SENTENCE_BOUNDARY_PAUSE_MS)
        elif regex.search(r"[,;:]|[\u2013\u2014]", punctuation):
            limit = max(limit, INTERNAL_CLAUSE_BOUNDARY_PAUSE_MS)
        if gap > limit:
            return True
    return False


async def verify_manuscript(
    audio: Path,
    manuscript: Path,
    expected_text: str,
    *,
    language: str,
    model: str,
    revision: str | None,
    aligner: WindowSpeechAligner | None = None,
    cache_root: Path | None = None,
    token_aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> ManuscriptVerification:
    """Verify a complete chapter against its normalized speakable manuscript."""
    resolved_aligner = aligner or MlxWhisperAligner(
        model=model,
        revision=revision,
        prompted_timing=False,
        prompt_sensitivity=False,
        decode_consensus=True,
    )
    cached_aligner = _with_cache(resolved_aligner, cache_root)
    cache_hits_before = (
        cached_aligner.hits if isinstance(cached_aligner, CachedWhisperAligner) else 0
    )
    cache_misses_before = (
        cached_aligner.misses if isinstance(cached_aligner, CachedWhisperAligner) else 0
    )
    result = await cached_aligner.align(audio, expected_text, language=language)
    expected = canonical_tokens(_expanded_lexical_tokens(expected_text), token_aliases)
    comparison_tokens = tuple(
        replace(token, text=canonical)
        for token in result.tokens
        for canonical in canonical_tokens(
            _expanded_lexical_tokens(token.text), token_aliases
        )
    )
    expected, comparison_tokens = _coalesce_digit_sequences(expected, comparison_tokens)
    expected, comparison_tokens = _coalesce_compound_equivalents(
        expected,
        comparison_tokens,
        token_aliases=token_aliases,
    )
    comparison_tokens = _coalesce_directional_alias_equivalents(
        expected,
        comparison_tokens,
        token_aliases=token_aliases,
    )
    comparison_tokens, ignored_insertions = _remove_decoder_insertions(
        expected, comparison_tokens
    )
    comparison_result = replace(result, tokens=comparison_tokens)
    recognized = tuple(token.text for token in comparison_tokens)
    matcher = SequenceMatcher(a=expected, b=recognized, autojunk=False)
    matching = sum(block.size for block in matcher.get_matching_blocks())
    denominator = max(1, len(expected), len(recognized))
    initial_mismatches = tuple(
        _manuscript_mismatch(expected, comparison_result, opcode)
        for opcode in matcher.get_opcodes()
        if opcode[0] != "equal"
    )
    mismatches = await _unresolved_manuscript_mismatches(
        cached_aligner,
        audio,
        language=language,
        expected=expected,
        recognized=comparison_tokens,
        mismatches=initial_mismatches,
        token_aliases=token_aliases,
    )
    locally_resolved = len(initial_mismatches) - len(mismatches)
    if initial_mismatches and not mismatches:
        matching = len(expected)
        recognized = expected
        denominator = max(1, len(expected))
    confidence = evaluate_alignment(
        result,
        clip_type=WhisperClipType.CHAPTER,
        expected_text=expected_text,
    )
    reasons: list[str] = []
    if mismatches:
        reasons.append("manuscript_transcript_mismatch")
    diagnostic_reasons = (*result.issues, *confidence.reason_codes)
    if ignored_insertions:
        diagnostic_reasons = (*diagnostic_reasons, "low_confidence_insert_ignored")
    if locally_resolved:
        diagnostic_reasons = (
            *diagnostic_reasons,
            "localized_mismatch_recheck_passed",
        )
    return ManuscriptVerification(
        audio=audio.resolve(),
        manuscript=manuscript.resolve(),
        manuscript_sha256=_text_sha256(expected_text),
        result=result,
        expected_token_count=len(expected),
        recognized_token_count=len(recognized),
        matched_token_count=matching,
        token_accuracy=matching / denominator,
        mismatches=mismatches,
        reason_codes=tuple(dict.fromkeys(reasons)),
        diagnostic_reason_codes=tuple(dict.fromkeys(diagnostic_reasons)),
        confidence=confidence,
        alignment_cache_hits=(
            cached_aligner.hits - cache_hits_before
            if isinstance(cached_aligner, CachedWhisperAligner)
            else 0
        ),
        alignment_cache_misses=(
            cached_aligner.misses - cache_misses_before
            if isinstance(cached_aligner, CachedWhisperAligner)
            else 0
        ),
    )


async def _unresolved_manuscript_mismatches(
    aligner: WindowSpeechAligner,
    audio: Path,
    *,
    language: str,
    expected: tuple[str, ...],
    recognized: tuple[AlignmentToken, ...],
    mismatches: tuple[ManuscriptMismatch, ...],
    token_aliases: Mapping[str, tuple[str, ...]] | None,
) -> tuple[ManuscriptMismatch, ...]:
    """Confirm long-form transcript edits in bounded independent decodes."""
    if not mismatches or not recognized:
        return mismatches
    duration = wav_duration_seconds(audio)
    unresolved = mismatches
    for context_tokens in (
        _MANUSCRIPT_RECHECK_CONTEXT_TOKENS,
        _MANUSCRIPT_NARROW_RECHECK_CONTEXT_TOKENS,
        _MANUSCRIPT_WIDE_RECHECK_CONTEXT_TOKENS,
        _MANUSCRIPT_VERY_WIDE_RECHECK_CONTEXT_TOKENS,
    ):
        remaining: list[ManuscriptMismatch] = []
        for group in _merged_mismatch_groups(
            unresolved,
            recognized_count=len(recognized),
            context_tokens=context_tokens,
        ):
            combined = _combined_mismatch(group)
            resolved = await _manuscript_mismatch_resolved(
                aligner,
                audio,
                language=language,
                expected=expected,
                recognized=recognized,
                mismatch=combined,
                token_aliases=token_aliases,
                duration=duration,
                context_tokens=context_tokens,
            )
            if not resolved:
                remaining.extend(group)
        unresolved = tuple(remaining)
        if not unresolved:
            break
    return unresolved


def _merged_mismatch_groups(
    mismatches: tuple[ManuscriptMismatch, ...],
    *,
    recognized_count: int,
    context_tokens: int,
) -> tuple[tuple[ManuscriptMismatch, ...], ...]:
    """Coalesce edits whose requested decode windows would overlap."""
    groups: list[list[ManuscriptMismatch]] = []
    end = -1
    for mismatch in sorted(mismatches, key=lambda item: item.recognized_start):
        start = max(0, mismatch.recognized_start - context_tokens)
        candidate_end = min(
            recognized_count,
            mismatch.recognized_end + context_tokens,
        )
        if groups and start <= end:
            groups[-1].append(mismatch)
            end = max(end, candidate_end)
        else:
            groups.append([mismatch])
            end = candidate_end
    return tuple(tuple(group) for group in groups)


def _combined_mismatch(
    group: tuple[ManuscriptMismatch, ...],
) -> ManuscriptMismatch:
    first, last = group[0], group[-1]
    return ManuscriptMismatch(
        operation="merged_recheck",
        expected_start=min(item.expected_start for item in group),
        expected_end=max(item.expected_end for item in group),
        recognized_start=first.recognized_start,
        recognized_end=last.recognized_end,
        expected_preview=(),
        recognized_preview=(),
        expected_tokens_omitted=0,
        recognized_tokens_omitted=0,
        audio_start_seconds=first.audio_start_seconds,
        audio_end_seconds=last.audio_end_seconds,
    )


async def _manuscript_mismatch_resolved(
    aligner: WindowSpeechAligner,
    audio: Path,
    *,
    language: str,
    expected: tuple[str, ...],
    recognized: tuple[AlignmentToken, ...],
    mismatch: ManuscriptMismatch,
    token_aliases: Mapping[str, tuple[str, ...]] | None,
    duration: float,
    context_tokens: int,
) -> bool:
    """Recheck one mismatch in a bounded context window."""
    expected_start = max(0, mismatch.expected_start - context_tokens)
    expected_end = min(
        len(expected),
        mismatch.expected_end + context_tokens,
    )
    recognized_start = max(0, mismatch.recognized_start - context_tokens)
    recognized_end = min(
        len(recognized),
        mismatch.recognized_end + context_tokens,
    )
    if expected_start >= expected_end or recognized_start >= recognized_end:
        return False
    expected_window = expected[expected_start:expected_end]
    start_seconds = max(
        0.0,
        recognized[recognized_start].start_seconds - _MANUSCRIPT_RECHECK_GUARD_SECONDS,
    )
    end_seconds = min(
        duration,
        recognized[recognized_end - 1].end_seconds + _MANUSCRIPT_RECHECK_GUARD_SECONDS,
    )
    if end_seconds <= start_seconds:
        return False
    local = await aligner.align_window(
        audio,
        " ".join(expected_window),
        language=language,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    return _independent_decodes_contain_expected(
        local,
        expected_window,
        token_aliases=token_aliases,
    )


def _independent_decodes_contain_expected(
    result: AlignmentResult,
    expected: tuple[str, ...],
    *,
    token_aliases: Mapping[str, tuple[str, ...]] | None,
) -> bool:
    passes = tuple(item for item in result.decode_passes if not item.issues)
    minimum_confidence = CONFIDENCE_PROFILES[
        WhisperClipType.CHAPTER
    ].minimum_word_confidence
    if result.issues or len(passes) < MINIMUM_CONSENSUS_DECODE_PASSES:
        return False
    return not any(
        item.minimum_confidence is None
        or item.minimum_confidence < minimum_confidence
        or not _tokens_contain_expected(
            item.tokens,
            expected,
            token_aliases=token_aliases,
        )
        for item in passes
    )


def _tokens_contain_expected(
    tokens: tuple[str, ...],
    expected: tuple[str, ...],
    *,
    token_aliases: Mapping[str, tuple[str, ...]] | None,
) -> bool:
    canonical = canonical_tokens(
        tuple(part for token in tokens for part in _expanded_lexical_tokens(token)),
        token_aliases,
    )
    canonical = _collapse_expected_digits(canonical)
    timed = tuple(
        AlignmentToken(token, float(index), float(index + 1), 1.0)
        for index, token in enumerate(canonical)
    )
    normalized_expected, normalized = _coalesce_compound_equivalents(
        expected,
        timed,
        token_aliases=token_aliases,
    )
    normalized = _coalesce_directional_alias_equivalents(
        normalized_expected,
        normalized,
        token_aliases=token_aliases,
    )
    values = tuple(token.text for token in normalized)
    return any(
        values[index : index + len(normalized_expected)] == normalized_expected
        for index in range(len(values) - len(normalized_expected) + 1)
    )


def _expanded_lexical_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        part for token in lexical_tokens(text) for part in token.split("-") if part
    )


def _coalesce_compound_equivalents(
    expected: tuple[str, ...],
    recognized: tuple[AlignmentToken, ...],
    *,
    token_aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[tuple[str, ...], tuple[AlignmentToken, ...]]:
    """Treat joined and split spellings as the same chapter transcript token."""
    normalized_expected = expected
    normalized_recognized = recognized
    while True:
        matcher = SequenceMatcher(
            a=normalized_expected,
            b=tuple(token.text for token in normalized_recognized),
            autojunk=False,
        )
        changed = False
        for operation, start_a, end_a, start_b, end_b in matcher.get_opcodes():
            if operation != "replace":
                continue
            expected_span = normalized_expected[start_a:end_a]
            recognized_span = normalized_recognized[start_b:end_b]
            if (
                len(expected_span) == _COMPOUND_PART_COUNT
                and len(recognized_span) == 1
                and (
                    "".join(expected_span) == recognized_span[0].text
                    or _joined_aliases_match(
                        expected_span,
                        recognized_span[0].text,
                        token_aliases,
                    )
                )
            ):
                normalized_expected = (
                    *normalized_expected[:start_a],
                    recognized_span[0].text,
                    *normalized_expected[end_a:],
                )
                changed = True
                break
            if (
                len(expected_span) == 1
                and len(recognized_span) == _COMPOUND_PART_COUNT
                and (
                    expected_span[0] == "".join(token.text for token in recognized_span)
                    or "".join(token.text for token in recognized_span)
                    in (token_aliases or {}).get(expected_span[0], ())
                )
            ):
                first, last = recognized_span
                merged = replace(
                    first,
                    text=expected_span[0],
                    end_seconds=last.end_seconds,
                    confidence=_minimum_optional_confidence(
                        first.confidence, last.confidence
                    ),
                )
                normalized_recognized = (
                    *normalized_recognized[:start_b],
                    merged,
                    *normalized_recognized[end_b:],
                )
                changed = True
                break
        if not changed:
            return normalized_expected, normalized_recognized


def _coalesce_directional_alias_equivalents(
    expected: tuple[str, ...],
    recognized: tuple[AlignmentToken, ...],
    *,
    token_aliases: Mapping[str, tuple[str, ...]] | None,
) -> tuple[AlignmentToken, ...]:
    """Resolve configured one-way aliases without remapping canonical terms."""
    if not token_aliases:
        return recognized
    normalized = recognized
    while True:
        matcher = SequenceMatcher(
            a=expected,
            b=tuple(token.text for token in normalized),
            autojunk=False,
        )
        changed = False
        for operation, start_a, end_a, start_b, end_b in matcher.get_opcodes():
            if operation != "replace" or end_a - start_a != 1 or end_b - start_b != 1:
                continue
            expected_token = expected[start_a]
            recognized_token = normalized[start_b]
            if recognized_token.text not in token_aliases.get(expected_token, ()):
                continue
            normalized = (
                *normalized[:start_b],
                replace(recognized_token, text=expected_token),
                *normalized[end_b:],
            )
            changed = True
            break
        if not changed:
            return normalized


def _joined_aliases_match(
    expected: tuple[str, ...],
    recognized: str,
    aliases: Mapping[str, tuple[str, ...]] | None,
) -> bool:
    if not aliases:
        return False
    choices = tuple((token, *aliases.get(token, ())) for token in expected)
    return any("".join(parts) == recognized for parts in product(*choices))


_SPOKEN_DIGITS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def _coalesce_digit_sequences(
    expected: tuple[str, ...],
    recognized: tuple[AlignmentToken, ...],
) -> tuple[tuple[str, ...], tuple[AlignmentToken, ...]]:
    """Normalize Whisper's written and spoken forms of digit sequences."""
    return _collapse_expected_digits(expected), _collapse_recognized_digits(recognized)


def _collapse_expected_digits(tokens: tuple[str, ...]) -> tuple[str, ...]:
    collapsed: list[str] = []
    digits: list[str] = []
    for token in (*tokens, ""):
        digit = _spoken_digit(token)
        if digit is not None:
            digits.append(digit)
            continue
        if digits:
            collapsed.append("".join(digits))
            digits.clear()
        if token:
            collapsed.append(token)
    return tuple(collapsed)


def _collapse_recognized_digits(
    tokens: tuple[AlignmentToken, ...],
) -> tuple[AlignmentToken, ...]:
    collapsed: list[AlignmentToken] = []
    sequence: list[AlignmentToken] = []
    for token in (*tokens, None):
        digit = _spoken_digit(token.text) if token is not None else None
        if digit is not None and token is not None:
            sequence.append(replace(token, text=digit))
            continue
        if sequence:
            first, last = sequence[0], sequence[-1]
            collapsed.append(
                replace(
                    first,
                    text="".join(item.text for item in sequence),
                    end_seconds=last.end_seconds,
                    confidence=_minimum_confidences(sequence),
                )
            )
            sequence.clear()
        if token is not None:
            collapsed.append(token)
    return tuple(collapsed)


def _spoken_digit(token: str) -> str | None:
    if token.isdigit():
        return token
    return _SPOKEN_DIGITS.get(token)


def _minimum_confidences(tokens: list[AlignmentToken]) -> float | None:
    values = tuple(token.confidence for token in tokens if token.confidence is not None)
    return min(values) if values else None


def _minimum_optional_confidence(
    first: float | None, second: float | None
) -> float | None:
    values = tuple(value for value in (first, second) if value is not None)
    return min(values) if values else None


def _remove_decoder_insertions(
    expected: tuple[str, ...],
    recognized: tuple[AlignmentToken, ...],
) -> tuple[tuple[AlignmentToken, ...], tuple[AlignmentToken, ...]]:
    """Remove narrowly evidenced Whisper-only insertions from comparison tokens."""
    filtered = recognized
    ignored: list[AlignmentToken] = []
    while True:
        matcher = SequenceMatcher(
            a=expected,
            b=tuple(token.text for token in filtered),
            autojunk=False,
        )
        removable: tuple[int, ...] = ()
        for operation, _, _, start, end in matcher.get_opcodes():
            if operation != "insert":
                continue
            indexes = tuple(range(start, end))
            if indexes and all(
                _is_decoder_insertion(filtered, index) for index in indexes
            ):
                removable = indexes
                break
        if not removable:
            return filtered, tuple(ignored)
        ignored.extend(filtered[index] for index in removable)
        removed = set(removable)
        filtered = tuple(
            token for index, token in enumerate(filtered) if index not in removed
        )


def _is_decoder_insertion(tokens: tuple[AlignmentToken, ...], index: int) -> bool:
    token = tokens[index]
    confidence = token.confidence
    if confidence is None:
        return False
    duration = token.end_seconds - token.start_seconds
    if duration > _MAXIMUM_IGNORED_INSERT_DURATION_SECONDS:
        return False
    if confidence <= _MAXIMUM_NEAR_ZERO_INSERT_CONFIDENCE:
        return True
    repeated = (index > 0 and tokens[index - 1].text == token.text) or (
        index + 1 < len(tokens) and tokens[index + 1].text == token.text
    )
    return repeated and confidence <= _MAXIMUM_REPEATED_INSERT_CONFIDENCE


async def inspect_joins(
    audio: Path,
    joins: tuple[JoinSpecification, ...],
    *,
    language: str,
    model: str,
    revision: str | None,
    window_seconds: float = 1.5,
    coalesce_gap_seconds: float = 0.1,
    aligner: WindowSpeechAligner | None = None,
    cache_root: Path | None = None,
) -> JoinInspectionReport:
    """Inspect every declared join using targeted ASR and internal PCM evidence."""
    if not joins:
        raise ValidationError("At least one join must be provided")
    if window_seconds <= 0 or not math.isfinite(window_seconds):
        raise ValidationError("Join inspection window must be positive")
    if coalesce_gap_seconds < 0 or not math.isfinite(coalesce_gap_seconds):
        raise ValidationError("Join coalescing gap cannot be negative")
    duration = wav_duration_seconds(audio)
    resolved_aligner = aligner or MlxWhisperAligner(
        model=model,
        revision=revision,
        prompted_timing=False,
        prompt_sensitivity=True,
        decode_consensus=True,
    )
    cached_aligner = _with_cache(resolved_aligner, cache_root)
    windows = _validated_join_windows(
        joins,
        duration=duration,
        window_seconds=window_seconds,
    )
    results, alignment_window_count, coalesced_join_count = await _align_join_windows(
        audio,
        windows,
        aligner=cached_aligner,
        language=language,
        coalesce_gap_seconds=coalesce_gap_seconds,
    )
    inspections: list[JoinInspection] = []
    for index, specification, start, end, expected in windows:
        result = results[index]
        confidence = evaluate_alignment(
            result,
            clip_type=WhisperClipType.JOIN,
            expected_text=expected if expected else None,
        )
        signal = inspect_join_signal(audio, at_seconds=specification.at_seconds)
        reasons, diagnostics = _join_reason_codes(
            specification, result, confidence, signal
        )
        inspections.append(
            JoinInspection(
                index=index,
                specification=specification,
                window_start_seconds=start,
                window_end_seconds=end,
                result=result,
                signal=signal,
                confidence=confidence,
                reason_codes=reasons,
                diagnostic_reason_codes=diagnostics,
            )
        )
    return JoinInspectionReport(
        audio.resolve(),
        duration,
        window_seconds,
        tuple(inspections),
        alignment_window_count,
        coalesced_join_count,
    )


def _validated_join_windows(
    joins: tuple[JoinSpecification, ...],
    *,
    duration: float,
    window_seconds: float,
) -> tuple[_JoinWindow, ...]:
    windows: list[_JoinWindow] = []
    previous = -1.0
    for index, specification in enumerate(joins, start=1):
        at = specification.at_seconds
        if not math.isfinite(at) or at <= 0 or at >= duration:
            raise ValidationError(f"Join {index} falls outside the audio")
        if at <= previous:
            raise ValidationError("Join timestamps must be strictly increasing")
        previous = at
        start = max(0.0, at - window_seconds)
        end = min(duration, at + window_seconds)
        expected = " ".join(
            value.strip()
            for value in (
                specification.expected_before,
                specification.expected_after,
            )
            if value.strip()
        )
        windows.append((index, specification, start, end, expected))
    return tuple(windows)


async def _align_join_windows(
    audio: Path,
    windows: tuple[_JoinWindow, ...],
    *,
    aligner: WindowSpeechAligner,
    language: str,
    coalesce_gap_seconds: float,
) -> tuple[dict[int, AlignmentResult], int, int]:
    results: dict[int, AlignmentResult] = {}
    alignment_window_count = 0
    contextual = tuple(item for item in windows if item[4])
    for index, _specification, start, end, expected in contextual:
        results[index] = await aligner.align_window(
            audio,
            expected,
            language=language,
            start_seconds=start,
            end_seconds=end,
        )
        alignment_window_count += 1
    context_free = tuple(item for item in windows if not item[4])
    groups = _coalesced_windows(context_free, gap_seconds=coalesce_gap_seconds)
    for group in groups:
        group_start = min(item[2] for item in group)
        group_end = max(item[3] for item in group)
        combined = await aligner.align_window(
            audio,
            "",
            language=language,
            start_seconds=group_start,
            end_seconds=group_end,
        )
        alignment_window_count += 1
        for index, _specification, start, end, _expected in group:
            results[index] = _slice_alignment(combined, start=start, end=end)
    return results, alignment_window_count, max(0, len(context_free) - len(groups))


def _join_reason_codes(
    specification: JoinSpecification,
    result: AlignmentResult,
    confidence: AlignmentEvaluation,
    signal: JoinSignalEvidence,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    diagnostics = [
        reason
        for reason in confidence.reason_codes
        if not (
            specification.boundary.startswith("explicit_pause")
            and reason == "excessive_internal_pause"
        )
    ]
    diagnostics.extend(
        _join_lexical_reasons(result, at_seconds=specification.at_seconds)
    )
    reasons: list[str] = []
    if signal.sample_jump_ratio > _MAXIMUM_JOIN_JUMP_RATIO:
        reasons.append("join_click")
    return (
        tuple(dict.fromkeys(reasons)),
        tuple(dict.fromkeys(diagnostics)),
    )


def _with_cache(
    aligner: WindowSpeechAligner,
    cache_root: Path | None,
) -> WindowSpeechAligner:
    if cache_root is None or isinstance(aligner, CachedWhisperAligner):
        return aligner
    return CachedWhisperAligner(aligner, cache_root)


def _coalesced_windows(
    windows: tuple[_JoinWindow, ...],
    *,
    gap_seconds: float,
) -> tuple[tuple[_JoinWindow, ...], ...]:
    """Merge context-free overlapping join windows into minimal ASR passes."""
    groups: list[list[_JoinWindow]] = []
    for item in windows:
        if not groups or item[2] > max(value[3] for value in groups[-1]) + gap_seconds:
            groups.append([item])
        else:
            groups[-1].append(item)
    return tuple(tuple(group) for group in groups)


def _slice_alignment(
    result: AlignmentResult,
    *,
    start: float,
    end: float,
) -> AlignmentResult:
    """Project one coalesced decode back onto a join-specific time window."""
    tokens = tuple(
        token
        for token in result.tokens
        if token.end_seconds > start and token.start_seconds < end
    )
    segments = tuple(
        segment
        for segment in result.segments
        if segment.end_seconds > start and segment.start_seconds < end
    )
    regions = tuple(
        region
        for region in result.speech_regions
        if region.end_seconds > start and region.start_seconds < end
    )
    return replace(
        result,
        tokens=tokens,
        segments=segments,
        speech_regions=regions,
        transcript=" ".join(token.text for token in tokens),
        clip_start_seconds=start,
        clip_end_seconds=end,
    )


def inspect_join_signal(path: Path, *, at_seconds: float) -> JoinSignalEvidence:
    """Measure a click and surrounding silence at an internal PCM boundary."""
    try:
        with wave.open(str(path), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            frame_count = reader.getnframes()
            compression = reader.getcomptype()
            content = reader.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as error:
        raise ArtifactError(f"Cannot read PCM WAV: {path}") from error
    if compression != "NONE" or channels < 1 or width not in {1, 2, 3, 4}:
        raise ArtifactError(f"Unsupported PCM WAV format: {path}")
    frame_size = channels * width
    boundary = round(at_seconds * rate)
    if boundary <= 0 or boundary >= frame_count:
        raise ValidationError("Join timestamp falls outside the PCM samples")
    window = max(1, round(rate * 0.02))
    start = max(0, boundary - window)
    end = min(frame_count, boundary + window)
    mono = tuple(
        sum(
            _decode_pcm_sample(
                content[
                    frame * frame_size + channel * width : frame * frame_size
                    + (channel + 1) * width
                ],
                width,
            )
            for channel in range(channels)
        )
        / channels
        for frame in range(start, end)
    )
    split = boundary - start
    full_scale = float((1 << (width * 8 - 1)) - 1)
    jump = abs(mono[split] - mono[split - 1]) / full_scale
    before_peak = max((abs(value) for value in mono[:split]), default=0.0)
    after_peak = max((abs(value) for value in mono[split:]), default=0.0)
    local_peak_delta = abs(after_peak - before_peak) / full_scale
    silence_threshold = full_scale * 10 ** (-48 / 20)
    return JoinSignalEvidence(
        sample_jump_ratio=jump,
        local_peak_delta_ratio=local_peak_delta,
        silence_before_ms=_edge_silence_ms(
            mono[:split], silence_threshold, rate, reverse=True
        ),
        silence_after_ms=_edge_silence_ms(
            mono[split:], silence_threshold, rate, reverse=False
        ),
    )


def alignment_evidence(
    result: AlignmentResult,
    *,
    include_transcripts: bool = True,
) -> dict[str, object]:
    """Serialize shared multi-pass evidence without manuscript contents."""
    return {
        "backend": result.backend,
        "model": result.model,
        "fingerprint": result.fingerprint,
        "language": result.language,
        "timing_source": result.timing_source,
        "clip_start_seconds": result.clip_start_seconds,
        "clip_end_seconds": result.clip_end_seconds,
        "consensus_score": result.consensus_score,
        "maximum_timing_delta_ms": result.maximum_timing_delta_ms,
        "consensus_reason_codes": list(result.consensus_reason_codes),
        "prompt_sensitivity": result.prompt_sensitivity,
        "decode_passes": [
            (
                asdict(item)
                if include_transcripts
                else {
                    "name": item.name,
                    "transcript_sha256": _text_sha256(item.transcript),
                    "token_count": len(item.tokens),
                    "issues": list(item.issues),
                    "minimum_confidence": item.minimum_confidence,
                    "matches_expected": item.matches_expected,
                }
            )
            for item in result.decode_passes
        ],
        "segments": [asdict(item) for item in result.segments],
        "issues": list(result.issues),
    }


def _manuscript_mismatch(
    expected: tuple[str, ...],
    result: AlignmentResult,
    opcode: tuple[str, int, int, int, int],
) -> ManuscriptMismatch:
    operation, expected_start, expected_end, recognized_start, recognized_end = opcode
    expected_values = expected[expected_start:expected_end]
    recognized_values = tuple(
        token.text for token in result.tokens[recognized_start:recognized_end]
    )
    recognized_slice = result.tokens[recognized_start:recognized_end]
    if recognized_slice:
        audio_start = recognized_slice[0].start_seconds
        audio_end = recognized_slice[-1].end_seconds
    elif recognized_start > 0:
        audio_start = result.tokens[recognized_start - 1].end_seconds
        audio_end = audio_start
    elif recognized_start < len(result.tokens):
        audio_start = result.tokens[recognized_start].start_seconds
        audio_end = audio_start
    else:
        audio_start = None
        audio_end = None
    return ManuscriptMismatch(
        operation=operation,
        expected_start=expected_start,
        expected_end=expected_end,
        recognized_start=recognized_start,
        recognized_end=recognized_end,
        expected_preview=expected_values[:12],
        recognized_preview=recognized_values[:12],
        expected_tokens_omitted=max(0, len(expected_values) - 12),
        recognized_tokens_omitted=max(0, len(recognized_values) - 12),
        audio_start_seconds=audio_start,
        audio_end_seconds=audio_end,
    )


def _join_lexical_reasons(
    result: AlignmentResult,
    *,
    at_seconds: float,
) -> tuple[str, ...]:
    before = tuple(token for token in result.tokens if token.end_seconds <= at_seconds)
    after = tuple(token for token in result.tokens if token.start_seconds >= at_seconds)
    reasons: list[str] = []
    if any(
        token.start_seconds + 0.04 < at_seconds < token.end_seconds - 0.04
        for token in result.tokens
    ):
        reasons.append("word_crosses_join")
    if before and after and before[-1].text == after[0].text:
        reasons.append("repeated_join_token")
    if (
        len(before) >= _JOIN_PHRASE_WORDS
        and len(after) >= _JOIN_PHRASE_WORDS
        and tuple(token.text for token in before[-_JOIN_PHRASE_WORDS:])
        == tuple(token.text for token in after[:_JOIN_PHRASE_WORDS])
    ):
        reasons.append("repeated_join_phrase")
    return tuple(reasons)


def _edge_silence_ms(
    samples: tuple[float, ...],
    threshold: float,
    sample_rate: int,
    *,
    reverse: bool,
) -> float:
    ordered = reversed(samples) if reverse else iter(samples)
    count = 0
    for value in ordered:
        if abs(value) >= threshold:
            break
        count += 1
    return count * 1_000 / sample_rate


def _decode_pcm_sample(content: bytes, width: int) -> int:
    if width == 1:
        return content[0] - 128
    return int.from_bytes(content, "little", signed=True)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
