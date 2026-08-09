"""Whisper-backed chapter, join, and clip-specific QA services."""

from __future__ import annotations

import hashlib
import math
import wave
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from enum import StrEnum
from pathlib import Path

from yakbox.audio.crop import wav_duration_seconds
from yakbox.contracts import runtime_metadata
from yakbox.errors import ArtifactError, ValidationError
from yakbox.local_alignment import MlxWhisperAligner
from yakbox.speech.alignment import (
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
    if clip_type is not WhisperClipType.ONE_WORD:
        maximum_gap = max(
            (
                following.start_seconds - previous.end_seconds
                for previous, following in zip(
                    result.tokens,
                    result.tokens[1:],
                    strict=False,
                )
            ),
            default=0.0,
        )
        if maximum_gap * 1_000 > profile.maximum_internal_gap_ms:
            reasons.append("excessive_internal_pause")
    return AlignmentEvaluation(
        clip_type=clip_type,
        accepted=not reasons,
        reason_codes=tuple(dict.fromkeys(reasons)),
        minimum_word_confidence=minimum,
        profile=profile,
    )


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
    result = await cached_aligner.align(audio, expected_text, language=language)
    expected = canonical_tokens(_expanded_lexical_tokens(expected_text), token_aliases)
    comparison_tokens = tuple(
        replace(token, text=canonical)
        for token in result.tokens
        for canonical in canonical_tokens(
            _expanded_lexical_tokens(token.text), token_aliases
        )
    )
    expected, comparison_tokens = _coalesce_compound_equivalents(
        expected, comparison_tokens
    )
    comparison_tokens, ignored_insertions = _remove_decoder_insertions(
        expected, comparison_tokens
    )
    comparison_result = replace(result, tokens=comparison_tokens)
    recognized = tuple(token.text for token in comparison_tokens)
    matcher = SequenceMatcher(a=expected, b=recognized, autojunk=False)
    matching = sum(block.size for block in matcher.get_matching_blocks())
    denominator = max(1, len(expected), len(recognized))
    mismatches = tuple(
        _manuscript_mismatch(expected, comparison_result, opcode)
        for opcode in matcher.get_opcodes()
        if opcode[0] != "equal"
    )
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
    )


def _expanded_lexical_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        part for token in lexical_tokens(text) for part in token.split("-") if part
    )


def _coalesce_compound_equivalents(
    expected: tuple[str, ...],
    recognized: tuple[AlignmentToken, ...],
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
                and "".join(expected_span) == recognized_span[0].text
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
                and expected_span[0] == "".join(token.text for token in recognized_span)
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
