"""Baseline-calibrated Whisper qualification for synthesized voice auditions."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from yakbox._files import sha256_file
from yakbox.audio.crop import (
    inspect_signal_quality,
    wav_duration_seconds,
)
from yakbox.contracts import runtime_metadata
from yakbox.errors import ValidationError
from yakbox.local_alignment import MlxWhisperAligner
from yakbox.speech.alignment import (
    AlignmentResult,
    SpeechAligner,
    WindowSpeechAligner,
    lexical_tokens,
)
from yakbox.whisper_cache import CachedWhisperAligner

_MINIMUM_BASELINE_VOICES = 2
_SHA256_LENGTH = 64


@dataclass(frozen=True, slots=True)
class VoiceQualityMetrics:
    """ASR intelligibility and waveform evidence for one fixed-text audition."""

    duration_seconds: float
    expected_token_count: int
    recognized_token_count: int
    token_error_count: int
    token_accuracy: float
    mean_word_confidence: float
    minimum_word_confidence: float
    decode_consensus: float
    minimum_average_log_probability: float
    maximum_compression_ratio: float
    maximum_no_speech_probability: float
    parser_issue_count: int
    peak_dbfs: float
    clipped_sample_ratio: float
    maximum_boundary_jump_ratio: float
    vad_disagreement_ratio: float
    longest_stationary_voiced_ms: float
    leading_silence_ms: float
    trailing_silence_ms: float
    high_frequency_energy_ratio: float
    words_per_minute: float


@dataclass(frozen=True, slots=True)
class VoiceQualityThresholds:
    """Gates derived from a human-approved baseline cohort."""

    minimum_token_accuracy: float
    minimum_mean_word_confidence: float
    minimum_decode_consensus: float
    minimum_average_log_probability: float
    maximum_compression_ratio: float
    maximum_no_speech_probability: float
    minimum_peak_dbfs: float
    maximum_clipped_sample_ratio: float
    maximum_boundary_jump_ratio: float
    maximum_vad_disagreement_ratio: float
    maximum_stationary_voiced_ms: float
    maximum_leading_silence_ms: float
    maximum_trailing_silence_ms: float


@dataclass(frozen=True, slots=True)
class VoiceQualityResult:
    """One voice's qualification decision and supporting evidence."""

    voice: str
    audio: Path
    audio_sha256: str
    baseline: bool
    status: str
    reason_codes: tuple[str, ...]
    metrics: VoiceQualityMetrics
    prompt_sensitivity: str
    consensus_reason_codes: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.status in {"baseline", "high_quality"}

    def to_dict(self, *, root: Path | None = None) -> dict[str, object]:
        audio = (
            self.audio.relative_to(root).as_posix()
            if root is not None and self.audio.is_relative_to(root)
            else str(self.audio)
        )
        return {
            "voice": self.voice,
            "audio": audio,
            "audio_sha256": self.audio_sha256,
            "baseline": self.baseline,
            "status": self.status,
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "metrics": asdict(self.metrics),
            "prompt_sensitivity": self.prompt_sensitivity,
            "consensus_reason_codes": list(self.consensus_reason_codes),
        }


@dataclass(frozen=True, slots=True)
class VoiceQualityReport:
    """Versioned qualification report for one comparable audition cohort."""

    audition_report: Path
    expected_text_sha256: str
    expected_token_count: int
    baseline_voices: tuple[str, ...]
    model: str
    alignment_fingerprint: str
    thresholds: VoiceQualityThresholds
    voices: tuple[VoiceQualityResult, ...]

    @property
    def accepted(self) -> bool:
        return all(item.accepted for item in self.voices)

    def to_dict(self) -> dict[str, object]:
        suspect = tuple(item.voice for item in self.voices if not item.accepted)
        evidence_root = self.audition_report.parent
        return {
            **runtime_metadata("voice-quality"),
            "audition_report": self.audition_report.name,
            "expected_text_sha256": self.expected_text_sha256,
            "expected_token_count": self.expected_token_count,
            "baseline_voices": list(self.baseline_voices),
            "model": self.model,
            "alignment_fingerprint": self.alignment_fingerprint,
            "scope": "automated_intelligibility_and_signal_quality",
            "accepted": self.accepted,
            "baseline_count": sum(item.baseline for item in self.voices),
            "high_quality_count": sum(
                item.status == "high_quality" for item in self.voices
            ),
            "suspect_count": len(suspect),
            "suspect_voices": list(suspect),
            "thresholds": asdict(self.thresholds),
            "voices": [item.to_dict(root=evidence_root) for item in self.voices],
        }


async def qualify_audition_voices(
    audition_report: Path,
    expected_text: str,
    *,
    baseline_voices: tuple[str, ...],
    language: str,
    model: str,
    revision: str | None,
    aligner: SpeechAligner | None = None,
    cache_root: Path | None = None,
) -> VoiceQualityReport:
    """Compare fixed-text synthesized auditions to approved reference voices."""
    expected = lexical_tokens(expected_text)
    if not expected:
        raise ValidationError("Voice qualification text must contain spoken words")
    samples, recorded_text_hash = load_audition_samples(audition_report)
    expected_hash = hashlib.sha256(expected_text.encode()).hexdigest()
    if recorded_text_hash is not None and recorded_text_hash != expected_hash:
        raise ValidationError(
            "The expected text does not match the audition report input hash"
        )
    baselines = tuple(dict.fromkeys(baseline_voices))
    if len(baselines) < _MINIMUM_BASELINE_VOICES:
        raise ValidationError("At least two baseline voices are required")
    missing = tuple(item for item in baselines if item not in samples)
    if missing:
        raise ValidationError(
            f"Baseline voices are absent from the audition: {', '.join(missing)}"
        )
    resolved_aligner = aligner or MlxWhisperAligner(
        model=model,
        revision=revision,
        prompted_timing=False,
        prompt_sensitivity=True,
        decode_consensus=True,
    )
    cached: SpeechAligner = (
        CachedWhisperAligner(cast(WindowSpeechAligner, resolved_aligner), cache_root)
        if cache_root is not None
        else resolved_aligner
    )
    evidence: dict[str, tuple[VoiceQualityMetrics, AlignmentResult]] = {}
    for voice, audio in samples.items():
        result = await cached.align(audio, expected_text, language=language)
        evidence[voice] = (_quality_metrics(audio, expected, result), result)
    thresholds = _derive_thresholds(tuple(evidence[voice][0] for voice in baselines))
    results = tuple(
        _quality_result(
            voice,
            samples[voice],
            metrics,
            result,
            thresholds,
            baseline=voice in baselines,
        )
        for voice, (metrics, result) in evidence.items()
    )
    return VoiceQualityReport(
        audition_report=audition_report.resolve(),
        expected_text_sha256=expected_hash,
        expected_token_count=len(expected),
        baseline_voices=baselines,
        model=model,
        alignment_fingerprint=resolved_aligner.fingerprint,
        thresholds=thresholds,
        voices=results,
    )


def load_audition_samples(path: Path) -> tuple[dict[str, Path], str | None]:
    """Load profile-named WAVs from a Yakbox audition report."""
    raw = _load_audition_report(path)
    comparisons = cast(list[object], raw["comparisons"])
    samples: dict[str, Path] = {}
    for index, raw_item in enumerate(comparisons, start=1):
        voice, audio = _audition_sample(path, raw_item, index=index)
        if voice in samples:
            raise ValidationError(f"Audition contains duplicate variant: {voice}")
        samples[voice] = audio
    if not samples:
        raise ValidationError("Audition report does not contain any voice samples")
    text_hash = _optional_sha256(raw.get("input_text_sha256"))
    return samples, text_hash


def _load_audition_report(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"Cannot read audition report {path}: {error}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("comparisons"), list):
        raise ValidationError("Audition report must contain a comparisons array")
    return cast(dict[str, object], raw)


def _audition_sample(path: Path, raw: object, *, index: int) -> tuple[str, Path]:
    if not isinstance(raw, dict):
        raise ValidationError(f"Audition comparison {index} must be an object")
    item = cast(dict[str, object], raw)
    voice = item.get("variant")
    artifact_value = item.get("artifact")
    if not isinstance(voice, str) or not voice:
        raise ValidationError(f"Audition comparison {index}.variant is invalid")
    if not isinstance(artifact_value, dict):
        raise ValidationError(f"Audition comparison {index}.artifact is invalid")
    artifact = cast(dict[str, object], artifact_value)
    artifact_path = artifact.get("path")
    if not isinstance(artifact_path, str):
        raise ValidationError(f"Audition comparison {index}.artifact.path is invalid")
    audio = path.parent / Path(artifact_path).name
    if not audio.is_file():
        raise ValidationError(f"Audition audio does not exist: {audio}")
    return voice, audio.resolve()


def _optional_sha256(value: object) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValidationError("Audition report input_text_sha256 is invalid")
    return value


def _quality_metrics(
    audio: Path,
    expected: tuple[str, ...],
    result: AlignmentResult,
) -> VoiceQualityMetrics:
    duration = wav_duration_seconds(audio)
    recognized = tuple(token.text.casefold() for token in result.tokens)
    confidences = tuple(
        token.confidence for token in result.tokens if token.confidence is not None
    )
    segments = result.segments
    signal = inspect_signal_quality(audio)
    errors = _edit_distance(expected, recognized)
    first = result.tokens[0].start_seconds if result.tokens else duration
    last = result.tokens[-1].end_seconds if result.tokens else 0.0
    return VoiceQualityMetrics(
        duration_seconds=duration,
        expected_token_count=len(expected),
        recognized_token_count=len(recognized),
        token_error_count=errors,
        token_accuracy=max(0.0, 1.0 - errors / len(expected)),
        mean_word_confidence=(
            statistics.fmean(confidences)
            if len(confidences) == len(result.tokens) and confidences
            else 0.0
        ),
        minimum_word_confidence=(
            min(confidences)
            if len(confidences) == len(result.tokens) and confidences
            else 0.0
        ),
        decode_consensus=result.consensus_score or 0.0,
        minimum_average_log_probability=_minimum_segment_value(
            segments, "average_log_probability", default=-10.0
        ),
        maximum_compression_ratio=_maximum_segment_value(
            segments, "compression_ratio", default=10.0
        ),
        maximum_no_speech_probability=_maximum_segment_value(
            segments, "no_speech_probability", default=1.0
        ),
        parser_issue_count=len(result.issues),
        peak_dbfs=signal.peak_dbfs,
        clipped_sample_ratio=signal.clipped_sample_ratio,
        maximum_boundary_jump_ratio=max(
            signal.leading_boundary_jump_ratio,
            signal.trailing_boundary_jump_ratio,
        ),
        vad_disagreement_ratio=(signal.vad_disagreement_ms / (duration * 1_000)),
        longest_stationary_voiced_ms=signal.longest_stationary_voiced_ms,
        leading_silence_ms=max(0.0, first) * 1_000,
        trailing_silence_ms=max(0.0, duration - last) * 1_000,
        high_frequency_energy_ratio=signal.high_frequency_energy_ratio,
        words_per_minute=len(recognized) / duration * 60,
    )


def _derive_thresholds(
    baseline: tuple[VoiceQualityMetrics, ...],
) -> VoiceQualityThresholds:
    if len(baseline) < _MINIMUM_BASELINE_VOICES:
        raise ValidationError("At least two baseline measurements are required")
    return VoiceQualityThresholds(
        minimum_token_accuracy=_lower_threshold(
            baseline, "token_accuracy", floor=0.90, tolerance=0.01, margin=0.04
        ),
        minimum_mean_word_confidence=_lower_threshold(
            baseline,
            "mean_word_confidence",
            floor=0.80,
            tolerance=0.03,
            margin=0.08,
        ),
        minimum_decode_consensus=_lower_threshold(
            baseline,
            "decode_consensus",
            floor=0.90,
            tolerance=0.01,
            margin=0.04,
        ),
        minimum_average_log_probability=_lower_threshold(
            baseline,
            "minimum_average_log_probability",
            floor=-0.75,
            tolerance=0.05,
            margin=0.15,
        ),
        maximum_compression_ratio=_upper_threshold(
            baseline,
            "maximum_compression_ratio",
            ceiling=2.4,
            tolerance=0.1,
            margin=0.3,
        ),
        maximum_no_speech_probability=_upper_threshold(
            baseline,
            "maximum_no_speech_probability",
            ceiling=0.20,
            tolerance=0.02,
            margin=0.08,
        ),
        minimum_peak_dbfs=_lower_threshold(
            baseline, "peak_dbfs", floor=-18.0, tolerance=3.0, margin=6.0
        ),
        maximum_clipped_sample_ratio=_upper_threshold(
            baseline,
            "clipped_sample_ratio",
            ceiling=0.00002,
            tolerance=0.000001,
            margin=0.000005,
        ),
        maximum_boundary_jump_ratio=_upper_threshold(
            baseline,
            "maximum_boundary_jump_ratio",
            ceiling=0.20,
            tolerance=0.03,
            margin=0.08,
        ),
        maximum_vad_disagreement_ratio=_upper_threshold(
            baseline,
            "vad_disagreement_ratio",
            ceiling=0.15,
            tolerance=0.02,
            margin=0.05,
        ),
        maximum_stationary_voiced_ms=_upper_threshold(
            baseline,
            "longest_stationary_voiced_ms",
            ceiling=1_000,
            tolerance=100,
            margin=300,
        ),
        maximum_leading_silence_ms=_upper_threshold(
            baseline,
            "leading_silence_ms",
            ceiling=1_500,
            tolerance=200,
            margin=500,
        ),
        maximum_trailing_silence_ms=_upper_threshold(
            baseline,
            "trailing_silence_ms",
            ceiling=1_500,
            tolerance=200,
            margin=500,
        ),
    )


def _quality_result(
    voice: str,
    audio: Path,
    metrics: VoiceQualityMetrics,
    result: AlignmentResult,
    thresholds: VoiceQualityThresholds,
    *,
    baseline: bool,
) -> VoiceQualityResult:
    reasons = _reason_codes(metrics, thresholds)
    status = (
        "baseline_invalid"
        if baseline and reasons
        else "baseline"
        if baseline
        else "suspect"
        if reasons
        else "high_quality"
    )
    return VoiceQualityResult(
        voice=voice,
        audio=audio,
        audio_sha256=sha256_file(audio),
        baseline=baseline,
        status=status,
        reason_codes=reasons,
        metrics=metrics,
        prompt_sensitivity=result.prompt_sensitivity,
        consensus_reason_codes=result.consensus_reason_codes,
    )


def _reason_codes(
    metrics: VoiceQualityMetrics,
    thresholds: VoiceQualityThresholds,
) -> tuple[str, ...]:
    checks = (
        (
            metrics.token_accuracy < thresholds.minimum_token_accuracy,
            "transcript_accuracy_below_baseline",
        ),
        (
            metrics.mean_word_confidence < thresholds.minimum_mean_word_confidence,
            "word_confidence_below_baseline",
        ),
        (
            metrics.decode_consensus < thresholds.minimum_decode_consensus,
            "decode_consensus_below_baseline",
        ),
        (
            metrics.minimum_average_log_probability
            < thresholds.minimum_average_log_probability,
            "segment_log_probability_below_baseline",
        ),
        (
            metrics.maximum_compression_ratio > thresholds.maximum_compression_ratio,
            "compression_ratio_above_baseline",
        ),
        (
            metrics.maximum_no_speech_probability
            > thresholds.maximum_no_speech_probability,
            "no_speech_probability_above_baseline",
        ),
        (
            metrics.peak_dbfs < thresholds.minimum_peak_dbfs,
            "peak_level_below_baseline",
        ),
        (
            metrics.clipped_sample_ratio > thresholds.maximum_clipped_sample_ratio,
            "clipping_above_baseline",
        ),
        (
            metrics.maximum_boundary_jump_ratio
            > thresholds.maximum_boundary_jump_ratio,
            "boundary_jump_above_baseline",
        ),
        (
            metrics.vad_disagreement_ratio > thresholds.maximum_vad_disagreement_ratio,
            "vad_disagreement_above_baseline",
        ),
        (
            metrics.longest_stationary_voiced_ms
            > thresholds.maximum_stationary_voiced_ms,
            "stationary_voice_above_baseline",
        ),
        (
            metrics.leading_silence_ms > thresholds.maximum_leading_silence_ms,
            "leading_silence_above_baseline",
        ),
        (
            metrics.trailing_silence_ms > thresholds.maximum_trailing_silence_ms,
            "trailing_silence_above_baseline",
        ),
    )
    return tuple(reason for failed, reason in checks if failed)


def _lower_threshold(
    values: tuple[VoiceQualityMetrics, ...],
    field: str,
    *,
    floor: float,
    tolerance: float,
    margin: float,
) -> float:
    measured = tuple(float(getattr(item, field)) for item in values)
    return min(
        min(measured) - tolerance, max(floor, statistics.median(measured) - margin)
    )


def _upper_threshold(
    values: tuple[VoiceQualityMetrics, ...],
    field: str,
    *,
    ceiling: float,
    tolerance: float,
    margin: float,
) -> float:
    measured = tuple(float(getattr(item, field)) for item in values)
    return max(
        max(measured) + tolerance, min(ceiling, statistics.median(measured) + margin)
    )


def _minimum_segment_value(
    segments: tuple[object, ...], field: str, *, default: float
) -> float:
    values = tuple(
        float(value)
        for segment in segments
        if (value := getattr(segment, field, None)) is not None
        and math.isfinite(float(value))
    )
    return min(values, default=default)


def _maximum_segment_value(
    segments: tuple[object, ...], field: str, *, default: float
) -> float:
    values = tuple(
        float(value)
        for segment in segments
        if (value := getattr(segment, field, None)) is not None
        and math.isfinite(float(value))
    )
    return max(values, default=default)


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


__all__ = [
    "VoiceQualityMetrics",
    "VoiceQualityReport",
    "VoiceQualityResult",
    "VoiceQualityThresholds",
    "load_audition_samples",
    "qualify_audition_voices",
]
