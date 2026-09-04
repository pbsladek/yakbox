"""Typed policy and calibration inputs for multi-model speech analysis."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from yakbox.errors import ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_models import (
    ClipClass,
    ParakeetEvidence,
    QwenEvidence,
    RecognitionResult,
    WhisperEvidence,
)

_BASELINE_RECOGNIZER_COUNT = 2
_MINIMUM_ONE_WORD_RECOGNIZER_COUNT = 2
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class EnginePolicy:
    """Immutable adapter and decode settings for one registered engine."""

    engine: str
    backend: str
    model: str
    revision: str
    timeout_seconds: float
    decode_mode: str
    chunk_seconds: int | None = None
    overlap_seconds: int | None = None
    maximum_window_seconds: int | None = None

    def __post_init__(self) -> None:
        if not all((self.engine, self.backend, self.model, self.revision)):
            raise ValidationError("Speech-analysis engine policy is incomplete")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValidationError("Speech-analysis timeout must be positive")
        if self.chunk_seconds is not None and self.chunk_seconds <= 0:
            raise ValidationError("Analysis chunk duration must be positive")
        if self.overlap_seconds is not None and self.overlap_seconds < 0:
            raise ValidationError("Analysis overlap cannot be negative")
        if self.maximum_window_seconds is not None and self.maximum_window_seconds <= 0:
            raise ValidationError("Analysis maximum window must be positive")
        if (
            self.chunk_seconds is not None
            and self.overlap_seconds is not None
            and self.overlap_seconds >= self.chunk_seconds
        ):
            raise ValidationError("Analysis overlap must be shorter than its chunk")


@dataclass(frozen=True, slots=True)
class SpeechAnalysisPolicy:
    """Expanded canonical decision policy used before fingerprinting."""

    version: int
    preset: str
    language: str
    baseline_recognizers: tuple[str, ...]
    escalation_recognizer: str
    forced_aligner: str
    always_escalate_clip_classes: tuple[ClipClass, ...]
    always_escalate_repairs: bool
    reject_unresolved_disagreement: bool
    reject_unexpected_speech: bool
    missing_required_engine: str
    valid_dissent: str
    engines: tuple[EnginePolicy, ...]
    one_word_required_recognizers: tuple[str, ...] = ("whisper", "qwen")

    def __post_init__(self) -> None:
        if self.version < 1 or self.language != "en":
            raise ValidationError("Strict speech analysis is currently English-only")
        if len(self.baseline_recognizers) != _BASELINE_RECOGNIZER_COUNT:
            raise ValidationError(
                "Strict policy requires exactly two baseline recognizers"
            )
        if len(set(self.baseline_recognizers)) != _BASELINE_RECOGNIZER_COUNT:
            raise ValidationError("Baseline recognizers must be distinct")
        names = tuple(engine.engine for engine in self.engines)
        required = (
            *self.baseline_recognizers,
            self.escalation_recognizer,
            self.forced_aligner,
        )
        if any(name not in names for name in required):
            raise ValidationError("Speech-analysis policy references an unknown engine")
        if len(names) != len(set(names)):
            raise ValidationError("Speech-analysis engine policies must be unique")
        if (
            len(self.one_word_required_recognizers) < _MINIMUM_ONE_WORD_RECOGNIZER_COUNT
            or len(set(self.one_word_required_recognizers))
            != len(self.one_word_required_recognizers)
            or any(name not in names for name in self.one_word_required_recognizers)
        ):
            raise ValidationError(
                "One-word analysis requires distinct configured recognizers"
            )
        if self.missing_required_engine != "error":
            raise ValidationError(
                "Strict policy must fail when required evidence is missing"
            )
        if self.valid_dissent != "retry_then_reject":
            raise ValidationError("Strict policy must reject persistent valid dissent")
        if not self.reject_unresolved_disagreement or not self.reject_unexpected_speech:
            raise ValidationError(
                "Strict policy must reject unresolved and unexpected speech"
            )

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-analysis-policy-v1", self)


@dataclass(frozen=True, slots=True)
class CalibrationThreshold:
    """Engine- and clip-specific decode-quality gate."""

    engine: str
    clip_class: ClipClass
    score_calibration_fingerprint: str
    minimum_token_score: float | None = None
    minimum_sentence_score: float | None = None
    minimum_average_log_probability: float | None = None
    maximum_compression_ratio: float | None = None
    maximum_no_speech_probability: float | None = None
    maximum_temperature: float | None = None
    allowed_finish_reasons: tuple[str, ...] = ("stop", "eos", "complete")

    def __post_init__(self) -> None:
        values = (
            self.minimum_token_score,
            self.minimum_sentence_score,
            self.minimum_average_log_probability,
            self.maximum_compression_ratio,
            self.maximum_no_speech_probability,
            self.maximum_temperature,
        )
        if not self.engine or any(
            value is not None and not math.isfinite(value) for value in values
        ):
            raise ValidationError("Calibration thresholds must be finite")
        _require_sha256(
            self.score_calibration_fingerprint,
            "threshold score calibration fingerprint",
        )


@dataclass(frozen=True, slots=True)
class CalibrationTable:
    """Versioned thresholds qualified for explicit execution classes."""

    version: int
    language: str
    corpus_fingerprint: str
    approved: bool
    reviewer_disposition_fingerprint: str | None
    execution_class_fingerprints: tuple[str, ...]
    thresholds: tuple[CalibrationThreshold, ...]

    def __post_init__(self) -> None:
        if self.version < 1 or self.language != "en" or not self.thresholds:
            raise ValidationError("Calibration table must be versioned for English")
        _require_sha256(self.corpus_fingerprint, "calibration corpus fingerprint")
        if not self.execution_class_fingerprints:
            raise ValidationError(
                "Calibration requires at least one qualified execution class"
            )
        for fingerprint in self.execution_class_fingerprints:
            _require_sha256(fingerprint, "calibration execution fingerprint")
        if self.approved and self.reviewer_disposition_fingerprint is None:
            raise ValidationError(
                "Approved calibration requires a reviewer disposition fingerprint"
            )
        if self.reviewer_disposition_fingerprint is not None:
            _require_sha256(
                self.reviewer_disposition_fingerprint,
                "calibration reviewer disposition fingerprint",
            )
        keys = tuple((item.engine, item.clip_class) for item in self.thresholds)
        if len(keys) != len(set(keys)):
            raise ValidationError(
                "Calibration thresholds must be unique per engine/class"
            )

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "speech-calibration-thresholds-v1",
            {
                "version": self.version,
                "language": self.language,
                "corpus_fingerprint": self.corpus_fingerprint,
                "execution_class_fingerprints": (self.execution_class_fingerprints),
                "thresholds": self.thresholds,
            },
        )

    @property
    def disposition_fingerprint(self) -> str:
        return semantic_fingerprint(
            "speech-calibration-disposition-v1",
            {
                "threshold_fingerprint": self.fingerprint,
                "approved": self.approved,
                "reviewer_disposition_fingerprint": (
                    self.reviewer_disposition_fingerprint
                ),
            },
        )

    def threshold(self, engine: str, clip_class: ClipClass) -> CalibrationThreshold:
        matches = tuple(
            item
            for item in self.thresholds
            if item.engine == engine and item.clip_class is clip_class
        )
        if len(matches) != 1:
            raise ValidationError(
                f"Missing calibration for {engine!r} and {clip_class.value!r}"
            )
        return matches[0]


def recognition_quality_issues(
    result: RecognitionResult,
    threshold: CalibrationThreshold,
) -> tuple[str, ...]:
    """Apply only the threshold fields meaningful to the result's engine."""
    if result.issues:
        return result.issues
    if threshold.engine != result.engine:
        raise ValidationError("Calibration engine does not match recognition evidence")
    invalid = _token_score_invalid(result, threshold)
    evidence = result.evidence
    if isinstance(evidence, WhisperEvidence):
        invalid = invalid or _whisper_invalid(evidence, threshold)
    elif isinstance(evidence, ParakeetEvidence):
        invalid = invalid or _parakeet_invalid(evidence, threshold)
    elif isinstance(evidence, QwenEvidence):
        invalid = (
            invalid or evidence.finish_reason not in threshold.allowed_finish_reasons
        )
    return ("engine_decode_invalid",) if invalid else ()


def _token_score_invalid(
    result: RecognitionResult, threshold: CalibrationThreshold
) -> bool:
    if threshold.minimum_token_score is None:
        return False
    scores = tuple(token.score for token in result.tokens)
    return not scores or any(
        score is None or score < threshold.minimum_token_score for score in scores
    )


def _whisper_invalid(
    evidence: WhisperEvidence, threshold: CalibrationThreshold
) -> bool:
    pairs = (
        (
            evidence.average_log_probability,
            threshold.minimum_average_log_probability,
            lambda actual, limit: actual < limit,
        ),
        (
            evidence.compression_ratio,
            threshold.maximum_compression_ratio,
            lambda actual, limit: actual > limit,
        ),
        (
            evidence.no_speech_probability,
            threshold.maximum_no_speech_probability,
            lambda actual, limit: actual > limit,
        ),
        (
            evidence.temperature,
            threshold.maximum_temperature,
            lambda actual, limit: actual > limit,
        ),
    )
    return any(
        limit is not None and (actual is None or compare(actual, limit))
        for actual, limit, compare in pairs
    )


def _parakeet_invalid(
    evidence: ParakeetEvidence, threshold: CalibrationThreshold
) -> bool:
    return threshold.minimum_sentence_score is not None and (
        evidence.sentence_confidence is None
        or evidence.sentence_confidence < threshold.minimum_sentence_score
    )


def _require_sha256(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label.capitalize()} must be a SHA-256")


__all__ = [
    "CalibrationTable",
    "CalibrationThreshold",
    "EnginePolicy",
    "SpeechAnalysisPolicy",
    "recognition_quality_issues",
]
