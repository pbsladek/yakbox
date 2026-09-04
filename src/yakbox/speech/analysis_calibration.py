"""Deterministic fitting of engine- and clip-specific decode-quality gates."""

from __future__ import annotations

from dataclasses import dataclass

from yakbox.errors import ValidationError
from yakbox.speech.analysis_models import (
    ClipClass,
    ParakeetEvidence,
    QwenEvidence,
    RecognitionResult,
    WhisperEvidence,
)
from yakbox.speech.analysis_policy import (
    CalibrationTable,
    CalibrationThreshold,
    recognition_quality_issues,
)


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    """One corpus-labeled determination that a decode is usable or unusable."""

    case_id: str
    clip_class: ClipClass
    result: RecognitionResult
    decode_usable: bool

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValidationError("Calibration observation requires a case ID")


def fit_calibration_table(
    observations: tuple[CalibrationObservation, ...],
    *,
    corpus_fingerprint: str,
    required_keys: tuple[tuple[str, ClipClass], ...],
    version: int = 1,
) -> CalibrationTable:
    """Fit conservative envelopes and reject empirically inseparable groups.

    Every usable decode remains inside its engine/class envelope, while every
    labeled unusable decode must fall outside it. Fitting fails closed instead
    of manufacturing a threshold from another model.
    """
    if not observations or not required_keys:
        raise ValidationError("Calibration fitting requires observations and keys")
    keys = tuple(sorted(set(required_keys), key=lambda item: (item[0], item[1].value)))
    if len(keys) != len(required_keys):
        raise ValidationError("Calibration fitting keys must be unique")
    case_ids = tuple(item.case_id for item in observations)
    if len(case_ids) != len(set(case_ids)):
        raise ValidationError("Calibration observation case IDs must be unique")
    observed_keys = {(item.result.engine, item.clip_class) for item in observations}
    if observed_keys != set(keys):
        raise ValidationError("Calibration observations do not match required keys")
    thresholds = tuple(
        _fit_threshold(
            engine,
            clip_class,
            tuple(
                item
                for item in observations
                if item.result.engine == engine and item.clip_class is clip_class
            ),
        )
        for engine, clip_class in keys
    )
    executions = tuple(
        sorted({item.result.execution.fingerprint for item in observations})
    )
    return CalibrationTable(
        version=version,
        language="en",
        corpus_fingerprint=corpus_fingerprint,
        approved=False,
        reviewer_disposition_fingerprint=None,
        execution_class_fingerprints=executions,
        thresholds=thresholds,
    )


def _fit_threshold(
    engine: str,
    clip_class: ClipClass,
    observations: tuple[CalibrationObservation, ...],
) -> CalibrationThreshold:
    usable = tuple(item.result for item in observations if item.decode_usable)
    unusable = tuple(item.result for item in observations if not item.decode_usable)
    if not usable or not unusable:
        raise ValidationError(
            f"Calibration for {engine!r}/{clip_class.value!r} requires usable and "
            "unusable observations"
        )
    evidence_types = {type(item.evidence) for item in usable + unusable}
    if len(evidence_types) != 1:
        raise ValidationError("Calibration group mixes incompatible evidence types")
    score_calibrations = {
        item.score_calibration_fingerprint for item in usable + unusable
    }
    if len(score_calibrations) != 1:
        raise ValidationError("Calibration group mixes score calibration identities")
    threshold = _threshold_from_usable(engine, clip_class, usable)
    if any(not recognition_quality_issues(item, threshold) for item in unusable):
        raise ValidationError(
            f"Calibration evidence cannot separate {engine!r}/{clip_class.value!r}"
        )
    return threshold


def _threshold_from_usable(
    engine: str,
    clip_class: ClipClass,
    usable: tuple[RecognitionResult, ...],
) -> CalibrationThreshold:
    token_scores = tuple(
        token.score
        for item in usable
        for token in item.tokens
        if token.score is not None
    )
    all_tokens_scored = all(
        item.tokens and all(token.score is not None for token in item.tokens)
        for item in usable
    )
    evidence = usable[0].evidence
    if isinstance(evidence, WhisperEvidence):
        values = tuple(_whisper_metrics(item) for item in usable)
        return CalibrationThreshold(
            engine=engine,
            clip_class=clip_class,
            score_calibration_fingerprint=usable[0].score_calibration_fingerprint,
            minimum_token_score=min(token_scores) if all_tokens_scored else None,
            minimum_average_log_probability=min(item[0] for item in values),
            maximum_compression_ratio=max(item[1] for item in values),
            maximum_no_speech_probability=max(item[2] for item in values),
            maximum_temperature=max(item[3] for item in values),
        )
    if isinstance(evidence, ParakeetEvidence):
        scores = tuple(_parakeet_score(item) for item in usable)
        return CalibrationThreshold(
            engine=engine,
            clip_class=clip_class,
            score_calibration_fingerprint=usable[0].score_calibration_fingerprint,
            minimum_token_score=min(token_scores) if all_tokens_scored else None,
            minimum_sentence_score=min(scores),
        )
    if isinstance(evidence, QwenEvidence):
        reasons = tuple(sorted({_qwen_finish_reason(item) for item in usable}))
        return CalibrationThreshold(
            engine=engine,
            clip_class=clip_class,
            score_calibration_fingerprint=usable[0].score_calibration_fingerprint,
            minimum_token_score=min(token_scores) if all_tokens_scored else None,
            allowed_finish_reasons=reasons,
        )
    raise ValidationError("Calibration evidence type is unsupported")


def _whisper_metrics(result: RecognitionResult) -> tuple[float, float, float, float]:
    evidence = result.evidence
    if not isinstance(evidence, WhisperEvidence):
        raise ValidationError("Whisper calibration evidence is inconsistent")
    return (
        _required_metric(evidence.average_log_probability, "average log probability"),
        _required_metric(evidence.compression_ratio, "compression ratio"),
        _required_metric(evidence.no_speech_probability, "no-speech probability"),
        _required_metric(evidence.temperature, "temperature"),
    )


def _parakeet_score(result: RecognitionResult) -> float:
    evidence = result.evidence
    if not isinstance(evidence, ParakeetEvidence):
        raise ValidationError("Parakeet calibration evidence is inconsistent")
    return _required_metric(evidence.sentence_confidence, "sentence confidence")


def _qwen_finish_reason(result: RecognitionResult) -> str:
    evidence = result.evidence
    if not isinstance(evidence, QwenEvidence):
        raise ValidationError("Qwen calibration evidence is inconsistent")
    return evidence.finish_reason


def _required_metric(value: float | None, label: str) -> float:
    if value is None:
        raise ValidationError(f"Usable calibration lacks {label}")
    return value


__all__ = ["CalibrationObservation", "fit_calibration_table"]
