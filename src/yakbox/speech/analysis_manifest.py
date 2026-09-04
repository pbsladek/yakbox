"""Internal parser for the draft version-2 speech-analysis manifest shape."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from yakbox.errors import ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_models import ClipClass
from yakbox.speech.analysis_policy import EnginePolicy, SpeechAnalysisPolicy
from yakbox.speech.model_registry import ModelRegistryData, load_model_registry

_ENGINE_NAMES = ("whisper", "parakeet", "qwen", "qwen-forced")
_DRAFT_MANIFEST_VERSION = 2
_DRAFT_CACHE_VERSION = 2
_SPEECH_KEYS = {
    "preset",
    "baseline_recognizers",
    "escalation_recognizer",
    "forced_aligner",
    "always_escalate_clip_classes",
    "always_escalate_repairs",
    "engines",
    "consensus",
    "cache",
    "short_utterances",
    "joins",
    "chapters",
    "phonemes",
    "repairs",
    "releases",
}
_ENGINE_KEYS = {
    "backend",
    "model",
    "revision",
    "timeout_seconds",
    "decode_mode",
    "chunk_seconds",
    "overlap_seconds",
    "maximum_window_seconds",
}

_CACHE_KEYS = {"version", "enabled", "directory", "legacy_evidence"}
_SHORT_KEYS = {
    "maximum_words",
    "candidate_count",
    "maximum_rounds",
    "require_independent_crop_verification",
}
_JOIN_KEYS = {"automatic_inspection", "window_seconds", "coalesce_gap_ms"}
_CHAPTER_KEYS = {"verification"}
_PHONEME_KEYS = {
    "enabled",
    "backend",
    "model",
    "revision",
    "language",
    "timeout_seconds",
    "minimum_confidence",
}
_REPAIR_KEYS = {
    "verification",
    "approval_output",
    "candidate_verification_scope",
    "candidates_per_round",
    "maximum_rounds",
}
_RELEASE_KEYS = {
    "complete_master_recognition",
    "verify_decoded_delivery",
    "codec_equivalence",
}


@dataclass(frozen=True, slots=True)
class DraftAnalysisCachePolicy:
    """Versioned cache namespace and explicit legacy-evidence disposition."""

    version: int = 2
    enabled: bool = True
    directory: str = ".yakbox/speech-analysis"
    legacy_evidence: str = "historical_only"

    def __post_init__(self) -> None:
        candidate = Path(self.directory)
        if (
            self.version != _DRAFT_CACHE_VERSION
            or not self.directory
            or candidate.is_absolute()
            or ".." in candidate.parts
        ):
            raise ValidationError(
                "Draft analysis cache must use version 2 and a safe relative path"
            )
        if self.legacy_evidence != "historical_only":
            raise ValidationError(
                "Version-1 analysis evidence must remain historical only"
            )


@dataclass(frozen=True, slots=True)
class DraftShortUtteranceAnalysisPolicy:
    """Candidate bounds and verification authority for carrier extraction."""

    maximum_words: int = 3
    candidate_count: int = 5
    maximum_rounds: int = 3
    require_independent_crop_verification: bool = True

    def __post_init__(self) -> None:
        if min(self.maximum_words, self.candidate_count, self.maximum_rounds) < 1:
            raise ValidationError("Draft short-utterance limits must be positive")
        if not self.require_independent_crop_verification:
            raise ValidationError(
                "Draft carrier crops require independent verification"
            )


@dataclass(frozen=True, slots=True)
class DraftJoinAnalysisPolicy:
    """Automatic contextual join-analysis controls."""

    automatic_inspection: bool = True
    window_seconds: float = 1.5
    coalesce_gap_ms: int = 100

    def __post_init__(self) -> None:
        if self.window_seconds <= 0 or self.coalesce_gap_ms < 0:
            raise ValidationError("Draft join-analysis timing is invalid")


@dataclass(frozen=True, slots=True)
class DraftChapterAnalysisPolicy:
    """Complete mastered-chapter verification policy."""

    verification: bool = False


@dataclass(frozen=True, slots=True)
class DraftPhonemeAnalysisPolicy:
    """Optional phoneme evidence retained as a non-lexical signal gate."""

    enabled: bool = False
    backend: str = "wav2vec2-ctc"
    model: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    revision: str | None = "c43348bbaa5a77692c8e7bf3409d683474fdf2a4"
    language: str = "en-us"
    timeout_seconds: float = 180.0
    minimum_confidence: float = 0.2

    def __post_init__(self) -> None:
        if not self.backend or not self.model or not self.language:
            raise ValidationError("Draft phoneme policy is incomplete")
        if self.timeout_seconds <= 0 or not 0 <= self.minimum_confidence <= 1:
            raise ValidationError("Draft phoneme policy thresholds are invalid")


@dataclass(frozen=True, slots=True)
class DraftRepairAnalysisPolicy:
    """Bounded candidate work and non-release approval semantics."""

    verification: str = "strict"
    approval_output: str = "repair_candidate"
    candidate_verification_scope: str = "affected"
    candidates_per_round: int = 5
    maximum_rounds: int = 3

    def __post_init__(self) -> None:
        if (
            self.verification != "strict"
            or self.approval_output != "repair_candidate"
            or self.candidate_verification_scope != "affected"
            or min(self.candidates_per_round, self.maximum_rounds) < 1
        ):
            raise ValidationError("Draft repair analysis must use bounded strict gates")


@dataclass(frozen=True, slots=True)
class DraftReleaseAnalysisPolicy:
    """Complete exact-master and decoded-delivery release requirements."""

    complete_master_recognition: bool = True
    verify_decoded_delivery: bool = True
    codec_equivalence: str = "disabled_until_qualified"

    def __post_init__(self) -> None:
        if (
            not self.complete_master_recognition
            or not self.verify_decoded_delivery
            or self.codec_equivalence != "disabled_until_qualified"
        ):
            raise ValidationError(
                "Draft releases require complete exact-audio analysis"
            )


@dataclass(frozen=True, slots=True)
class DraftSpeechAnalysisConfig:
    """Parsed internal draft; version 1 remains the public manifest contract."""

    policy: SpeechAnalysisPolicy
    cache: DraftAnalysisCachePolicy
    short_utterances: DraftShortUtteranceAnalysisPolicy
    joins: DraftJoinAnalysisPolicy
    chapters: DraftChapterAnalysisPolicy
    phonemes: DraftPhonemeAnalysisPolicy
    repairs: DraftRepairAnalysisPolicy
    releases: DraftReleaseAnalysisPolicy

    @property
    def cache_enabled(self) -> bool:
        """Compatibility view used by internal Phase-1 callers."""
        return self.cache.enabled

    @property
    def cache_directory(self) -> str:
        """Compatibility view used by internal Phase-1 callers."""
        return self.cache.directory

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("draft-speech-analysis-config-v1", self)

    def to_manifest_value(self) -> dict[str, object]:
        policy = self.policy
        return {
            "preset": policy.preset,
            "baseline_recognizers": list(policy.baseline_recognizers),
            "escalation_recognizer": policy.escalation_recognizer,
            "forced_aligner": policy.forced_aligner,
            "always_escalate_clip_classes": [
                item.value for item in policy.always_escalate_clip_classes
            ],
            "always_escalate_repairs": policy.always_escalate_repairs,
            "engines": {
                engine.engine: _engine_value(engine) for engine in policy.engines
            },
            "consensus": {
                "ordinary_acceptance": "all_baseline_match",
                "escalated_acceptance": "two_matches_no_valid_dissent",
                "reject_unresolved_disagreement": (
                    policy.reject_unresolved_disagreement
                ),
                "reject_unexpected_speech": policy.reject_unexpected_speech,
                "valid_dissent": policy.valid_dissent,
                "missing_required_engine": policy.missing_required_engine,
                "one_word_required_recognizers": list(
                    policy.one_word_required_recognizers
                ),
            },
            "cache": {
                "version": self.cache.version,
                "enabled": self.cache.enabled,
                "directory": self.cache.directory,
                "legacy_evidence": self.cache.legacy_evidence,
            },
            "short_utterances": {
                "maximum_words": self.short_utterances.maximum_words,
                "candidate_count": self.short_utterances.candidate_count,
                "maximum_rounds": self.short_utterances.maximum_rounds,
                "require_independent_crop_verification": (
                    self.short_utterances.require_independent_crop_verification
                ),
            },
            "joins": {
                "automatic_inspection": self.joins.automatic_inspection,
                "window_seconds": self.joins.window_seconds,
                "coalesce_gap_ms": self.joins.coalesce_gap_ms,
            },
            "chapters": {"verification": self.chapters.verification},
            "phonemes": {
                "enabled": self.phonemes.enabled,
                "backend": self.phonemes.backend,
                "model": self.phonemes.model,
                "revision": self.phonemes.revision,
                "language": self.phonemes.language,
                "timeout_seconds": self.phonemes.timeout_seconds,
                "minimum_confidence": self.phonemes.minimum_confidence,
            },
            "repairs": {
                "verification": self.repairs.verification,
                "approval_output": self.repairs.approval_output,
                "candidate_verification_scope": (
                    self.repairs.candidate_verification_scope
                ),
                "candidates_per_round": self.repairs.candidates_per_round,
                "maximum_rounds": self.repairs.maximum_rounds,
            },
            "releases": {
                "complete_master_recognition": (
                    self.releases.complete_master_recognition
                ),
                "verify_decoded_delivery": self.releases.verify_decoded_delivery,
                "codec_equivalence": self.releases.codec_equivalence,
            },
        }


def default_draft_speech_analysis_config(
    registry: ModelRegistryData | None = None,
) -> DraftSpeechAnalysisConfig:
    """Build the strict English default from the reviewed model registry."""
    data = registry or load_model_registry()
    records = {record.engine: record for record in data.models}
    if set(records) != set(_ENGINE_NAMES):
        raise ValidationError("Reviewed registry does not define the strict engine set")
    timeouts = {
        "whisper": 180.0,
        "parakeet": 180.0,
        "qwen": 300.0,
        "qwen-forced": 300.0,
    }
    engines = tuple(
        EnginePolicy(
            engine=engine,
            backend=records[engine].backend_package,
            model=records[engine].converted_repository,
            revision=records[engine].converted_revision,
            timeout_seconds=timeouts[engine],
            decode_mode="argmax" if engine == "qwen-forced" else "greedy",
            chunk_seconds=120 if engine == "parakeet" else None,
            overlap_seconds=15 if engine == "parakeet" else None,
            maximum_window_seconds=300 if engine == "qwen-forced" else None,
        )
        for engine in _ENGINE_NAMES
    )
    return DraftSpeechAnalysisConfig(
        policy=SpeechAnalysisPolicy(
            version=1,
            preset="strict",
            language="en",
            baseline_recognizers=("whisper", "parakeet"),
            escalation_recognizer="qwen",
            forced_aligner="qwen-forced",
            always_escalate_clip_classes=(
                ClipClass.ONE_WORD,
                ClipClass.SHORT_PHRASE,
                ClipClass.JOIN,
                ClipClass.REPAIRED_REGION,
            ),
            always_escalate_repairs=True,
            reject_unresolved_disagreement=True,
            reject_unexpected_speech=True,
            missing_required_engine="error",
            valid_dissent="retry_then_reject",
            engines=engines,
        ),
        cache=DraftAnalysisCachePolicy(),
        short_utterances=DraftShortUtteranceAnalysisPolicy(),
        joins=DraftJoinAnalysisPolicy(),
        chapters=DraftChapterAnalysisPolicy(),
        phonemes=DraftPhonemeAnalysisPolicy(),
        repairs=DraftRepairAnalysisPolicy(),
        releases=DraftReleaseAnalysisPolicy(),
    )


def parse_draft_manifest_speech_analysis(
    manifest: Mapping[str, object],
    *,
    registry: ModelRegistryData | None = None,
) -> DraftSpeechAnalysisConfig:
    """Parse only the internal version-2 section and reject engine substitution."""
    if _integer(manifest, "schema_version") != _DRAFT_MANIFEST_VERSION:
        raise ValidationError("Draft speech analysis requires manifest version 2")
    book = _mapping(manifest.get("book"), "book")
    if _string(book, "language") != "en":
        raise ValidationError("Draft speech analysis currently supports English only")
    value = _mapping(manifest.get("speech_analysis"), "speech_analysis")
    _reject_unknown(value, _SPEECH_KEYS, "speech_analysis")
    return _parse_speech_analysis(value, registry or load_model_registry())


def _parse_speech_analysis(
    value: Mapping[str, object], registry: ModelRegistryData
) -> DraftSpeechAnalysisConfig:
    if _string(value, "preset") != "strict":
        raise ValidationError("Draft speech analysis accepts the strict preset only")
    baseline = _string_tuple(value, "baseline_recognizers")
    if baseline != ("whisper", "parakeet"):
        raise ValidationError("Draft baseline recognizers cannot be replaced")
    escalation = _string(value, "escalation_recognizer")
    forced = _string(value, "forced_aligner")
    if escalation != "qwen" or forced != "qwen-forced":
        raise ValidationError("Draft escalation engines cannot be replaced")
    engines = _parse_engines(_mapping(value.get("engines"), "engines"), registry)
    consensus = _mapping(value.get("consensus"), "consensus")
    _validate_consensus(consensus)
    cache = _parse_cache(_mapping(value.get("cache"), "cache"))
    short_utterances = _parse_short(
        _mapping(value.get("short_utterances"), "short_utterances")
    )
    joins = _parse_joins(_mapping(value.get("joins"), "joins"))
    chapters = _parse_chapters(_mapping(value.get("chapters"), "chapters"))
    phonemes = _parse_phonemes(_mapping(value.get("phonemes"), "phonemes"))
    repairs = _parse_repairs(_mapping(value.get("repairs"), "repairs"))
    releases = _parse_releases(_mapping(value.get("releases"), "releases"))
    try:
        clip_classes = tuple(
            ClipClass(item)
            for item in _string_tuple(value, "always_escalate_clip_classes")
        )
    except ValueError as error:
        raise ValidationError("Unknown draft speech-analysis clip class") from error
    return DraftSpeechAnalysisConfig(
        policy=SpeechAnalysisPolicy(
            version=1,
            preset="strict",
            language="en",
            baseline_recognizers=baseline,
            escalation_recognizer=escalation,
            forced_aligner=forced,
            always_escalate_clip_classes=clip_classes,
            always_escalate_repairs=_boolean(value, "always_escalate_repairs"),
            reject_unresolved_disagreement=_boolean(
                consensus, "reject_unresolved_disagreement"
            ),
            reject_unexpected_speech=_boolean(consensus, "reject_unexpected_speech"),
            missing_required_engine=_string(consensus, "missing_required_engine"),
            valid_dissent=_string(consensus, "valid_dissent"),
            engines=engines,
        ),
        cache=cache,
        short_utterances=short_utterances,
        joins=joins,
        chapters=chapters,
        phonemes=phonemes,
        repairs=repairs,
        releases=releases,
    )


def _parse_cache(raw: Mapping[str, object]) -> DraftAnalysisCachePolicy:
    _reject_unknown(raw, _CACHE_KEYS, "cache")
    return DraftAnalysisCachePolicy(
        version=_integer(raw, "version"),
        enabled=_boolean(raw, "enabled"),
        directory=_string(raw, "directory"),
        legacy_evidence=_string(raw, "legacy_evidence"),
    )


def _parse_short(raw: Mapping[str, object]) -> DraftShortUtteranceAnalysisPolicy:
    _reject_unknown(raw, _SHORT_KEYS, "short_utterances")
    return DraftShortUtteranceAnalysisPolicy(
        maximum_words=_integer(raw, "maximum_words"),
        candidate_count=_integer(raw, "candidate_count"),
        maximum_rounds=_integer(raw, "maximum_rounds"),
        require_independent_crop_verification=_boolean(
            raw, "require_independent_crop_verification"
        ),
    )


def _parse_joins(raw: Mapping[str, object]) -> DraftJoinAnalysisPolicy:
    _reject_unknown(raw, _JOIN_KEYS, "joins")
    return DraftJoinAnalysisPolicy(
        automatic_inspection=_boolean(raw, "automatic_inspection"),
        window_seconds=_number(raw, "window_seconds"),
        coalesce_gap_ms=_integer(raw, "coalesce_gap_ms"),
    )


def _parse_chapters(raw: Mapping[str, object]) -> DraftChapterAnalysisPolicy:
    _reject_unknown(raw, _CHAPTER_KEYS, "chapters")
    return DraftChapterAnalysisPolicy(verification=_boolean(raw, "verification"))


def _parse_phonemes(raw: Mapping[str, object]) -> DraftPhonemeAnalysisPolicy:
    _reject_unknown(raw, _PHONEME_KEYS, "phonemes")
    return DraftPhonemeAnalysisPolicy(
        enabled=_boolean(raw, "enabled"),
        backend=_string(raw, "backend"),
        model=_string(raw, "model"),
        revision=_optional_nullable_string(raw, "revision"),
        language=_string(raw, "language"),
        timeout_seconds=_number(raw, "timeout_seconds"),
        minimum_confidence=_number(raw, "minimum_confidence"),
    )


def _parse_repairs(raw: Mapping[str, object]) -> DraftRepairAnalysisPolicy:
    _reject_unknown(raw, _REPAIR_KEYS, "repairs")
    return DraftRepairAnalysisPolicy(
        verification=_string(raw, "verification"),
        approval_output=_string(raw, "approval_output"),
        candidate_verification_scope=_string(raw, "candidate_verification_scope"),
        candidates_per_round=_integer(raw, "candidates_per_round"),
        maximum_rounds=_integer(raw, "maximum_rounds"),
    )


def _parse_releases(raw: Mapping[str, object]) -> DraftReleaseAnalysisPolicy:
    _reject_unknown(raw, _RELEASE_KEYS, "releases")
    return DraftReleaseAnalysisPolicy(
        complete_master_recognition=_boolean(raw, "complete_master_recognition"),
        verify_decoded_delivery=_boolean(raw, "verify_decoded_delivery"),
        codec_equivalence=_string(raw, "codec_equivalence"),
    )


def _parse_engines(
    raw: Mapping[str, object], registry: ModelRegistryData
) -> tuple[EnginePolicy, ...]:
    if set(raw) != set(_ENGINE_NAMES):
        raise ValidationError("Draft speech analysis requires the built-in engine set")
    records = {record.engine: record for record in registry.models}
    engines: list[EnginePolicy] = []
    for name in _ENGINE_NAMES:
        value = _mapping(raw.get(name), f"engine {name}")
        _reject_unknown(value, _ENGINE_KEYS, f"engine {name}")
        record = records.get(name)
        if record is None:
            raise ValidationError(f"Engine {name!r} is absent from the registry")
        backend = _string(value, "backend")
        model = _string(value, "model")
        revision = _string(value, "revision")
        if (
            backend != record.backend_package
            or model != record.converted_repository
            or revision != record.converted_revision
        ):
            raise ValidationError(
                f"Engine {name!r} differs from its reviewed registry record"
            )
        engines.append(
            EnginePolicy(
                engine=name,
                backend=backend,
                model=model,
                revision=revision,
                timeout_seconds=_number(value, "timeout_seconds"),
                decode_mode=_string(value, "decode_mode"),
                chunk_seconds=_optional_integer(value, "chunk_seconds"),
                overlap_seconds=_optional_integer(value, "overlap_seconds"),
                maximum_window_seconds=_optional_integer(
                    value, "maximum_window_seconds"
                ),
            )
        )
    return tuple(engines)


def _validate_consensus(raw: Mapping[str, object]) -> None:
    required = {
        "ordinary_acceptance": "all_baseline_match",
        "escalated_acceptance": "two_matches_no_valid_dissent",
        "valid_dissent": "retry_then_reject",
        "missing_required_engine": "error",
    }
    allowed = {
        *required,
        "reject_unresolved_disagreement",
        "reject_unexpected_speech",
        "one_word_required_recognizers",
    }
    _reject_unknown(raw, allowed, "consensus")
    if any(_string(raw, key) != expected for key, expected in required.items()):
        raise ValidationError("Draft strict consensus truth table cannot be changed")
    if _string_tuple(raw, "one_word_required_recognizers") != (
        "whisper",
        "qwen",
    ):
        raise ValidationError("Draft one-word recognizer authority cannot be changed")


def _engine_value(engine: EnginePolicy) -> dict[str, object]:
    value: dict[str, object] = {
        "backend": engine.backend,
        "model": engine.model,
        "revision": engine.revision,
        "timeout_seconds": engine.timeout_seconds,
        "decode_mode": engine.decode_mode,
    }
    if engine.chunk_seconds is not None:
        value["chunk_seconds"] = engine.chunk_seconds
    if engine.overlap_seconds is not None:
        value["overlap_seconds"] = engine.overlap_seconds
    if engine.maximum_window_seconds is not None:
        value["maximum_window_seconds"] = engine.maximum_window_seconds
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"Draft {label} must be a table")
    return cast(Mapping[str, object], value)


def _reject_unknown(value: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = tuple(sorted(set(value) - allowed))
    if unknown:
        raise ValidationError(f"Unknown draft {label} field: {unknown[0]}")


def _string(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Draft speech-analysis {key!r} must be text")
    return value


def _optional_string(raw: Mapping[str, object], key: str, default: str) -> str:
    return default if key not in raw else _string(raw, key)


def _optional_nullable_string(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValidationError(f"Draft speech-analysis {key!r} must be text or null")
    return value


def _string_tuple(raw: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValidationError(f"Draft speech-analysis {key!r} must be an array")
    if not all(isinstance(item, str) and item for item in value):
        raise ValidationError(f"Draft speech-analysis {key!r} entries must be text")
    return cast(tuple[str, ...], tuple(value))


def _boolean(raw: Mapping[str, object], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValidationError(f"Draft speech-analysis {key!r} must be boolean")
    return value


def _optional_boolean(raw: Mapping[str, object], key: str, default: bool) -> bool:
    return default if key not in raw else _boolean(raw, key)


def _integer(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"Draft speech-analysis {key!r} must be an integer")
    return value


def _optional_integer(raw: Mapping[str, object], key: str) -> int | None:
    return None if key not in raw else _integer(raw, key)


def _number(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"Draft speech-analysis {key!r} must be numeric")
    return float(value)


__all__ = [
    "DraftAnalysisCachePolicy",
    "DraftChapterAnalysisPolicy",
    "DraftJoinAnalysisPolicy",
    "DraftPhonemeAnalysisPolicy",
    "DraftReleaseAnalysisPolicy",
    "DraftRepairAnalysisPolicy",
    "DraftShortUtteranceAnalysisPolicy",
    "DraftSpeechAnalysisConfig",
    "default_draft_speech_analysis_config",
    "parse_draft_manifest_speech_analysis",
]
