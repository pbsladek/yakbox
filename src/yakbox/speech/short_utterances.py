"""Deterministic policy and carrier recipes for short speech synthesis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum

import regex

from yakbox.errors import ValidationError
from yakbox.speech.alignment import lexical_tokens

SHORT_UTTERANCE_POLICY_VERSION = "short-utterance-v17"
SHORT_UTTERANCE_SEED_VERSION = "short-utterance-v6"
_MAXIMUM_NATURAL_CONTEXT_CHARACTERS = 180
_TARGET = "{target}"
_CARRIER_TEMPLATES = (
    (
        "neutral-exchange-v1",
        "The exchange continued. {target} The others remained quiet.",
    ),
    (
        "neutral-moment-v1",
        "A quiet moment passed. {target} Then the conversation continued.",
    ),
    (
        "neutral-room-v1",
        "Everyone in the room listened. {target} The room settled again.",
    ),
)


class ShortUtteranceStrategy(StrEnum):
    """Configured synthesis path for risky chunks."""

    DIRECT = "direct"
    CONTEXT_EXTRACT = "context_extract"


class ShortUtteranceFailure(StrEnum):
    """Build outcome when no verified candidate exists."""

    ERROR = "error"
    REVIEW = "review"


class CarrierPosition(StrEnum):
    """Location of the target within a hidden carrier."""

    MIDDLE = "middle"
    INITIAL = "initial"
    FINAL = "final"
    DIRECT = "direct"


_MINIMUM_ACOUSTIC_THRESHOLD_DBFS = -120


@dataclass(frozen=True, slots=True)
class ShortUtterancePolicy:
    """Manifest-controlled safety and candidate-generation settings."""

    strategy: ShortUtteranceStrategy = ShortUtteranceStrategy.DIRECT
    maximum_words: int = 3
    candidate_count: int = 5
    prefer_natural_context: bool = True
    carrier_positions: tuple[CarrierPosition, ...] = (CarrierPosition.MIDDLE,)
    alignment_backend: str = "mlx-whisper"
    alignment_model: str = "mlx-community/whisper-large-v3-turbo"
    alignment_revision: str | None = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
    alignment_aliases: tuple[tuple[str, tuple[str, ...]], ...] = ()
    prompted_timing: bool = True
    decode_consensus: bool = True
    prompt_sensitivity: bool = True
    maximum_consensus_timing_delta_ms: int = 180
    hallucination_silence_threshold: float = 0.8
    automatic_join_inspection: bool = True
    join_inspection_window_seconds: float = 1.5
    alignment_timeout_seconds: float = 180.0
    minimum_alignment_confidence: float = 0.5
    minimum_extracted_confidence: float = 0.2
    minimum_one_word_confidence: float = 0.6
    minimum_short_phrase_confidence: float = 0.5
    minimum_segment_average_log_probability: float = -1.0
    maximum_segment_compression_ratio: float = 2.4
    maximum_segment_no_speech_probability: float = 0.6
    maximum_segment_temperature: float = 0.2
    candidate_confidence_tolerance: float = 0.05
    maximum_extra_speech_ms: int = 60
    maximum_internal_token_gap_ms: int = 350
    maximum_token_duration_ms: int = 1_200
    acoustic_refinement: bool = True
    acoustic_threshold_dbfs: float = -48.0
    speech_island_gap_ms: int = 300
    minimum_edge_silence_ms: int = 10
    maximum_edge_silence_ms: int = 120
    maximum_clipped_sample_ratio: float = 0.005
    maximum_boundary_jump_ratio: float = 0.35
    maximum_vad_disagreement_ms: int = 500
    maximum_stationary_voiced_ms: int = 1_200
    minimum_pause_ms: int = 180
    pre_roll_ms: int = 30
    post_roll_ms: int = 40
    fade_ms: int = 8
    failure: ShortUtteranceFailure = ShortUtteranceFailure.ERROR
    require_review_for_one_word: bool = True
    keep_candidates: bool = False

    def __post_init__(self) -> None:
        if self.maximum_words < 1:
            raise ValidationError("short_utterances.maximum_words must be positive")
        if self.candidate_count < 1:
            raise ValidationError("short_utterances.candidate_count must be positive")
        if not self.carrier_positions:
            raise ValidationError(
                "short_utterances.carrier_positions must not be empty"
            )
        _validate_confidence(
            self.minimum_alignment_confidence,
            "minimum_alignment_confidence",
        )
        _validate_confidence(
            self.minimum_extracted_confidence,
            "minimum_extracted_confidence",
        )
        _validate_confidence(
            self.minimum_one_word_confidence,
            "minimum_one_word_confidence",
        )
        _validate_confidence(
            self.minimum_short_phrase_confidence,
            "minimum_short_phrase_confidence",
        )
        _validate_confidence(
            self.candidate_confidence_tolerance,
            "candidate_confidence_tolerance",
        )
        _validate_confidence(
            self.maximum_segment_no_speech_probability,
            "maximum_segment_no_speech_probability",
        )
        _validate_confidence(
            self.maximum_segment_temperature,
            "maximum_segment_temperature",
        )
        _validate_positive(
            self.hallucination_silence_threshold,
            "hallucination_silence_threshold",
        )
        _validate_confidence(
            self.maximum_clipped_sample_ratio,
            "maximum_clipped_sample_ratio",
        )
        _validate_confidence(
            self.maximum_boundary_jump_ratio,
            "maximum_boundary_jump_ratio",
        )
        if self.alignment_timeout_seconds <= 0:
            raise ValidationError(
                "short_utterances.alignment_timeout_seconds must be positive"
            )
        _validate_positive(
            self.join_inspection_window_seconds,
            "join_inspection_window_seconds",
        )
        _validate_positive(
            self.maximum_token_duration_ms,
            "maximum_token_duration_ms",
        )
        if self.maximum_segment_compression_ratio <= 0:
            raise ValidationError(
                "short_utterances.maximum_segment_compression_ratio must be positive"
            )
        nonnegative = (
            self.maximum_extra_speech_ms,
            self.maximum_internal_token_gap_ms,
            self.maximum_token_duration_ms,
            self.maximum_consensus_timing_delta_ms,
            self.speech_island_gap_ms,
            self.minimum_edge_silence_ms,
            self.maximum_edge_silence_ms,
            self.maximum_vad_disagreement_ms,
            self.maximum_stationary_voiced_ms,
            self.minimum_pause_ms,
            self.pre_roll_ms,
            self.post_roll_ms,
            self.fade_ms,
        )
        if any(value < 0 for value in nonnegative):
            raise ValidationError("short_utterance timing settings cannot be negative")
        if self.minimum_edge_silence_ms > self.maximum_edge_silence_ms:
            raise ValidationError(
                "short_utterances.minimum_edge_silence_ms must not exceed "
                "maximum_edge_silence_ms"
            )
        if not _MINIMUM_ACOUSTIC_THRESHOLD_DBFS <= self.acoustic_threshold_dbfs <= 0:
            raise ValidationError(
                "short_utterances.acoustic_threshold_dbfs must be between -120 and 0"
            )
        if not self.alignment_backend.strip() or not self.alignment_model.strip():
            raise ValidationError("short_utterance alignment settings cannot be empty")
        _validate_alignment_aliases(self.alignment_aliases)

    @property
    def alignment_alias_map(self) -> dict[str, tuple[str, ...]]:
        """Return explicitly reviewed ASR spellings keyed by source spelling."""
        return dict(self.alignment_aliases)

    def minimum_confidence_for(self, word_count: int, *, extracted: bool) -> float:
        """Return the calibrated gate for a one-word or short-phrase clip."""
        baseline = (
            self.minimum_extracted_confidence
            if extracted
            else self.minimum_alignment_confidence
        )
        calibrated = (
            self.minimum_one_word_confidence
            if word_count == 1
            else self.minimum_short_phrase_confidence
        )
        return max(baseline, calibrated)

    @property
    def generation_fingerprint(self) -> str:
        """Return the identity of candidate recipes, independent of QA policy."""
        payload = json.dumps(
            {
                "version": f"{SHORT_UTTERANCE_POLICY_VERSION}-generation-v1",
                "strategy": self.strategy,
                "maximum_words": self.maximum_words,
                "candidate_count": self.candidate_count,
                "prefer_natural_context": self.prefer_natural_context,
                "carrier_positions": self.carrier_positions,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def evaluation_fingerprint(self) -> str:
        """Return the identity of per-candidate hard gates."""
        generation_fields = {
            "strategy",
            "maximum_words",
            "candidate_count",
            "prefer_natural_context",
            "carrier_positions",
            "automatic_join_inspection",
            "join_inspection_window_seconds",
            "keep_candidates",
        }
        extraction_fields = {
            "alignment_backend",
            "alignment_model",
            "alignment_revision",
            "alignment_aliases",
            "prompted_timing",
            "decode_consensus",
            "prompt_sensitivity",
            "maximum_consensus_timing_delta_ms",
            "hallucination_silence_threshold",
            "alignment_timeout_seconds",
            "pre_roll_ms",
            "post_roll_ms",
            "fade_ms",
        }
        selection_fields = {
            "candidate_confidence_tolerance",
            "failure",
            "require_review_for_one_word",
        }
        payload = json.dumps(
            {
                "version": f"{SHORT_UTTERANCE_POLICY_VERSION}-evaluation-v1",
                **{
                    key: value
                    for key, value in asdict(self).items()
                    if key
                    not in generation_fields | extraction_fields | selection_fields
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def extraction_fingerprint(self) -> str:
        """Return the identity of alignment and carrier-crop behavior."""
        fields = (
            "alignment_backend",
            "alignment_model",
            "alignment_revision",
            "alignment_aliases",
            "prompted_timing",
            "decode_consensus",
            "prompt_sensitivity",
            "maximum_consensus_timing_delta_ms",
            "hallucination_silence_threshold",
            "alignment_timeout_seconds",
            "pre_roll_ms",
            "post_roll_ms",
            "fade_ms",
        )
        values = asdict(self)
        payload = json.dumps(
            {
                "version": f"{SHORT_UTTERANCE_POLICY_VERSION}-extraction-v1",
                **{key: values[key] for key in fields},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def selection_fingerprint(self) -> str:
        """Return the identity of ranking and post-selection review behavior."""
        payload = json.dumps(
            {
                "version": f"{SHORT_UTTERANCE_POLICY_VERSION}-selection-v1",
                "candidate_confidence_tolerance": self.candidate_confidence_tolerance,
                "prefer_natural_context": self.prefer_natural_context,
                "failure": self.failure,
                "require_review_for_one_word": self.require_review_for_one_word,
                "keep_candidates": self.keep_candidates,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def join_fingerprint(self) -> str:
        """Return the identity of post-assembly join inspection policy."""
        payload = json.dumps(
            {
                "version": f"{SHORT_UTTERANCE_POLICY_VERSION}-join-v1",
                "automatic_join_inspection": self.automatic_join_inspection,
                "join_inspection_window_seconds": self.join_inspection_window_seconds,
                "alignment_backend": self.alignment_backend,
                "alignment_model": self.alignment_model,
                "alignment_revision": self.alignment_revision,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def fingerprint(self) -> str:
        """Return the compatibility identity covering every behavior setting."""
        payload = json.dumps(
            {
                "version": SHORT_UTTERANCE_POLICY_VERSION,
                **asdict(self),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ShortUtteranceRisk:
    """Explainable risk classification for one normalized speech chunk."""

    risky: bool
    word_count: int
    reason: str | None


def _validate_confidence(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValidationError(f"short_utterances.{name} must be between 0 and 1")


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValidationError(f"short_utterances.{name} must be positive")


def _validate_alignment_aliases(
    entries: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    canonical: set[str] = set()
    accepted: set[str] = set()
    for word, aliases in entries:
        if lexical_tokens(word) != (word,) or not aliases:
            raise ValidationError(
                "short_utterances.alignment_aliases must map one normalized "
                "word to one or more normalized words"
            )
        values = {word, *aliases}
        if any(lexical_tokens(value) != (value,) for value in values):
            raise ValidationError(
                "short_utterances.alignment_aliases values must be "
                "single normalized words"
            )
        if word in canonical or accepted.intersection(values):
            raise ValidationError(
                "short_utterances.alignment_aliases must not be ambiguous"
            )
        canonical.add(word)
        accepted.update(values)


@dataclass(frozen=True, slots=True)
class CarrierRecipe:
    """Hidden generation text and reproducibility metadata for one candidate."""

    candidate_index: int
    text: str
    target_text: str
    template_id: str
    position: CarrierPosition
    natural: bool
    seed: int


def classify_short_utterance(
    text: str,
    policy: ShortUtterancePolicy,
    *,
    explicit_pause: bool = False,
) -> ShortUtteranceRisk:
    """Classify any routed speech chunk using normalized Unicode words."""
    words = lexical_tokens(text)
    if explicit_pause:
        return ShortUtteranceRisk(False, 0, "explicit_pause")
    if not words:
        return ShortUtteranceRisk(False, 0, "empty")
    if len(words) <= policy.maximum_words:
        return ShortUtteranceRisk(True, len(words), "word_count")
    return ShortUtteranceRisk(False, len(words), None)


def carrier_recipes(
    target_text: str,
    policy: ShortUtterancePolicy,
    *,
    seed_material: str,
    previous_context: str | None = None,
    next_context: str | None = None,
) -> tuple[CarrierRecipe, ...]:
    """Build a bounded, deterministic direct/natural/synthetic candidate matrix."""
    adapted = adapt_target_punctuation(target_text)
    entries: list[tuple[str, str, CarrierPosition, bool]] = [
        ("direct-v1", target_text.strip(), CarrierPosition.DIRECT, False)
    ]
    for position in policy.carrier_positions:
        for template_id, template in _CARRIER_TEMPLATES:
            entries.append(
                (
                    template_id,
                    _render_template(template, adapted, position),
                    position,
                    False,
                )
            )
    if policy.prefer_natural_context:
        natural = _natural_carrier(adapted, previous_context, next_context)
        if natural is not None:
            entries.append(("natural-v1", natural, CarrierPosition.MIDDLE, True))
    safe_entries = tuple(
        entry for entry in entries if _target_occurrences(entry[1], target_text) == 1
    )
    bounded_entries = tuple(
        safe_entries[index % len(safe_entries)]
        for index in range(policy.candidate_count)
    )
    recipes = tuple(
        CarrierRecipe(
            candidate_index=index,
            text=text,
            target_text=target_text.strip(),
            template_id=template_id,
            position=position,
            natural=natural,
            seed=_candidate_seed(
                seed_material,
                target_text,
                template_id,
                position,
                index,
            ),
        )
        for index, (template_id, text, position, natural) in enumerate(
            bounded_entries, start=1
        )
    )
    if not recipes:
        raise ValidationError("Short-utterance policy produced no candidates")
    return recipes


def adapt_target_punctuation(text: str) -> str:
    """Create a sentence boundary without changing the lexical target."""
    value = text.strip()
    if not value:
        raise ValidationError("Short-utterance target must not be empty")
    if value.endswith(("?", "!", ".")):
        return value
    if value.endswith((",", ";", ":")):
        return f"{value[:-1].rstrip()}."
    return f"{value}."


def _render_template(template: str, target: str, position: CarrierPosition) -> str:
    if template.count(_TARGET) != 1:
        raise ValidationError("Carrier templates must contain {target} exactly once")
    prefix, suffix = template.split(_TARGET)
    if position is CarrierPosition.MIDDLE:
        return _clean_spaces(template.replace(_TARGET, target))
    if position is CarrierPosition.INITIAL:
        return _clean_spaces(f"{target} {suffix}")
    if position is CarrierPosition.FINAL:
        return _clean_spaces(f"{prefix} {target}")
    raise ValidationError(f"Unsupported carrier position: {position}")


def _natural_carrier(
    target: str,
    previous_context: str | None,
    next_context: str | None,
) -> str | None:
    before = _context_fragment(previous_context, last=True)
    after = _context_fragment(next_context, last=False)
    if before is None and after is None:
        return None
    if before is None:
        before = "A quiet moment passed."
    if after is None:
        after = "Then the conversation continued."
    return _clean_spaces(f"{before} {target} {after}")


def _context_fragment(text: str | None, *, last: bool) -> str | None:
    if text is None:
        return None
    value = text.strip()
    if not value:
        return None
    sentences = tuple(
        part.strip() for part in regex.split(r"(?<=[.!?])\s+", value) if part.strip()
    )
    selected = sentences[-1] if last else sentences[0]
    if len(selected) > _MAXIMUM_NATURAL_CONTEXT_CHARACTERS:
        return None
    return adapt_target_punctuation(selected)


def _candidate_seed(
    material: str,
    target: str,
    template_id: str,
    position: CarrierPosition,
    attempt: int,
) -> int:
    payload = json.dumps(
        {
            "version": SHORT_UTTERANCE_SEED_VERSION,
            "material": material,
            "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
            "template": template_id,
            "position": position.value,
            "attempt": attempt,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _target_occurrences(text: str, target: str) -> int:
    carrier_tokens = lexical_tokens(text)
    target_tokens = lexical_tokens(target)
    if not target_tokens or len(target_tokens) > len(carrier_tokens):
        return 0
    return sum(
        carrier_tokens[index : index + len(target_tokens)] == target_tokens
        for index in range(len(carrier_tokens) - len(target_tokens) + 1)
    )


def _clean_spaces(text: str) -> str:
    return regex.sub(r"\s+", " ", text).strip()
