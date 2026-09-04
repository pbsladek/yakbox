"""Backend-neutral speech-recognition, alignment, and consensus contracts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from yakbox.errors import ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint

_SHA256 = re.compile(r"[0-9a-f]{64}")


class ClipClass(StrEnum):
    """Audio classes calibrated independently by every recognizer."""

    ONE_WORD = "one_word"
    SHORT_PHRASE = "short_phrase"
    SENTENCE = "sentence"
    JOIN = "join"
    CHAPTER = "chapter"
    REPAIRED_REGION = "repaired_region"


class AlignmentPurpose(StrEnum):
    """Authority carried by one forced-alignment request."""

    VERIFIED_TARGET = "verified_target"
    VERIFIED_ANCHORS = "verified_anchors"
    NON_AUTHORITATIVE = "non_authoritative"


class ScoreKind(StrEnum):
    """Recognizer-specific score semantics retained without conflation."""

    PROBABILITY = "probability"
    LOG_PROBABILITY = "log_probability"
    TDT_CONFIDENCE = "tdt_confidence"
    UNAVAILABLE = "unavailable"


class VoteState(StrEnum):
    """Result of applying an engine gate to one expected lexical span."""

    MATCH = "match"
    DISSENT = "dissent"
    INVALID = "invalid"
    NOT_RUN = "not_run"


class ConsensusOutcome(StrEnum):
    """Strict policy terminal outcome."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class VerificationScope(StrEnum):
    """Artifact scope proven by one final verification document."""

    CANDIDATE = "candidate"
    MASTER = "master"
    DELIVERY = "delivery"


@dataclass(frozen=True, slots=True)
class SourceTextSpan:
    """Exact source location corresponding to one spoken segment."""

    source_digest: str
    start_line: int
    start_character: int
    end_line: int
    end_character: int

    def __post_init__(self) -> None:
        _require_sha256(self.source_digest, "source digest")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValidationError("Source text span has invalid line bounds")
        if self.start_character < 0 or self.end_character < 0:
            raise ValidationError(
                "Source text span character bounds cannot be negative"
            )


@dataclass(frozen=True, slots=True)
class TextTransform:
    """One deterministic source-to-spoken text transformation."""

    kind: str
    version: int
    input_start: int
    input_end: int
    output_start: int
    output_end: int
    rule_fingerprint: str

    def __post_init__(self) -> None:
        if not self.kind or self.version < 1:
            raise ValidationError("Text transforms require a kind and positive version")
        if (
            min(
                self.input_start,
                self.input_end,
                self.output_start,
                self.output_end,
            )
            < 0
        ):
            raise ValidationError("Text transform bounds cannot be negative")
        if self.input_end < self.input_start or self.output_end < self.output_start:
            raise ValidationError("Text transform bounds must be ordered")
        _require_sha256(self.rule_fingerprint, "text transform rule fingerprint")


@dataclass(frozen=True, slots=True)
class SpokenTextSegment:
    """Resolved text authority for one synthesized speech segment."""

    segment_id: str
    source: SourceTextSpan
    display_text_hash: str
    synthesis_text_hash: str
    expected_lexical_tokens: tuple[str, ...]
    expected_phonemes: tuple[str, ...]
    speaker: str
    profile: str
    language: str
    boundary: str
    transforms: tuple[TextTransform, ...]

    def __post_init__(self) -> None:
        if not self.segment_id or not self.speaker or not self.profile:
            raise ValidationError("Spoken text segment identity is incomplete")
        _require_sha256(self.display_text_hash, "display text hash")
        _require_sha256(self.synthesis_text_hash, "synthesis text hash")
        if self.language != "en":
            raise ValidationError(
                "The initial speech-analysis release supports en only"
            )
        if not self.expected_lexical_tokens:
            raise ValidationError(
                "Spoken text segment requires expected lexical tokens"
            )
        _require_normalized_tokens(self.expected_lexical_tokens)


@dataclass(frozen=True, slots=True)
class SpokenTextPlan:
    """Immutable authority for the words intended to reach the listener."""

    version: int
    source_digest: str
    normalization_fingerprint: str
    segments: tuple[SpokenTextSegment, ...]

    def __post_init__(self) -> None:
        if self.version < 1 or not self.segments:
            raise ValidationError("Spoken text plan must be versioned and non-empty")
        _require_sha256(self.source_digest, "spoken text source digest")
        _require_sha256(
            self.normalization_fingerprint,
            "spoken text normalization fingerprint",
        )
        identifiers = tuple(segment.segment_id for segment in self.segments)
        if len(set(identifiers)) != len(identifiers):
            raise ValidationError("Spoken text segment identifiers must be unique")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("spoken-text-plan-v1", self)


@dataclass(frozen=True, slots=True)
class AudioSpan:
    """Exact sample-indexed span in one immutable audio coordinate space."""

    audio_digest: str
    start_frame: int
    end_frame: int
    sample_rate: int

    def __post_init__(self) -> None:
        _require_sha256(self.audio_digest, "audio digest")
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValidationError("Audio span requires ordered, non-empty frames")
        if self.sample_rate <= 0:
            raise ValidationError("Audio span sample rate must be positive")


@dataclass(frozen=True, slots=True)
class FrameCoordinateMap:
    """Deterministic mapping from canonical frames back to source frames."""

    source_rate: int
    analysis_rate: int
    source_frame_count: int
    analysis_frame_count: int
    source_delay_frames: int = 0

    def __post_init__(self) -> None:
        if (
            min(
                self.source_rate,
                self.analysis_rate,
                self.source_frame_count,
                self.analysis_frame_count,
            )
            <= 0
        ):
            raise ValidationError("Frame coordinate map values must be positive")


@dataclass(frozen=True, slots=True)
class CanonicalAudioIdentity:
    """Source and canonical analysis audio identity plus coordinate mapping."""

    source_digest: str
    source_format: str
    canonical_digest: str
    canonical_format: str
    preprocessing_fingerprint: str
    frame_map: FrameCoordinateMap

    def __post_init__(self) -> None:
        _require_sha256(self.source_digest, "source audio digest")
        _require_sha256(self.canonical_digest, "canonical audio digest")
        _require_sha256(
            self.preprocessing_fingerprint,
            "audio preprocessing fingerprint",
        )

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("canonical-audio-identity-v1", self)


@dataclass(frozen=True, slots=True)
class FrameTimingMap:
    """Qualified monotonic affine map between two exact audio frame spaces."""

    source_rate: int
    destination_rate: int
    source_frame_count: int
    destination_frame_count: int
    source_origin_frame: int
    destination_origin_frame: int
    scale_numerator: int
    scale_denominator: int
    delay_frames: int
    uncertainty_frames: int
    monotonic: bool
    qualified: bool
    calibration_fingerprint: str

    def __post_init__(self) -> None:
        positive = (
            self.source_rate,
            self.destination_rate,
            self.source_frame_count,
            self.destination_frame_count,
            self.scale_numerator,
            self.scale_denominator,
        )
        if (
            min(positive) <= 0
            or min(
                self.source_origin_frame,
                self.destination_origin_frame,
                self.uncertainty_frames,
            )
            < 0
        ):
            raise ValidationError("Frame timing map values are invalid")
        _require_sha256(self.calibration_fingerprint, "timing-map calibration")
        if self.qualified and not self.monotonic:
            raise ValidationError("A qualified frame timing map must be monotonic")

    def map_span_outward(self, start_frame: int, end_frame: int) -> tuple[int, int]:
        """Map a source interval outward, including delay and uncertainty."""
        if not self.qualified or not self.monotonic:
            raise ValidationError("Frame timing map is not qualified and monotonic")
        if (
            start_frame < 0
            or end_frame <= start_frame
            or end_frame > self.source_frame_count
        ):
            raise ValidationError("Source interval falls outside the timing map")
        relative_start = start_frame - self.source_origin_frame
        relative_end = end_frame - self.source_origin_frame
        start = (
            self.destination_origin_frame
            + self.delay_frames
            + relative_start * self.scale_numerator // self.scale_denominator
            - self.uncertainty_frames
        )
        end_numerator = relative_end * self.scale_numerator
        end = (
            self.destination_origin_frame
            + self.delay_frames
            + (end_numerator + self.scale_denominator - 1) // self.scale_denominator
            + self.uncertainty_frames
        )
        return max(0, start), min(self.destination_frame_count, end)


@dataclass(frozen=True, slots=True)
class MasteringAudioIdentity:
    """Exact raw/master identities and a measured coordinate transformation."""

    raw_assembly_digest: str
    mastered_digest: str
    raw_format: str
    mastered_format: str
    raw_frame_count: int
    mastered_frame_count: int
    ffmpeg_fingerprint: str
    filter_graph_fingerprint: str
    timing_map: FrameTimingMap
    mapping_validation_fingerprint: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.raw_assembly_digest, "raw assembly digest"),
            (self.mastered_digest, "mastered audio digest"),
            (self.ffmpeg_fingerprint, "FFmpeg fingerprint"),
            (self.filter_graph_fingerprint, "filter graph fingerprint"),
            (self.mapping_validation_fingerprint, "mapping validation fingerprint"),
        ):
            _require_sha256(value, label)
        if not self.raw_format or not self.mastered_format:
            raise ValidationError("Mastering identity formats are required")
        if (
            self.raw_frame_count != self.timing_map.source_frame_count
            or self.mastered_frame_count != self.timing_map.destination_frame_count
        ):
            raise ValidationError("Mastering identity does not match its timing map")
        if not self.timing_map.qualified or not self.timing_map.monotonic:
            raise ValidationError("Mastering identity requires a qualified timing map")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("mastering-audio-identity-v1", self)


@dataclass(frozen=True, slots=True)
class DeliveryAudioIdentity:
    """Container, selected stream, decoded PCM, and timing-map identity."""

    mastered_digest: str
    container_digest: str
    stream_index: int
    stream_codec: str
    encoder_fingerprint: str
    decoder_fingerprint: str
    decoded_pcm_digest: str
    decoded_format: str
    decoded_frame_count: int
    timing_map: FrameTimingMap
    metadata_fingerprint: str
    container_inspection_fingerprint: str
    codec_equivalence_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.mastered_digest, "mastered audio digest"),
            (self.container_digest, "delivery container digest"),
            (self.encoder_fingerprint, "delivery encoder fingerprint"),
            (self.decoder_fingerprint, "delivery decoder fingerprint"),
            (self.decoded_pcm_digest, "decoded delivery PCM digest"),
            (self.metadata_fingerprint, "delivery metadata fingerprint"),
            (
                self.container_inspection_fingerprint,
                "delivery container inspection fingerprint",
            ),
        ):
            _require_sha256(value, label)
        if self.codec_equivalence_fingerprint is not None:
            _require_sha256(
                self.codec_equivalence_fingerprint,
                "codec equivalence fingerprint",
            )
        if (
            self.stream_index < 0
            or not self.stream_codec
            or not self.decoded_format
            or self.decoded_frame_count != self.timing_map.destination_frame_count
        ):
            raise ValidationError("Delivery identity stream or audio shape is invalid")
        if not self.timing_map.qualified or not self.timing_map.monotonic:
            raise ValidationError("Delivery identity requires a qualified timing map")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("delivery-audio-identity-v1", self)


@dataclass(frozen=True, slots=True)
class AnalysisWindow:
    """One stable owned window shared by all participating recognizers."""

    window_id: str
    span: AudioSpan
    expected_segment_ids: tuple[str, ...]
    owned_frame_start: int
    owned_frame_end: int
    owned_token_start: int
    owned_token_end: int
    boundary_reason: str
    maximum_context_frames: int

    def __post_init__(self) -> None:
        if not self.window_id or not self.expected_segment_ids:
            raise ValidationError("Analysis window identity is incomplete")
        if len(set(self.expected_segment_ids)) != len(self.expected_segment_ids):
            raise ValidationError("Analysis window segment references must be unique")
        if (
            self.owned_frame_start < self.span.start_frame
            or self.owned_frame_end > self.span.end_frame
            or self.owned_frame_end <= self.owned_frame_start
        ):
            raise ValidationError("Analysis window frame ownership is invalid")
        if self.owned_token_start < 0 or self.owned_token_end <= self.owned_token_start:
            raise ValidationError("Analysis window token ownership is invalid")
        if self.maximum_context_frames <= 0:
            raise ValidationError("Maximum context frames must be positive")
        if self.span.end_frame - self.span.start_frame > self.maximum_context_frames:
            raise ValidationError("Analysis window exceeds its maximum context")


@dataclass(frozen=True, slots=True)
class AnalysisWindowPlan:
    """Deterministic shared windows for one canonical audio stream."""

    version: int
    canonical_audio_fingerprint: str
    spoken_text_plan_fingerprint: str
    stitching_fingerprint: str
    windows: tuple[AnalysisWindow, ...]

    def __post_init__(self) -> None:
        if self.version < 1 or not self.windows:
            raise ValidationError(
                "Analysis window plan must be versioned and non-empty"
            )
        for value, label in (
            (self.canonical_audio_fingerprint, "canonical audio fingerprint"),
            (self.spoken_text_plan_fingerprint, "spoken text plan fingerprint"),
            (self.stitching_fingerprint, "stitching fingerprint"),
        ):
            _require_sha256(value, label)
        if len({window.window_id for window in self.windows}) != len(self.windows):
            raise ValidationError("Analysis window identifiers must be unique")
        first = self.windows[0]
        if first.owned_token_start != 0:
            raise ValidationError("Analysis window token ownership must begin at zero")
        audio_shape = (first.span.audio_digest, first.span.sample_rate)
        previous_token_end = 0
        previous_frame_end = 0
        for window in self.windows:
            if (window.span.audio_digest, window.span.sample_rate) != audio_shape:
                raise ValidationError("Analysis windows must share one audio space")
            if window.owned_token_start != previous_token_end:
                raise ValidationError(
                    "Analysis window token ownership must be contiguous"
                )
            if window.owned_frame_start < previous_frame_end:
                raise ValidationError("Analysis window frame ownership cannot overlap")
            previous_token_end = window.owned_token_end
            previous_frame_end = window.owned_frame_end

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("analysis-window-plan-v1", self)


@dataclass(frozen=True, slots=True)
class ConversionIdentity:
    """Provenance for a converted or quantized model artifact."""

    source: str
    tool: str
    tool_version: str
    recipe_fingerprint: str
    precision_policy: str
    verified: bool

    def __post_init__(self) -> None:
        if not self.source or not self.tool or not self.tool_version:
            raise ValidationError("Model conversion provenance is incomplete")
        _require_sha256(self.recipe_fingerprint, "conversion recipe fingerprint")


@dataclass(frozen=True, slots=True)
class ModelArtifactIdentity:
    """Exact package, conversion, and model bytes used by an engine."""

    engine: str
    backend_package: str
    backend_version: str
    adapter_version: int
    worker_protocol_version: int
    converted_repository: str
    converted_revision: str
    converted_directory_fingerprint: str
    upstream_repository: str
    upstream_revision: str
    conversion: ConversionIdentity
    precision: str
    decode_fingerprint: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.engine,
                self.backend_package,
                self.backend_version,
                self.converted_repository,
                self.converted_revision,
                self.upstream_repository,
                self.upstream_revision,
                self.precision,
            )
        ):
            raise ValidationError("Model artifact identity is incomplete")
        if self.adapter_version < 1 or self.worker_protocol_version < 1:
            raise ValidationError("Adapter and worker versions must be positive")
        _require_sha256(
            self.converted_directory_fingerprint,
            "converted model directory fingerprint",
        )
        _require_sha256(self.decode_fingerprint, "decode fingerprint")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("model-artifact-identity-v1", self)


@dataclass(frozen=True, slots=True)
class ExecutionIdentity:
    """Redacted execution class used to qualify model evidence."""

    worker_artifact_digest: str
    lock_digest: str
    python_version: str
    os_family: str
    os_version: str
    architecture: str
    mlx_version: str | None
    metal_version: str | None
    device_class: str
    determinism_mode: str
    decode_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.worker_artifact_digest, "worker artifact digest")
        _require_sha256(self.lock_digest, "worker lock digest")
        if not self.python_version or not self.os_family or not self.architecture:
            raise ValidationError("Execution identity is incomplete")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("execution-identity-v1", self)


@dataclass(frozen=True, slots=True)
class RecognitionToken:
    """One normalized recognizer token and optional word timing."""

    text: str
    start_frame: int | None
    end_frame: int | None
    score: float | None
    score_kind: ScoreKind
    calibration_fingerprint: str

    def __post_init__(self) -> None:
        if not self.text or self.text != self.text.casefold():
            raise ValidationError("Recognition tokens must be normalized and non-empty")
        if (self.start_frame is None) != (self.end_frame is None):
            raise ValidationError("Recognition token timing must be complete or absent")
        if self.start_frame is not None and (
            self.start_frame < 0
            or self.end_frame is None
            or self.end_frame <= self.start_frame
        ):
            raise ValidationError("Recognition token timing is invalid")
        if self.score is not None and not math.isfinite(self.score):
            raise ValidationError("Recognition token score must be finite")
        if (
            self.score is not None
            and self.score_kind in {ScoreKind.PROBABILITY, ScoreKind.TDT_CONFIDENCE}
            and not 0 <= self.score <= 1
        ):
            raise ValidationError("Recognition token probability is out of range")
        if self.score_kind is ScoreKind.UNAVAILABLE and self.score is not None:
            raise ValidationError("Unavailable recognition scores must be absent")
        _require_sha256(
            self.calibration_fingerprint,
            "recognition token calibration fingerprint",
        )


@dataclass(frozen=True, slots=True)
class WhisperEvidence:
    average_log_probability: float | None
    compression_ratio: float | None
    no_speech_probability: float | None
    temperature: float | None


@dataclass(frozen=True, slots=True)
class ParakeetEvidence:
    sentence_confidence: float | None
    decoding: str
    chunk_duration_frames: int
    overlap_frames: int


@dataclass(frozen=True, slots=True)
class QwenEvidence:
    finish_reason: str
    prompt_tokens: int
    generation_tokens: int


type RecognitionEvidence = WhisperEvidence | ParakeetEvidence | QwenEvidence


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """Validated independent recognition evidence from exactly one engine."""

    engine: str
    model: ModelArtifactIdentity
    execution: ExecutionIdentity
    span: AudioSpan
    requested_language: str
    detected_language: str | None
    normalized_transcript_hash: str
    raw_transcript_hash: str
    score_calibration_fingerprint: str
    tokens: tuple[RecognitionToken, ...]
    evidence: RecognitionEvidence
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.engine != self.model.engine:
            raise ValidationError("Recognition engine does not match model identity")
        if self.requested_language != "en":
            raise ValidationError(
                "The initial speech-analysis release supports en only"
            )
        _require_sha256(
            self.normalized_transcript_hash,
            "normalized transcript hash",
        )
        _require_sha256(self.raw_transcript_hash, "raw transcript hash")
        _require_sha256(
            self.score_calibration_fingerprint,
            "recognition score calibration fingerprint",
        )
        if any(
            token.calibration_fingerprint != self.score_calibration_fingerprint
            for token in self.tokens
        ):
            raise ValidationError(
                "Recognition token score calibration fingerprints must match"
            )
        _require_known_reason_codes(self.issues)

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("recognition-result-v1", self)


@dataclass(frozen=True, slots=True)
class ForcedAlignmentUnit:
    """One forced word or character boundary in canonical frames."""

    text_hash: str
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        _require_sha256(self.text_hash, "forced alignment unit text hash")
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValidationError("Forced alignment unit timing is invalid")


@dataclass(frozen=True, slots=True)
class ForcedAlignmentResult:
    """Timing-only evidence that can never establish transcript authority."""

    engine: str
    model: ModelArtifactIdentity
    execution: ExecutionIdentity
    span: AudioSpan
    purpose: AlignmentPurpose
    aligner_text_hash: str
    expected_lexical_span_hash: str
    units: tuple[ForcedAlignmentUnit, ...]
    coverage_ratio: float
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.engine != self.model.engine:
            raise ValidationError("Forced-aligner engine does not match model identity")
        _require_sha256(self.aligner_text_hash, "aligner text hash")
        _require_sha256(
            self.expected_lexical_span_hash,
            "expected lexical span hash",
        )
        if not math.isfinite(self.coverage_ratio) or not 0 <= self.coverage_ratio <= 1:
            raise ValidationError(
                "Forced alignment coverage must be between zero and one"
            )
        _require_monotonic_units(self.units, self.span)
        _require_known_reason_codes(self.issues)

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("forced-alignment-result-v1", self)


@dataclass(frozen=True, slots=True)
class VerifiedTextSpan:
    """Consensus-authorized lexical span eligible for timing projection."""

    consensus_fingerprint: str
    token_start: int
    token_end: int
    lexical_span_hash: str

    def __post_init__(self) -> None:
        _require_sha256(self.consensus_fingerprint, "consensus fingerprint")
        _require_sha256(self.lexical_span_hash, "verified lexical span hash")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValidationError("Verified text span bounds are invalid")


@dataclass(frozen=True, slots=True)
class LexicalSpan:
    """Half-open lexical-token span used by consensus and repair selectors."""

    start: int
    end: int
    reason_codes: tuple[str, ...] = ()
    review_eligible: bool = False

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValidationError("Lexical span bounds are invalid")
        _require_known_reason_codes(self.reason_codes)


@dataclass(frozen=True, slots=True)
class TokenVote:
    """One engine's gated vote for an expected lexical token."""

    expected_index: int
    engine: str
    recognition_fingerprint: str
    state: VoteState
    recognized_token_hash: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.expected_index < 0 or not self.engine:
            raise ValidationError("Token vote identity is invalid")
        _require_sha256(self.recognition_fingerprint, "recognition fingerprint")
        if self.recognized_token_hash is not None:
            _require_sha256(self.recognized_token_hash, "recognized token hash")
        _require_known_reason_codes(self.reason_codes)


@dataclass(frozen=True, slots=True)
class AppliedLexicalEquivalence:
    """One reviewed directional alias used by a recognizer comparison."""

    engine: str
    recognition_fingerprint: str
    rule_fingerprint: str
    reason_code: str
    expected_start: int
    expected_end: int
    recognized_start: int
    recognized_end: int
    recognized_sequence_hash: str

    def __post_init__(self) -> None:
        reason_valid = (
            re.fullmatch(r"[a-z][a-z0-9_]{1,63}", self.reason_code) is not None
        )
        if not self.engine or not reason_valid:
            raise ValidationError("Applied lexical equivalence identity is invalid")
        for value, label in (
            (self.recognition_fingerprint, "recognition fingerprint"),
            (self.rule_fingerprint, "equivalence rule fingerprint"),
            (self.recognized_sequence_hash, "recognized equivalence sequence hash"),
        ):
            _require_sha256(value, label)
        if (
            self.expected_start < 0
            or self.expected_end <= self.expected_start
            or self.recognized_start < 0
            or self.recognized_end <= self.recognized_start
        ):
            raise ValidationError("Applied lexical equivalence bounds are invalid")


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    """Deterministic decision over independent recognition evidence only."""

    policy_fingerprint: str
    expected_tokens_hash: str
    equivalence_set_fingerprint: str
    recognition_fingerprints: tuple[str, ...]
    applied_equivalences: tuple[AppliedLexicalEquivalence, ...]
    votes: tuple[TokenVote, ...]
    accepted_spans: tuple[LexicalSpan, ...]
    rejected_spans: tuple[LexicalSpan, ...]
    disagreement_spans: tuple[LexicalSpan, ...]
    high_risk_spans: tuple[LexicalSpan, ...]
    escalation_reason: str
    outcome: ConsensusOutcome
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.policy_fingerprint, "consensus policy fingerprint")
        _require_sha256(self.expected_tokens_hash, "expected tokens hash")
        _require_sha256(
            self.equivalence_set_fingerprint,
            "consensus equivalence set fingerprint",
        )
        for fingerprint in self.recognition_fingerprints:
            _require_sha256(fingerprint, "recognition result fingerprint")
        if (
            tuple(sorted(self.recognition_fingerprints))
            != self.recognition_fingerprints
        ):
            raise ValidationError("Recognition fingerprints must use stable ordering")
        if any(
            item.recognition_fingerprint not in self.recognition_fingerprints
            for item in self.applied_equivalences
        ):
            raise ValidationError(
                "Applied equivalence must reference consensus recognition evidence"
            )
        if (
            tuple(
                sorted(
                    self.applied_equivalences,
                    key=lambda item: (
                        item.engine,
                        item.expected_start,
                        item.recognized_start,
                        item.rule_fingerprint,
                    ),
                )
            )
            != self.applied_equivalences
        ):
            raise ValidationError("Applied equivalences must use stable ordering")
        _require_known_reason_codes(self.reason_codes)

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("consensus-result-v1", self)


@dataclass(frozen=True, slots=True)
class SpeechVerification:
    """Common verification result for candidates, masters, and deliveries."""

    policy_fingerprint: str
    consensus_fingerprint: str
    forced_alignment_fingerprint: str | None
    signal_evidence_fingerprint: str | None
    artifact_digest: str
    scope: VerificationScope
    accepted: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.policy_fingerprint, "verification policy fingerprint"),
            (self.consensus_fingerprint, "verification consensus fingerprint"),
            (self.artifact_digest, "verified artifact digest"),
        ):
            _require_sha256(value, label)
        for optional, label in (
            (self.forced_alignment_fingerprint, "forced alignment fingerprint"),
            (self.signal_evidence_fingerprint, "signal evidence fingerprint"),
        ):
            if optional is not None:
                _require_sha256(optional, label)
        _require_known_reason_codes(self.reason_codes)
        if self.accepted == bool(self.reason_codes):
            raise ValidationError("Verification acceptance and reasons disagree")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-verification-v1", self)


KNOWN_REASON_CODES = frozenset(
    {
        "ambiguous_target_projection",
        "engine_decode_invalid",
        "engine_result_missing",
        "forced_alignment_incomplete",
        "invalid_engine_result",
        "lexical_deletion",
        "lexical_insertion",
        "lexical_substitution",
        "missing_required_engine",
        "persistent_valid_dissent",
        "recognition_match",
        "recognition_not_run",
        "unexpected_speech",
    }
)


def _require_sha256(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label.capitalize()} must be a lowercase SHA-256")


def _require_normalized_tokens(tokens: tuple[str, ...]) -> None:
    if any(not token or token != token.casefold() for token in tokens):
        raise ValidationError("Expected lexical tokens must be normalized")


def _require_known_reason_codes(reason_codes: tuple[str, ...]) -> None:
    unknown = tuple(code for code in reason_codes if code not in KNOWN_REASON_CODES)
    if unknown:
        raise ValidationError("Unknown speech-analysis reason code: " + unknown[0])
    if len(set(reason_codes)) != len(reason_codes):
        raise ValidationError("Speech-analysis reason codes must be unique")


def _require_monotonic_units(
    units: tuple[ForcedAlignmentUnit, ...], span: AudioSpan
) -> None:
    previous_end = span.start_frame
    for unit in units:
        if unit.start_frame < previous_end or unit.end_frame > span.end_frame:
            raise ValidationError(
                "Forced alignment units must be monotonic and bounded"
            )
        previous_end = unit.end_frame


__all__ = [
    "KNOWN_REASON_CODES",
    "AlignmentPurpose",
    "AnalysisWindow",
    "AnalysisWindowPlan",
    "AppliedLexicalEquivalence",
    "AudioSpan",
    "CanonicalAudioIdentity",
    "ClipClass",
    "ConsensusOutcome",
    "ConsensusResult",
    "ConversionIdentity",
    "DeliveryAudioIdentity",
    "ExecutionIdentity",
    "ForcedAlignmentResult",
    "ForcedAlignmentUnit",
    "FrameCoordinateMap",
    "LexicalSpan",
    "MasteringAudioIdentity",
    "ModelArtifactIdentity",
    "ParakeetEvidence",
    "QwenEvidence",
    "RecognitionEvidence",
    "RecognitionResult",
    "RecognitionToken",
    "ScoreKind",
    "SourceTextSpan",
    "SpeechVerification",
    "SpokenTextPlan",
    "SpokenTextSegment",
    "TextTransform",
    "TokenVote",
    "VerificationScope",
    "VerifiedTextSpan",
    "VoteState",
    "WhisperEvidence",
]
