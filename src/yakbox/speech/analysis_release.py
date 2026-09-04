"""Join, chapter, mastering, delivery, and strict release verification."""

from __future__ import annotations

import re
import wave
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from statistics import median

from yakbox._files import sha256_file
from yakbox.contracts import runtime_metadata
from yakbox.errors import SpeechAnalysisError, ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_models import (
    ClipClass,
    ConsensusOutcome,
    DeliveryAudioIdentity,
    FrameTimingMap,
    LexicalSpan,
    MasteringAudioIdentity,
    VerificationScope,
)
from yakbox.speech.analysis_pipeline import EnsembleAnalysis, SpeechAnalysisEnsemble
from yakbox.speech.analysis_repair import AnalyzedArtifactState

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MINIMUM_TIMING_MARKERS = 3
_SAFE_RELEASE_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")


def release_verification_path(output_root: Path, release_id: str) -> Path:
    """Return the release-only evidence path, separate from repair candidates."""
    if _SAFE_RELEASE_ID.fullmatch(release_id) is None:
        raise ValidationError("Release verification ID is not path-safe")
    return output_root.resolve() / "release" / "verified" / f"{release_id}.json"


@dataclass(frozen=True, slots=True)
class TimingMarkerObservation:
    """One measured marker correspondence across an audio transform."""

    source_frame: int
    destination_frame: int

    def __post_init__(self) -> None:
        if self.source_frame < 0 or self.destination_frame < 0:
            raise ValidationError("Timing marker frames cannot be negative")


def qualify_frame_timing_map(
    *,
    source_rate: int,
    destination_rate: int,
    source_frame_count: int,
    destination_frame_count: int,
    markers: tuple[TimingMarkerObservation, ...],
    maximum_residual_frames: int,
) -> FrameTimingMap:
    """Measure affine delay and uncertainty from controlled marker fixtures."""
    if (
        min(
            source_rate,
            destination_rate,
            source_frame_count,
            destination_frame_count,
        )
        <= 0
        or len(markers) < _MINIMUM_TIMING_MARKERS
        or maximum_residual_frames < 0
    ):
        raise ValidationError("Timing-map qualification needs three markers")
    ordered = tuple(sorted(markers, key=lambda item: item.source_frame))
    if any(
        previous.source_frame >= current.source_frame
        or previous.destination_frame >= current.destination_frame
        for previous, current in pairwise(ordered)
    ):
        raise ValidationError("Timing markers must be strictly monotonic")
    delays = tuple(
        item.destination_frame
        - round(item.source_frame * destination_rate / source_rate)
        for item in ordered
    )
    delay = round(median(delays))
    residual = max(abs(item - delay) for item in delays)
    if residual > maximum_residual_frames:
        raise ValidationError("Timing markers exceed the qualified residual")
    calibration = semantic_fingerprint(
        "frame-timing-calibration-v1",
        {
            "source_rate": source_rate,
            "destination_rate": destination_rate,
            "source_frame_count": source_frame_count,
            "destination_frame_count": destination_frame_count,
            "markers": ordered,
            "maximum_residual_frames": maximum_residual_frames,
        },
    )
    return FrameTimingMap(
        source_rate,
        destination_rate,
        source_frame_count,
        destination_frame_count,
        0,
        0,
        destination_rate,
        source_rate,
        delay,
        residual,
        True,
        True,
        calibration,
    )


@dataclass(frozen=True, slots=True)
class JoinPcmEvidence:
    """Independent hard signal measurements around one join."""

    sample_jump_ratio: float
    local_peak_change_db: float
    surrounding_silence_ms: float
    accepted: bool
    measurement_fingerprint: str

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.measurement_fingerprint) is None:
            raise ValidationError("Join PCM measurement fingerprint must be SHA-256")


@dataclass(frozen=True, slots=True)
class JoinVerification:
    """Signal, contextual lexical, and forced word-edge evidence for a join."""

    join_id: str
    pcm: JoinPcmEvidence
    analysis: EnsembleAnalysis
    accepted: bool
    reason_codes: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "speech-join-verification-v1",
            {
                "join_id": self.join_id,
                "pcm": self.pcm,
                "analysis": self.analysis.verification.fingerprint,
                "accepted": self.accepted,
                "reason_codes": self.reason_codes,
            },
        )


class JoinEvidenceStore:
    """Digest-bound contextual join evidence for localized repair reruns."""

    def __init__(self, ensemble: SpeechAnalysisEnsemble) -> None:
        self.ensemble = ensemble
        self._evidence: dict[str, JoinVerification] = {}

    async def verify(
        self,
        *,
        join_id: str,
        contextual_audio: Path,
        expected_tokens: tuple[str, ...],
        pcm: JoinPcmEvidence,
        language: str = "en",
    ) -> tuple[JoinVerification, bool]:
        key = semantic_fingerprint(
            "join-evidence-key-v1",
            {
                "join_id": join_id,
                "audio_digest": sha256_file(contextual_audio),
                "expected_tokens": expected_tokens,
                "pcm": pcm,
                "language": language,
                "ensemble": self.ensemble.fingerprint,
            },
        )
        cached = self._evidence.get(key)
        if cached is not None:
            return cached, True
        result = await verify_join(
            self.ensemble,
            join_id=join_id,
            contextual_audio=contextual_audio,
            expected_tokens=expected_tokens,
            pcm=pcm,
            language=language,
        )
        self._evidence[key] = result
        return result, False


@dataclass(frozen=True, slots=True)
class ExpectedChunkSpan:
    """Assembly-manifest projection from lexical tokens to a repair selector."""

    chunk_id: str
    token_start: int
    token_end: int
    source_path: str
    source_start_line: int
    source_end_line: int
    speaker: str
    profile: str
    before_join_id: str | None
    after_join_id: str | None

    def __post_init__(self) -> None:
        if (
            not self.chunk_id
            or self.token_start < 0
            or self.token_end <= self.token_start
            or self.source_start_line < 1
            or self.source_end_line < self.source_start_line
        ):
            raise ValidationError("Expected chunk token projection is invalid")


@dataclass(frozen=True, slots=True)
class StableRepairSelector:
    """Actionable, stable source selector for one verified mismatch."""

    chunk_id: str
    source_path: str
    source_start_line: int
    source_end_line: int
    speaker: str
    profile: str
    token_start: int
    token_end: int
    adjacent_join_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChapterDefect:
    """One rejected lexical span mapped back to source and assembly identity."""

    reason_codes: tuple[str, ...]
    lexical_span: LexicalSpan
    selector: StableRepairSelector


@dataclass(frozen=True, slots=True)
class ChapterVerification:
    """Exact chapter analysis plus actionable repair selectors."""

    chapter_id: str
    audio_digest: str
    analysis: EnsembleAnalysis
    defects: tuple[ChapterDefect, ...]

    @property
    def accepted(self) -> bool:
        return self.analysis.verification.accepted and not self.defects

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "speech-chapter-verification-v1",
            {
                "chapter_id": self.chapter_id,
                "audio_digest": self.audio_digest,
                "analysis": self.analysis.verification.fingerprint,
                "defects": self.defects,
            },
        )


@dataclass(frozen=True, slots=True)
class ChapterWindowInput:
    """One deterministic bounded chapter window and its owned token interval."""

    window_id: str
    audio: Path
    expected_tokens: tuple[str, ...]
    owned_token_start: int
    owned_token_end: int
    high_risk_spans: tuple[LexicalSpan, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.window_id
            or not self.expected_tokens
            or self.owned_token_start < 0
            or self.owned_token_end <= self.owned_token_start
            or self.owned_token_end - self.owned_token_start
            != len(self.expected_tokens)
        ):
            raise ValidationError("Chapter analysis window ownership is invalid")


@dataclass(frozen=True, slots=True)
class HierarchicalChapterVerification:
    """Bounded baseline analyses, escalations, timing, and global repair map."""

    chapter_id: str
    windows: tuple[tuple[str, EnsembleAnalysis], ...]
    defects: tuple[ChapterDefect, ...]

    @property
    def accepted(self) -> bool:
        return not self.defects and all(
            analysis.verification.accepted for _window_id, analysis in self.windows
        )

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "hierarchical-chapter-verification-v1",
            {
                "chapter_id": self.chapter_id,
                "windows": tuple(
                    (window_id, analysis.verification.fingerprint)
                    for window_id, analysis in self.windows
                ),
                "defects": self.defects,
            },
        )


@dataclass(frozen=True, slots=True)
class EvidenceInvalidation:
    """Exact evidence scopes invalidated by one repaired raw chunk."""

    raw_chunk_ids: tuple[str, ...]
    join_ids: tuple[str, ...]
    post_master_windows: tuple[tuple[int, int], ...]
    prior_master_digest: str
    full_master_stale: bool = True


@dataclass(frozen=True, slots=True)
class DeliveryTechnicalEvidence:
    """Independent stream, timing, signal, and chapter-boundary checks."""

    decoded_pcm_digest: str
    selected_stream_index: int
    channel_count: int
    duration_delta_frames: int
    codec_delay_frames: int
    leading_audio_accepted: bool
    trailing_audio_accepted: bool
    clipping_accepted: bool
    chapter_boundaries_accepted: bool
    inspection_fingerprint: str

    def __post_init__(self) -> None:
        if self.selected_stream_index < 0 or self.channel_count < 1:
            raise ValidationError("Delivery technical stream selection is invalid")
        for value in (self.decoded_pcm_digest, self.inspection_fingerprint):
            if _SHA256.fullmatch(value) is None:
                raise ValidationError("Delivery technical fingerprints must be SHA-256")

    @property
    def accepted(self) -> bool:
        return all(
            (
                self.leading_audio_accepted,
                self.trailing_audio_accepted,
                self.clipping_accepted,
                self.chapter_boundaries_accepted,
            )
        )


@dataclass(frozen=True, slots=True)
class DeliveryVerification:
    """Container identity and speech verification of decoded delivery PCM."""

    identity: DeliveryAudioIdentity
    technical: DeliveryTechnicalEvidence
    analysis: EnsembleAnalysis
    reused_lexical_evidence: bool

    @property
    def accepted(self) -> bool:
        return self.technical.accepted and self.analysis.verification.accepted


@dataclass(frozen=True, slots=True)
class ReleaseVerification:
    """Strict proof for one exact master and all exact delivery containers."""

    release_id: str
    mastering_identity: MasteringAudioIdentity
    master: ChapterVerification
    deliveries: tuple[DeliveryVerification, ...]
    state: AnalyzedArtifactState
    reason_codes: tuple[str, ...]
    master_cache_hit: bool = False
    delivery_cache_hits: tuple[bool, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.state is AnalyzedArtifactState.RELEASE_VERIFIED

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "speech-release-verification-v1",
            {
                "release_id": self.release_id,
                "mastering_identity": self.mastering_identity.fingerprint,
                "master": self.master.fingerprint,
                "deliveries": tuple(
                    item.identity.fingerprint for item in self.deliveries
                ),
                "delivery_verifications": tuple(
                    item.analysis.verification.fingerprint for item in self.deliveries
                ),
                "state": self.state,
                "reason_codes": self.reason_codes,
                "master_cache_hit": self.master_cache_hit,
                "delivery_cache_hits": self.delivery_cache_hits,
            },
        )


class ReleaseEvidenceStore:
    """Digest-bound master and delivery evidence with explicit reuse reporting."""

    def __init__(self, ensemble: SpeechAnalysisEnsemble) -> None:
        self.ensemble = ensemble
        self._masters: dict[str, ChapterVerification] = {}
        self._deliveries: dict[str, DeliveryVerification] = {}

    async def verify(
        self,
        *,
        release_id: str,
        mastering: MasteringAudioIdentity,
        mastered_audio: Path,
        expected_tokens: tuple[str, ...],
        chunk_spans: tuple[ExpectedChunkSpan, ...],
        deliveries: tuple[
            tuple[DeliveryAudioIdentity, DeliveryTechnicalEvidence, Path, Path], ...
        ],
        language: str = "en",
    ) -> ReleaseVerification:
        """Reuse evidence only when every identity input remains byte-exact."""
        if sha256_file(mastered_audio) != mastering.mastered_digest:
            raise ValidationError("Release master bytes differ from mastering identity")
        text_map = semantic_fingerprint(
            "release-expected-map-v1",
            {"tokens": expected_tokens, "chunks": chunk_spans, "language": language},
        )
        master_key = semantic_fingerprint(
            "release-master-evidence-key-v1",
            {
                "mastered_digest": mastering.mastered_digest,
                "identity": mastering.fingerprint,
                "expected_map": text_map,
            },
        )
        master = self._masters.get(master_key)
        master_hit = master is not None
        if master is None:
            master = await verify_chapter(
                self.ensemble,
                chapter_id=release_id,
                audio=mastered_audio,
                expected_tokens=expected_tokens,
                chunk_spans=chunk_spans,
                scope=VerificationScope.MASTER,
                language=language,
            )
            self._masters[master_key] = master
        delivery_results: list[DeliveryVerification] = []
        delivery_hits: list[bool] = []
        for identity, technical, container, decoded_audio in deliveries:
            if (
                identity.mastered_digest != mastering.mastered_digest
                or sha256_file(container) != identity.container_digest
                or sha256_file(decoded_audio) != identity.decoded_pcm_digest
                or technical.decoded_pcm_digest != identity.decoded_pcm_digest
                or technical.selected_stream_index != identity.stream_index
                or technical.inspection_fingerprint
                != identity.container_inspection_fingerprint
            ):
                raise ValidationError(
                    "Delivery bytes differ from their release identity"
                )
            key = semantic_fingerprint(
                "release-delivery-evidence-key-v1",
                {
                    "identity": identity.fingerprint,
                    "technical": technical,
                    "expected_map": text_map,
                },
            )
            delivery = self._deliveries.get(key)
            delivery_hits.append(delivery is not None)
            if delivery is None:
                decoded = await verify_chapter(
                    self.ensemble,
                    chapter_id=release_id,
                    audio=decoded_audio,
                    expected_tokens=expected_tokens,
                    chunk_spans=chunk_spans,
                    scope=VerificationScope.DELIVERY,
                    language=language,
                )
                delivery = DeliveryVerification(
                    identity,
                    technical,
                    decoded.analysis,
                    not any(
                        stage.startswith("recognition:")
                        for stage in decoded.analysis.cache.miss_stages
                    ),
                )
                self._deliveries[key] = delivery
            delivery_results.append(delivery)
        reasons = []
        if not master.accepted:
            reasons.append("master_speech_rejected")
        if any(not item.accepted for item in delivery_results):
            reasons.append("delivery_speech_rejected")
        state = (
            AnalyzedArtifactState.RELEASE_VERIFIED
            if not reasons
            else AnalyzedArtifactState.REPAIR_CANDIDATE
        )
        return ReleaseVerification(
            release_id,
            mastering,
            master,
            tuple(delivery_results),
            state,
            tuple(reasons),
            master_hit,
            tuple(delivery_hits),
        )


async def verify_join(
    ensemble: SpeechAnalysisEnsemble,
    *,
    join_id: str,
    contextual_audio: Path,
    expected_tokens: tuple[str, ...],
    pcm: JoinPcmEvidence,
    language: str = "en",
) -> JoinVerification:
    """Combine hard PCM checks with contextual consensus and forced timing."""
    analysis = await ensemble.analyze(
        contextual_audio,
        expected_tokens=expected_tokens,
        clip_class=ClipClass.JOIN,
        scope=VerificationScope.CANDIDATE,
        language=language,
        high_risk=True,
        signal_evidence_fingerprint=pcm.measurement_fingerprint,
    )
    reasons = list(analysis.verification.reason_codes)
    if not pcm.accepted:
        reasons.append("join_signal_rejected")
    accepted = pcm.accepted and analysis.verification.accepted
    return JoinVerification(
        join_id,
        pcm,
        analysis,
        accepted,
        tuple(dict.fromkeys(reasons)),
    )


async def verify_chapter(
    ensemble: SpeechAnalysisEnsemble,
    *,
    chapter_id: str,
    audio: Path,
    expected_tokens: tuple[str, ...],
    chunk_spans: tuple[ExpectedChunkSpan, ...],
    scope: VerificationScope = VerificationScope.MASTER,
    high_risk_spans: tuple[LexicalSpan, ...] = (),
    language: str = "en",
) -> ChapterVerification:
    """Run strict baseline recognition and map every mismatch to source."""
    _validate_chunk_coverage(chunk_spans, len(expected_tokens))
    analysis = await ensemble.analyze(
        audio,
        expected_tokens=expected_tokens,
        clip_class=ClipClass.CHAPTER,
        scope=scope,
        language=language,
        high_risk=bool(high_risk_spans),
        require_forced_alignment=True,
    )
    spans = analysis.consensus.rejected_spans
    if not spans and analysis.consensus.outcome is ConsensusOutcome.REJECTED:
        spans = analysis.consensus.disagreement_spans
    defects = tuple(
        ChapterDefect(
            span.reason_codes or analysis.consensus.reason_codes,
            span,
            selector_for_span(span, chunk_spans),
        )
        for span in spans
    )
    return ChapterVerification(
        chapter_id,
        sha256_file(audio),
        analysis,
        defects,
    )


async def verify_chapter_hierarchical(
    ensemble: SpeechAnalysisEnsemble,
    *,
    chapter_id: str,
    windows: tuple[ChapterWindowInput, ...],
    chunk_spans: tuple[ExpectedChunkSpan, ...],
    language: str = "en",
) -> HierarchicalChapterVerification:
    """Analyze shared bounded windows and merge defects into global selectors."""
    if not windows or windows[0].owned_token_start != 0:
        raise ValidationError("Hierarchical chapter windows must begin at token zero")
    if any(
        previous.owned_token_end != current.owned_token_start
        for previous, current in pairwise(windows)
    ):
        raise ValidationError("Hierarchical chapter token ownership must be contiguous")
    _validate_chunk_coverage(chunk_spans, windows[-1].owned_token_end)
    analyzed: list[tuple[str, EnsembleAnalysis]] = []
    defects: list[ChapterDefect] = []
    for window in windows:
        analysis = await ensemble.analyze(
            window.audio,
            expected_tokens=window.expected_tokens,
            clip_class=ClipClass.CHAPTER,
            scope=VerificationScope.MASTER,
            language=language,
            high_risk=bool(window.high_risk_spans),
            require_forced_alignment=True,
        )
        analyzed.append((window.window_id, analysis))
        local_spans = analysis.consensus.rejected_spans
        if not local_spans and analysis.consensus.outcome is ConsensusOutcome.REJECTED:
            local_spans = analysis.consensus.disagreement_spans
        for local in local_spans:
            global_span = LexicalSpan(
                window.owned_token_start + local.start,
                window.owned_token_start + local.end,
                local.reason_codes or analysis.consensus.reason_codes,
                local.review_eligible,
            )
            defects.append(
                ChapterDefect(
                    global_span.reason_codes,
                    global_span,
                    selector_for_span(global_span, chunk_spans),
                )
            )
    return HierarchicalChapterVerification(chapter_id, tuple(analyzed), tuple(defects))


def selector_for_span(
    span: LexicalSpan,
    chunk_spans: tuple[ExpectedChunkSpan, ...],
) -> StableRepairSelector:
    """Return one deterministic repair selector for a lexical mismatch."""
    matches = tuple(
        item
        for item in chunk_spans
        if item.token_start < span.end and span.start < item.token_end
    )
    if not matches:
        raise SpeechAnalysisError("Mismatch has no assembly-manifest source projection")
    primary = matches[0]
    joins = tuple(
        dict.fromkeys(
            join
            for item in matches
            for join in (item.before_join_id, item.after_join_id)
            if join is not None
        )
    )
    return StableRepairSelector(
        primary.chunk_id,
        primary.source_path,
        primary.source_start_line,
        primary.source_end_line,
        primary.speaker,
        primary.profile,
        span.start,
        span.end,
        joins,
    )


def mastering_identity(
    *,
    raw_audio: Path,
    mastered_audio: Path,
    raw_format: str,
    mastered_format: str,
    ffmpeg_fingerprint: str,
    filter_graph_fingerprint: str,
    timing_map: FrameTimingMap,
    mapping_validation_fingerprint: str,
) -> MasteringAudioIdentity:
    """Bind measured transform evidence to exact raw and mastered bytes."""
    raw_rate, raw_frames = _wav_shape(raw_audio)
    mastered_rate, mastered_frames = _wav_shape(mastered_audio)
    if (raw_rate, raw_frames) != (
        timing_map.source_rate,
        timing_map.source_frame_count,
    ) or (mastered_rate, mastered_frames) != (
        timing_map.destination_rate,
        timing_map.destination_frame_count,
    ):
        raise ValidationError("Measured mastering map does not match the exact WAVs")
    return MasteringAudioIdentity(
        sha256_file(raw_audio),
        sha256_file(mastered_audio),
        raw_format,
        mastered_format,
        raw_frames,
        mastered_frames,
        ffmpeg_fingerprint,
        filter_graph_fingerprint,
        timing_map,
        mapping_validation_fingerprint,
    )


def delivery_identity(
    *,
    mastered_audio: Path,
    container: Path,
    decoded_audio: Path,
    stream_index: int,
    stream_codec: str,
    encoder_fingerprint: str,
    decoder_fingerprint: str,
    decoded_format: str,
    timing_map: FrameTimingMap,
    metadata_fingerprint: str,
    container_inspection_fingerprint: str,
) -> DeliveryAudioIdentity:
    """Bind one selected container stream to its exact decoded canonical PCM."""
    master_rate, master_frames = _wav_shape(mastered_audio)
    decoded_rate, decoded_frames = _wav_shape(decoded_audio)
    if (master_rate, master_frames) != (
        timing_map.source_rate,
        timing_map.source_frame_count,
    ) or (decoded_rate, decoded_frames) != (
        timing_map.destination_rate,
        timing_map.destination_frame_count,
    ):
        raise ValidationError("Delivery timing map does not match decoded audio")
    return DeliveryAudioIdentity(
        sha256_file(mastered_audio),
        sha256_file(container),
        stream_index,
        stream_codec,
        encoder_fingerprint,
        decoder_fingerprint,
        sha256_file(decoded_audio),
        decoded_format,
        decoded_frames,
        timing_map,
        metadata_fingerprint,
        container_inspection_fingerprint,
    )


def map_repair_windows_to_master(
    identity: MasteringAudioIdentity,
    windows: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Map repair windows outward or fail closed to broader verification."""
    return tuple(
        identity.timing_map.map_span_outward(start, end) for start, end in windows
    )


def invalidation_for_repair(
    *,
    raw_chunk_ids: tuple[str, ...],
    affected_join_ids: tuple[str, ...],
    raw_windows: tuple[tuple[int, int], ...],
    prior_master: MasteringAudioIdentity,
) -> EvidenceInvalidation:
    """Invalidate exact raw, join, mapped-window, and prior full-master evidence."""
    return EvidenceInvalidation(
        tuple(sorted(set(raw_chunk_ids))),
        tuple(sorted(set(affected_join_ids))),
        map_repair_windows_to_master(prior_master, raw_windows),
        prior_master.mastered_digest,
    )


async def verify_release(
    ensemble: SpeechAnalysisEnsemble,
    *,
    release_id: str,
    mastering: MasteringAudioIdentity,
    mastered_audio: Path,
    expected_tokens: tuple[str, ...],
    chunk_spans: tuple[ExpectedChunkSpan, ...],
    deliveries: tuple[
        tuple[DeliveryAudioIdentity, DeliveryTechnicalEvidence, Path, Path], ...
    ],
    language: str = "en",
) -> ReleaseVerification:
    """Verify exact master and decoded bytes for every delivery artifact."""
    return await ReleaseEvidenceStore(ensemble).verify(
        release_id=release_id,
        mastering=mastering,
        mastered_audio=mastered_audio,
        expected_tokens=expected_tokens,
        chunk_spans=chunk_spans,
        deliveries=deliveries,
        language=language,
    )


def release_report(value: ReleaseVerification) -> dict[str, object]:
    """Serialize state explicitly so a candidate cannot masquerade as a release."""
    return {
        **runtime_metadata("speech-release-verification"),
        "release_id": value.release_id,
        "fingerprint": value.fingerprint,
        "state": value.state.value,
        "accepted": value.accepted,
        "mastering_identity": {
            **asdict(value.mastering_identity),
            "fingerprint": value.mastering_identity.fingerprint,
        },
        "master": _chapter_value(value.master),
        "deliveries": [
            {
                "identity": {
                    **asdict(item.identity),
                    "fingerprint": item.identity.fingerprint,
                },
                "technical": asdict(item.technical),
                "verification_fingerprint": item.analysis.verification.fingerprint,
                "accepted": item.accepted,
                "reused_lexical_evidence": item.reused_lexical_evidence,
            }
            for item in value.deliveries
        ],
        "reason_codes": list(value.reason_codes),
        "master_cache_hit": value.master_cache_hit,
        "delivery_cache_hits": list(value.delivery_cache_hits),
    }


def release_status_message(value: ReleaseVerification) -> str:
    """Return wording that never labels a repair candidate as a release."""
    if value.state is AnalyzedArtifactState.RELEASE_VERIFIED:
        return f"Release verified: {value.release_id}"
    return f"Repair candidate is not release verified: {value.release_id}"


def require_release_verified(value: ReleaseVerification) -> None:
    """Fail a release boundary unless the exact report is release verified."""
    if value.state is not AnalyzedArtifactState.RELEASE_VERIFIED or not value.accepted:
        raise SpeechAnalysisError(release_status_message(value))


def join_report(value: JoinVerification) -> dict[str, object]:
    """Serialize contextual join evidence without embedding transcript text."""
    return {
        **runtime_metadata("speech-join-verification"),
        "join_id": value.join_id,
        "fingerprint": value.fingerprint,
        "state": AnalyzedArtifactState.REPAIR_CANDIDATE.value,
        "accepted": value.accepted,
        "pcm": asdict(value.pcm),
        "speech_verification_fingerprint": value.analysis.verification.fingerprint,
        "consensus_fingerprint": value.analysis.consensus.fingerprint,
        "forced_alignment_fingerprint": (
            value.analysis.forced_alignment.fingerprint
            if value.analysis.forced_alignment is not None
            else None
        ),
        "reason_codes": list(value.reason_codes),
    }


def chapter_report(value: ChapterVerification) -> dict[str, object]:
    """Serialize exact chapter evidence and stable repair selectors."""
    return {
        **runtime_metadata("speech-chapter-verification"),
        **_chapter_value(value),
        "scope": value.analysis.verification.scope.value,
        "consensus_fingerprint": value.analysis.consensus.fingerprint,
        "forced_alignment_fingerprint": (
            value.analysis.forced_alignment.fingerprint
            if value.analysis.forced_alignment is not None
            else None
        ),
    }


def _chapter_value(value: ChapterVerification) -> dict[str, object]:
    return {
        "chapter_id": value.chapter_id,
        "audio_digest": value.audio_digest,
        "fingerprint": value.fingerprint,
        "verification_fingerprint": value.analysis.verification.fingerprint,
        "accepted": value.accepted,
        "defects": [
            {
                "reason_codes": list(item.reason_codes),
                "lexical_span": asdict(item.lexical_span),
                "selector": asdict(item.selector),
            }
            for item in value.defects
        ],
    }


def _validate_chunk_coverage(
    spans: tuple[ExpectedChunkSpan, ...], total_tokens: int
) -> None:
    if not spans or spans[0].token_start != 0 or spans[-1].token_end != total_tokens:
        raise ValidationError("Assembly chunk mapping does not cover chapter tokens")
    if any(
        previous.token_end != current.token_start
        for previous, current in pairwise(spans)
    ):
        raise ValidationError("Assembly chunk token mapping must be contiguous")


def _wav_shape(path: Path) -> tuple[int, int]:
    try:
        with wave.open(str(path), "rb") as reader:
            return reader.getframerate(), reader.getnframes()
    except (OSError, EOFError, wave.Error) as error:
        raise ValidationError(f"Cannot inspect release WAV {path}: {error}") from error


__all__ = [
    "ChapterDefect",
    "ChapterVerification",
    "ChapterWindowInput",
    "DeliveryTechnicalEvidence",
    "DeliveryVerification",
    "EvidenceInvalidation",
    "ExpectedChunkSpan",
    "HierarchicalChapterVerification",
    "JoinEvidenceStore",
    "JoinPcmEvidence",
    "JoinVerification",
    "ReleaseEvidenceStore",
    "ReleaseVerification",
    "StableRepairSelector",
    "TimingMarkerObservation",
    "chapter_report",
    "delivery_identity",
    "invalidation_for_repair",
    "join_report",
    "map_repair_windows_to_master",
    "mastering_identity",
    "qualify_frame_timing_map",
    "release_report",
    "release_status_message",
    "release_verification_path",
    "require_release_verified",
    "selector_for_span",
    "verify_chapter",
    "verify_chapter_hierarchical",
    "verify_join",
    "verify_release",
]
