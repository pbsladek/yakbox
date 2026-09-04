"""Strict ensemble orchestration for candidates, repairs, and release audio."""

from __future__ import annotations

import asyncio
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from yakbox._files import sha256_file
from yakbox.errors import SpeechAnalysisError, ValidationError
from yakbox.speech.analysis_cache import (
    ConsensusCacheIdentity,
    ForcedAlignmentCacheIdentity,
    LayeredEvidenceCache,
    RecognitionCacheIdentity,
    VerificationCacheIdentity,
)
from yakbox.speech.analysis_fingerprints import semantic_fingerprint, text_fingerprint
from yakbox.speech.analysis_models import (
    AlignmentPurpose,
    AudioSpan,
    ClipClass,
    ConsensusOutcome,
    ConsensusResult,
    ForcedAlignmentResult,
    LexicalSpan,
    RecognitionResult,
    SpeechVerification,
    VerificationScope,
    VerifiedTextSpan,
    VoteState,
)
from yakbox.speech.analysis_policy import CalibrationTable, SpeechAnalysisPolicy
from yakbox.speech.analysis_serialization import (
    consensus_from_report,
    consensus_report,
    forced_alignment_from_report,
    forced_alignment_report,
    recognition_from_report,
    recognition_report,
    verification_from_report,
    verification_report,
)
from yakbox.speech.analysis_services import ForcedAligner, SpeechRecognizer
from yakbox.speech.consensus import evaluate_consensus
from yakbox.speech.normalization import EquivalenceSet

_MINIMUM_BOUNDARY_OBSERVATIONS = 3


@dataclass(frozen=True, slots=True)
class AnalysisCacheTrace:
    """Exact stage reuse and first cache miss for one analysis request."""

    hit_stages: tuple[str, ...]
    miss_stages: tuple[str, ...]

    @property
    def first_miss_stage(self) -> str | None:
        return self.miss_stages[0] if self.miss_stages else None


class AnalysisGateError(SpeechAnalysisError):
    """Terminal gate failure carrying the exact stage and cache trace."""

    def __init__(self, stage: str, message: str, cache: AnalysisCacheTrace) -> None:
        super().__init__(message)
        self.stage = stage
        self.cache = cache


@dataclass(frozen=True, slots=True)
class EnsembleAnalysis:
    """Independent recognition, consensus, timing, and terminal verification."""

    recognitions: tuple[RecognitionResult, ...]
    consensus: ConsensusResult
    forced_alignment: ForcedAlignmentResult | None
    verification: SpeechVerification
    cache: AnalysisCacheTrace


@dataclass(frozen=True, slots=True)
class BoundaryObservation:
    """One independent proposal for target speech boundaries in canonical frames."""

    source: str
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if (
            not self.source
            or self.start_frame < 0
            or self.end_frame <= self.start_frame
        ):
            raise ValidationError("Boundary observations must be named and ordered")


@dataclass(frozen=True, slots=True)
class BoundaryAgreement:
    """Agreement envelope; timestamps are deliberately never averaged."""

    observations: tuple[BoundaryObservation, ...]
    start_envelope: tuple[int, int]
    end_envelope: tuple[int, int]
    tolerance_frames: int
    safe_crop_start: int
    safe_crop_end: int

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-boundary-agreement-v1", self)


@dataclass(frozen=True, slots=True)
class CarrierExtractionEvidence:
    """Proof that carrier and final crop both passed independent analysis."""

    candidate_id: str
    target_token_start: int
    target_token_end: int
    carrier: EnsembleAnalysis
    extracted: EnsembleAnalysis
    boundary: BoundaryAgreement
    terminal_reason: str

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "carrier-extraction-evidence-v1",
            {
                "candidate_id": self.candidate_id,
                "target_token_start": self.target_token_start,
                "target_token_end": self.target_token_end,
                "carrier_verification": self.carrier.verification.fingerprint,
                "extracted_verification": self.extracted.verification.fingerprint,
                "boundary": self.boundary.fingerprint,
                "terminal_reason": self.terminal_reason,
            },
        )


class SpeechAnalysisEnsemble:
    """One strict decision engine shared by every speech-analysis scope."""

    def __init__(
        self,
        *,
        recognizers: Mapping[str, SpeechRecognizer],
        forced_aligner: ForcedAligner,
        policy: SpeechAnalysisPolicy,
        calibration: CalibrationTable,
        equivalences: EquivalenceSet,
        evidence_cache: LayeredEvidenceCache | None = None,
    ) -> None:
        by_name = dict(recognizers)
        if set(by_name) != {
            *policy.baseline_recognizers,
            policy.escalation_recognizer,
        }:
            raise ValidationError(
                "Ensemble recognizers must match the policy engine set"
            )
        if len({item.fingerprint for item in by_name.values()}) != len(by_name):
            raise ValidationError("Recognizer fingerprints must be unique")
        self._recognizers = by_name
        self._forced_aligner = forced_aligner
        self._policy = policy
        self._calibration = calibration
        self._equivalences = equivalences
        self._evidence_cache = evidence_cache
        self._recognition_cache: dict[str, RecognitionResult] = {}
        self._consensus_cache: dict[str, ConsensusResult] = {}
        self._consensus_escalation_cache: set[str] = set()
        self._alignment_cache: dict[str, ForcedAlignmentResult] = {}
        self._verification_cache: dict[str, SpeechVerification] = {}

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "speech-analysis-ensemble-v1",
            {
                "recognizers": tuple(
                    (name, item.fingerprint)
                    for name, item in sorted(self._recognizers.items())
                ),
                "forced_aligner": self._forced_aligner.fingerprint,
                "policy": self._policy.fingerprint,
                "calibration": self._calibration.fingerprint,
                "equivalences": self._equivalences.fingerprint,
            },
        )

    async def analyze(
        self,
        audio: Path,
        *,
        expected_tokens: tuple[str, ...],
        clip_class: ClipClass,
        scope: VerificationScope,
        language: str = "en",
        span: AudioSpan | None = None,
        high_risk: bool = False,
        repair: bool = False,
        require_forced_alignment: bool = True,
        signal_evidence_fingerprint: str | None = None,
    ) -> EnsembleAnalysis:
        """Run the policy without passing expected text to any recognizer."""
        if not expected_tokens or any(
            not token or token != token.casefold() for token in expected_tokens
        ):
            raise ValidationError("Ensemble expected tokens must be normalized")
        audio_span = span or _wav_span(audio)
        hits: list[str] = []
        misses: list[str] = []
        baseline = await asyncio.gather(
            *(
                self._recognize(
                    name,
                    audio,
                    language=language,
                    span=audio_span,
                    hits=hits,
                    misses=misses,
                )
                for name in self._policy.baseline_recognizers
            )
        )
        recognitions = tuple(baseline)
        consensus = self._consensus(
            expected_tokens,
            recognitions,
            clip_class=clip_class,
            high_risk=high_risk,
            repair=repair,
            hits=hits,
            misses=misses,
        )
        if consensus is None:
            escalated = await self._recognize(
                self._policy.escalation_recognizer,
                audio,
                language=language,
                span=audio_span,
                hits=hits,
                misses=misses,
            )
            recognitions = (*recognitions, escalated)
            consensus = self._consensus(
                expected_tokens,
                recognitions,
                clip_class=clip_class,
                high_risk=high_risk,
                repair=repair,
                hits=hits,
                misses=misses,
            )
        if consensus is None:
            raise SpeechAnalysisError(
                "Escalated analysis did not reach a terminal result"
            )
        accepted = consensus.outcome is ConsensusOutcome.ACCEPTED
        reasons = list(consensus.reason_codes)
        if accepted and clip_class is ClipClass.ONE_WORD:
            required = set(self._policy.one_word_required_recognizers)
            matching = {
                vote.engine for vote in consensus.votes if vote.state is VoteState.MATCH
            }
            if not required <= matching:
                accepted = False
                reasons.append("missing_required_engine")
        alignment: ForcedAlignmentResult | None = None
        if accepted and require_forced_alignment:
            alignment = await self._align(
                audio,
                expected_tokens=expected_tokens,
                consensus=consensus,
                language=language,
                span=audio_span,
                hits=hits,
                misses=misses,
            )
            if (
                alignment.issues
                or alignment.coverage_ratio != 1
                or (len(alignment.units) != len(expected_tokens))
            ):
                accepted = False
                reasons.append("forced_alignment_incomplete")
        verification = self._verification(
            consensus=consensus,
            alignment=alignment,
            artifact_digest=audio_span.audio_digest,
            scope=scope,
            accepted=accepted,
            reason_codes=tuple(dict.fromkeys(reasons)),
            signal_evidence_fingerprint=signal_evidence_fingerprint,
            hits=hits,
            misses=misses,
        )
        return EnsembleAnalysis(
            recognitions=tuple(sorted(recognitions, key=lambda item: item.engine)),
            consensus=consensus,
            forced_alignment=alignment,
            verification=verification,
            cache=AnalysisCacheTrace(tuple(hits), tuple(misses)),
        )

    async def verify_carrier_extraction(
        self,
        *,
        candidate_id: str,
        carrier_audio: Path,
        extracted_audio: Path,
        carrier_tokens: tuple[str, ...],
        target_token_start: int,
        target_token_end: int,
        vad: BoundaryObservation,
        waveform: BoundaryObservation,
        tolerance_frames: int,
        language: str = "en",
        signal_evidence_fingerprint: str | None = None,
    ) -> CarrierExtractionEvidence:
        """Verify the complete carrier, exact target projection, and final crop."""
        if (
            target_token_start < 0
            or target_token_end <= target_token_start
            or target_token_end > len(carrier_tokens)
        ):
            raise ValidationError("Carrier target token indices are invalid")
        target = carrier_tokens[target_token_start:target_token_end]
        carrier = await self.analyze(
            carrier_audio,
            expected_tokens=carrier_tokens,
            clip_class=ClipClass.SENTENCE,
            scope=VerificationScope.CANDIDATE,
            language=language,
            high_risk=True,
            repair=True,
        )
        if not carrier.verification.accepted or carrier.forced_alignment is None:
            raise AnalysisGateError(
                "carrier_verification",
                "Carrier did not pass independent ensemble analysis",
                carrier.cache,
            )
        try:
            observations = _target_boundary_observations(
                carrier,
                target_token_start=target_token_start,
                target_token_end=target_token_end,
            )
            boundary = boundary_agreement(
                (*observations, vad, waveform),
                tolerance_frames=tolerance_frames,
            )
        except SpeechAnalysisError as error:
            raise AnalysisGateError(
                "boundary_agreement",
                str(error),
                carrier.cache,
            ) from error
        clip_class = ClipClass.ONE_WORD if len(target) == 1 else ClipClass.SHORT_PHRASE
        extracted = await self.analyze(
            extracted_audio,
            expected_tokens=target,
            clip_class=clip_class,
            scope=VerificationScope.CANDIDATE,
            language=language,
            high_risk=True,
            repair=True,
            signal_evidence_fingerprint=signal_evidence_fingerprint,
        )
        if not extracted.verification.accepted:
            combined = AnalysisCacheTrace(
                (*carrier.cache.hit_stages, *extracted.cache.hit_stages),
                (*carrier.cache.miss_stages, *extracted.cache.miss_stages),
            )
            raise AnalysisGateError(
                "crop_verification",
                "Extracted crop failed independent verification",
                combined,
            )
        return CarrierExtractionEvidence(
            candidate_id,
            target_token_start,
            target_token_end,
            carrier,
            extracted,
            boundary,
            "accepted_all_required_gates",
        )

    async def _recognize(
        self,
        engine: str,
        audio: Path,
        *,
        language: str,
        span: AudioSpan,
        hits: list[str],
        misses: list[str],
    ) -> RecognitionResult:
        recognizer = self._recognizers.get(engine)
        if recognizer is None:
            raise SpeechAnalysisError(f"No recognizer is configured for {engine!r}")
        key = RecognitionCacheIdentity(
            model_fingerprint=recognizer.fingerprint,
            execution_fingerprint=recognizer.fingerprint,
            canonical_audio_fingerprint=span.audio_digest,
            span=span,
            language=language,
            preprocessing_fingerprint=semantic_fingerprint(
                "canonical-pcm-preprocessing-v1",
                {"sample_rate": span.sample_rate, "channels": 1},
            ),
            decode_settings_fingerprint=recognizer.fingerprint,
        ).fingerprint
        cached = self._recognition_cache.get(key)
        stage = f"recognition:{engine}"
        if cached is not None:
            hits.append(stage)
            return cached
        if self._evidence_cache is not None:

            async def produce() -> dict[str, object]:
                value = await recognizer.recognize(audio, language=language, span=span)
                _validate_recognition_result(value, engine=engine, span=span)
                return recognition_report(value)

            entry, lookup = await self._evidence_cache.get_or_compute(
                stage="recognition",
                key=key,
                dependencies=(),
                producer=produce,
                validator=recognition_from_report,
            )
            result = recognition_from_report(entry.evidence)
            (hits if lookup.hit else misses).append(stage)
            self._recognition_cache[key] = result
            return result
        misses.append(stage)
        result = await recognizer.recognize(audio, language=language, span=span)
        _validate_recognition_result(result, engine=engine, span=span)
        self._recognition_cache[key] = result
        return result

    def _consensus(
        self,
        expected_tokens: tuple[str, ...],
        recognitions: tuple[RecognitionResult, ...],
        *,
        clip_class: ClipClass,
        high_risk: bool,
        repair: bool,
        hits: list[str],
        misses: list[str],
    ) -> ConsensusResult | None:
        expected_hash = text_fingerprint("\u001f".join(expected_tokens))
        decision_policy = semantic_fingerprint(
            "consensus-request-policy-v1",
            {
                "analysis_policy": self._policy.fingerprint,
                "clip_class": clip_class,
                "high_risk": high_risk,
                "repair": repair,
            },
        )
        key = ConsensusCacheIdentity(
            recognition_fingerprints=tuple(
                sorted(item.fingerprint for item in recognitions)
            ),
            expected_tokens_hash=expected_hash,
            policy_fingerprint=decision_policy,
            equivalence_fingerprint=self._equivalences.fingerprint,
            calibration_fingerprint=self._calibration.fingerprint,
        ).fingerprint
        cached = self._consensus_cache.get(key)
        if cached is not None:
            hits.append("consensus")
            return cached
        if self._evidence_cache is not None:
            entry, lookup = self._evidence_cache.lookup("consensus", key)
            if entry is not None:
                result = consensus_from_report(entry.evidence)
                self._consensus_cache[key] = result
                hits.append("consensus")
                return result
            escalation, escalation_lookup = self._evidence_cache.lookup(
                "consensus-escalation", key
            )
            if escalation is not None:
                self._consensus_escalation_cache.add(key)
                hits.append("consensus:escalation")
                return None
            # Quarantine and miss are both recomputed; neither is evidence.
            del lookup, escalation_lookup
        if key in self._consensus_escalation_cache:
            hits.append("consensus:escalation")
            return None
        misses.append("consensus")
        evaluation = evaluate_consensus(
            expected_tokens=expected_tokens,
            recognitions=recognitions,
            clip_class=clip_class,
            policy=self._policy,
            calibration=self._calibration,
            equivalences=self._equivalences,
            high_risk_spans=(() if not high_risk else (_whole_span(expected_tokens),)),
            repair=repair,
        )
        if evaluation.result is not None:
            self._consensus_cache[key] = evaluation.result
            if self._evidence_cache is not None:
                self._evidence_cache.store(
                    stage="consensus",
                    key=key,
                    evidence=consensus_report(evaluation.result),
                    dependencies=tuple(
                        sorted(item.fingerprint for item in recognitions)
                    ),
                    validator=consensus_from_report,
                )
        else:
            self._consensus_escalation_cache.add(key)
            if self._evidence_cache is not None:
                self._evidence_cache.store(
                    stage="consensus-escalation",
                    key=key,
                    evidence={
                        "terminal": False,
                        "recognition_fingerprints": [
                            item.fingerprint
                            for item in sorted(
                                recognitions, key=lambda item: item.fingerprint
                            )
                        ],
                    },
                    dependencies=tuple(
                        sorted(item.fingerprint for item in recognitions)
                    ),
                )
        return evaluation.result

    async def _align(
        self,
        audio: Path,
        *,
        expected_tokens: tuple[str, ...],
        consensus: ConsensusResult,
        language: str,
        span: AudioSpan,
        hits: list[str],
        misses: list[str],
    ) -> ForcedAlignmentResult:
        lexical_hash = text_fingerprint("\u001f".join(expected_tokens))
        verified = VerifiedTextSpan(
            consensus.fingerprint,
            0,
            len(expected_tokens),
            lexical_hash,
        )
        key = ForcedAlignmentCacheIdentity(
            model_fingerprint=self._forced_aligner.fingerprint,
            execution_fingerprint=self._forced_aligner.fingerprint,
            canonical_audio_fingerprint=span.audio_digest,
            span=span,
            language=language,
            aligner_text_hash=lexical_hash,
            expected_lexical_span_hash=lexical_hash,
            purpose=AlignmentPurpose.VERIFIED_TARGET,
            alignment_settings_fingerprint=consensus.fingerprint,
        ).fingerprint
        cached = self._alignment_cache.get(key)
        if cached is not None:
            hits.append("forced_alignment")
            return cached

        async def produce() -> dict[str, object]:
            value = await self._forced_aligner.force_align(
                audio,
                " ".join(expected_tokens),
                language=language,
                purpose=AlignmentPurpose.VERIFIED_TARGET,
                verified_span=verified,
                span=span,
            )
            _validate_alignment_result(value, span=span, lexical_hash=lexical_hash)
            return forced_alignment_report(value)

        if self._evidence_cache is not None:
            entry, lookup = await self._evidence_cache.get_or_compute(
                stage="forced-alignment",
                key=key,
                dependencies=(consensus.fingerprint,),
                producer=produce,
                validator=forced_alignment_from_report,
            )
            result = forced_alignment_from_report(entry.evidence)
            (hits if lookup.hit else misses).append("forced_alignment")
            self._alignment_cache[key] = result
            return result
        misses.append("forced_alignment")
        result = forced_alignment_from_report(await produce())
        self._alignment_cache[key] = result
        return result

    def _verification(
        self,
        *,
        consensus: ConsensusResult,
        alignment: ForcedAlignmentResult | None,
        artifact_digest: str,
        scope: VerificationScope,
        accepted: bool,
        reason_codes: tuple[str, ...],
        signal_evidence_fingerprint: str | None,
        hits: list[str],
        misses: list[str],
    ) -> SpeechVerification:
        terminal_policy = semantic_fingerprint(
            "terminal-verification-policy-v1",
            {
                "policy": self._policy.fingerprint,
                "scope": scope,
                "accepted": accepted,
                "reason_codes": reason_codes,
            },
        )
        key = VerificationCacheIdentity(
            consensus_fingerprint=consensus.fingerprint,
            forced_alignment_fingerprint=(alignment.fingerprint if alignment else None),
            signal_evidence_fingerprint=signal_evidence_fingerprint,
            artifact_digest=artifact_digest,
            policy_fingerprint=terminal_policy,
            human_disposition_fingerprint=None,
            calibration_fingerprint=self._calibration.fingerprint,
        ).fingerprint
        cached = self._verification_cache.get(key)
        if cached is not None:
            hits.append("verification")
            return cached
        if self._evidence_cache is not None:
            entry, _lookup = self._evidence_cache.lookup("verification", key)
            if entry is not None:
                verification = verification_from_report(entry.evidence)
                self._verification_cache[key] = verification
                hits.append("verification")
                return verification
        misses.append("verification")
        verification = SpeechVerification(
            self._policy.fingerprint,
            consensus.fingerprint,
            alignment.fingerprint if alignment is not None else None,
            signal_evidence_fingerprint,
            artifact_digest,
            scope,
            accepted,
            reason_codes,
        )
        self._verification_cache[key] = verification
        if self._evidence_cache is not None:
            self._evidence_cache.store(
                stage="verification",
                key=key,
                evidence=verification_report(verification),
                dependencies=tuple(
                    item
                    for item in (
                        consensus.fingerprint,
                        alignment.fingerprint if alignment is not None else None,
                        signal_evidence_fingerprint,
                    )
                    if item is not None
                ),
                validator=verification_from_report,
            )
        return verification


def _validate_recognition_result(
    result: RecognitionResult, *, engine: str, span: AudioSpan
) -> None:
    if result.engine != engine or result.span != span:
        raise SpeechAnalysisError("Recognizer returned evidence for the wrong request")


def _validate_alignment_result(
    result: ForcedAlignmentResult, *, span: AudioSpan, lexical_hash: str
) -> None:
    if result.span != span or result.purpose is not AlignmentPurpose.VERIFIED_TARGET:
        raise SpeechAnalysisError(
            "Forced aligner returned evidence for the wrong request"
        )
    if result.expected_lexical_span_hash != lexical_hash:
        raise SpeechAnalysisError(
            "Forced aligner returned timing for different lexical text"
        )


def boundary_agreement(
    observations: tuple[BoundaryObservation, ...],
    *,
    tolerance_frames: int,
) -> BoundaryAgreement:
    """Validate a safe intersection without averaging engine timestamps."""
    if len(observations) < _MINIMUM_BOUNDARY_OBSERVATIONS or tolerance_frames < 0:
        raise ValidationError(
            "Boundary agreement requires three sources and a tolerance"
        )
    if len({item.source for item in observations}) != len(observations):
        raise ValidationError("Boundary observation sources must be unique")
    starts = tuple(item.start_frame for item in observations)
    ends = tuple(item.end_frame for item in observations)
    start_envelope = (min(starts), max(starts))
    end_envelope = (min(ends), max(ends))
    if (
        start_envelope[1] - start_envelope[0] > tolerance_frames
        or end_envelope[1] - end_envelope[0] > tolerance_frames
    ):
        raise SpeechAnalysisError("Target boundary disagreement exceeds tolerance")
    safe_start = start_envelope[0]
    safe_end = end_envelope[1]
    if safe_end <= safe_start:
        raise SpeechAnalysisError("Target boundaries have no safe crop intersection")
    return BoundaryAgreement(
        tuple(sorted(observations, key=lambda item: item.source)),
        start_envelope,
        end_envelope,
        tolerance_frames,
        safe_start,
        safe_end,
    )


def _target_boundary_observations(
    analysis: EnsembleAnalysis,
    *,
    target_token_start: int,
    target_token_end: int,
) -> tuple[BoundaryObservation, ...]:
    alignment = analysis.forced_alignment
    if alignment is None or target_token_end > len(alignment.units):
        raise SpeechAnalysisError("Forced target projection is incomplete")
    units = alignment.units[target_token_start:target_token_end]
    observations = [
        BoundaryObservation("qwen-forced", units[0].start_frame, units[-1].end_frame)
    ]
    for result in analysis.recognitions:
        if len(result.tokens) != len(alignment.units):
            continue
        selected = result.tokens[target_token_start:target_token_end]
        if (
            not selected
            or selected[0].start_frame is None
            or selected[-1].end_frame is None
        ):
            continue
        observations.append(
            BoundaryObservation(
                result.engine,
                selected[0].start_frame,
                selected[-1].end_frame,
            )
        )
    required = {"qwen-forced", "whisper", "qwen"}
    if not required <= {item.source for item in observations}:
        raise SpeechAnalysisError("Target lacks independent forced and ASR timing")
    return tuple(observations)


def _whole_span(expected_tokens: tuple[str, ...]) -> LexicalSpan:
    return LexicalSpan(0, len(expected_tokens))


def _wav_span(path: Path) -> AudioSpan:
    try:
        with wave.open(str(path), "rb") as reader:
            if reader.getnchannels() != 1:
                raise ValidationError("Speech analysis requires mono WAV input")
            sample_rate = reader.getframerate()
            frames = reader.getnframes()
    except (OSError, EOFError, wave.Error) as error:
        raise ValidationError(f"Cannot inspect analysis WAV {path}: {error}") from error
    return AudioSpan(sha256_file(path), 0, frames, sample_rate)


__all__ = [
    "AnalysisCacheTrace",
    "AnalysisGateError",
    "BoundaryAgreement",
    "BoundaryObservation",
    "CarrierExtractionEvidence",
    "EnsembleAnalysis",
    "SpeechAnalysisEnsemble",
    "boundary_agreement",
]
