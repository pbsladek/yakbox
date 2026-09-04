"""Immutable multi-repair planning and one-pass candidate reconstruction."""

from __future__ import annotations

import asyncio
import re
import wave
from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path

from yakbox._files import sha256_file
from yakbox.audio.splice import (
    AdaptiveSpliceEvidence,
    FrameReplacement,
    splice_wav_regions,
)
from yakbox.audiobook.repair_cache import RepairCacheEvent, RepairStageCache
from yakbox.contracts import runtime_metadata
from yakbox.errors import ArtifactError, SpeechAnalysisError, ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint
from yakbox.speech.analysis_models import (
    ClipClass,
    SpeechVerification,
    VerificationScope,
)
from yakbox.speech.analysis_pipeline import (
    CarrierExtractionEvidence,
    EnsembleAnalysis,
    SpeechAnalysisEnsemble,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def repair_candidate_output_path(workspace: Path, batch_fingerprint: str) -> Path:
    """Return the managed candidate path, distinct from every release path."""
    if _SHA256.fullmatch(batch_fingerprint) is None:
        raise ValidationError("Repair candidate batch fingerprint must be SHA-256")
    return (
        workspace.resolve()
        / ".yakbox"
        / "repair-candidates"
        / batch_fingerprint
        / "candidate.wav"
    )


class AnalyzedArtifactState(StrEnum):
    """States that must remain distinct in paths, reports, and release checks."""

    REPAIR_CANDIDATE = "repair_candidate"
    RELEASE_VERIFIED = "release_verified"


@dataclass(frozen=True, slots=True)
class ApprovedReplacementTake:
    """One independently accepted take bound to an immutable base interval."""

    repair_id: str
    take: int
    chunk_id: str
    base_audio_digest: str
    start_frame: int
    end_frame: int
    replacement_audio: Path
    replacement_digest: str
    recognition_engines: tuple[str, ...]
    generation_fingerprint: str
    recognition_fingerprint: str
    forced_alignment_fingerprint: str
    extraction_fingerprint: str
    dsp_fingerprint: str
    approval_fingerprint: str
    verification: SpeechVerification

    def __post_init__(self) -> None:
        if not self.repair_id or not self.chunk_id or self.take < 1:
            raise ValidationError("Approved replacement identity is incomplete")
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValidationError("Approved replacement interval is invalid")
        for value in (
            self.base_audio_digest,
            self.replacement_digest,
            self.generation_fingerprint,
            self.recognition_fingerprint,
            self.forced_alignment_fingerprint,
            self.extraction_fingerprint,
            self.dsp_fingerprint,
            self.approval_fingerprint,
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValidationError(
                    "Approved replacement fingerprints must be SHA-256"
                )
        if len(set(self.recognition_engines)) != len(self.recognition_engines):
            raise ValidationError(
                "Approved replacement recognizer names must be unique"
            )
        if not {"whisper", "qwen"} <= set(self.recognition_engines):
            raise ValidationError(
                "Approved replacement requires Whisper and Qwen evidence"
            )
        if (
            not self.verification.accepted
            or self.verification.scope is not VerificationScope.CANDIDATE
            or self.verification.forced_alignment_fingerprint
            != self.forced_alignment_fingerprint
        ):
            raise ValidationError("Approved replacement lacks candidate verification")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "approved-replacement-take-v1",
            {
                "repair_id": self.repair_id,
                "take": self.take,
                "chunk_id": self.chunk_id,
                "base_audio_digest": self.base_audio_digest,
                "interval": (self.start_frame, self.end_frame),
                "replacement_digest": self.replacement_digest,
                "recognition_engines": tuple(sorted(self.recognition_engines)),
                "generation": self.generation_fingerprint,
                "recognition": self.recognition_fingerprint,
                "forced_alignment": self.forced_alignment_fingerprint,
                "extraction": self.extraction_fingerprint,
                "dsp": self.dsp_fingerprint,
                "approval": self.approval_fingerprint,
                "verification": self.verification.fingerprint,
            },
        )


@dataclass(frozen=True, slots=True)
class RepairQaWindow:
    """Coalesced affected window retaining every contributing repair ID."""

    start_frame: int
    end_frame: int
    repair_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RepairBatchPlan:
    """Canonical timeline plan independent of caller input ordering."""

    base_audio_digest: str
    sample_rate: int
    replacements: tuple[ApprovedReplacementTake, ...]
    affected_chunk_ids: tuple[str, ...]
    affected_join_ids: tuple[str, ...]
    qa_windows: tuple[RepairQaWindow, ...]
    splice_policy_fingerprint: str
    state: AnalyzedArtifactState = AnalyzedArtifactState.REPAIR_CANDIDATE

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint(
            "repair-batch-plan-v2",
            {
                "base_audio_digest": self.base_audio_digest,
                "sample_rate": self.sample_rate,
                "replacements": tuple(item.fingerprint for item in self.replacements),
                "affected_chunk_ids": self.affected_chunk_ids,
                "affected_join_ids": self.affected_join_ids,
                "qa_windows": self.qa_windows,
                "splice_policy": self.splice_policy_fingerprint,
                "state": self.state,
            },
        )


@dataclass(frozen=True, slots=True)
class RepairBatchCandidate:
    """One reconstructed raw candidate, never a release-verification claim."""

    plan: RepairBatchPlan
    output: Path
    output_digest: str
    splice_evidence: tuple[AdaptiveSpliceEvidence, ...]
    cache_event: RepairCacheEvent | None
    state: AnalyzedArtifactState = AnalyzedArtifactState.REPAIR_CANDIDATE

    @property
    def cache_miss_stage(self) -> str | None:
        if self.cache_event is None or self.cache_event.hit:
            return None
        return self.cache_event.stage


@dataclass(frozen=True, slots=True)
class RepairJoinEvidence:
    """One affected contextual join result bound by stable join ID."""

    join_id: str
    verification_fingerprint: str
    accepted: bool

    def __post_init__(self) -> None:
        if not self.join_id or _SHA256.fullmatch(self.verification_fingerprint) is None:
            raise ValidationError("Repair join evidence identity is incomplete")


@dataclass(frozen=True, slots=True)
class ChangedChunkEvidence:
    """Independent verification of one reconstructed affected raw chunk."""

    chunk_id: str
    verification: SpeechVerification

    def __post_init__(self) -> None:
        if (
            not self.chunk_id
            or not self.verification.accepted
            or self.verification.scope is not VerificationScope.CANDIDATE
        ):
            raise ValidationError("Changed-chunk repair evidence is invalid")


@dataclass(frozen=True, slots=True)
class PostMasterWindowEvidence:
    """Verified outward-mapped mastered window with repair provenance."""

    raw_start_frame: int
    raw_end_frame: int
    start_frame: int
    end_frame: int
    repair_ids: tuple[str, ...]
    verification: SpeechVerification

    def __post_init__(self) -> None:
        if (
            self.start_frame < 0
            or self.end_frame <= self.start_frame
            or self.raw_start_frame < 0
            or self.raw_end_frame <= self.raw_start_frame
            or not self.repair_ids
            or not self.verification.accepted
            or self.verification.scope is not VerificationScope.CANDIDATE
        ):
            raise ValidationError("Post-master repair-window evidence is invalid")


@dataclass(frozen=True, slots=True)
class RepairCandidateQualification:
    """Affected-scope proof that remains explicitly short of release status."""

    candidate_digest: str
    changed_chunks: tuple[ChangedChunkEvidence, ...]
    joins: tuple[RepairJoinEvidence, ...]
    post_master_windows: tuple[PostMasterWindowEvidence, ...]
    technical_fingerprint: str
    accepted: bool
    state: AnalyzedArtifactState = AnalyzedArtifactState.REPAIR_CANDIDATE

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("repair-candidate-qualification-v1", self)


def qualify_repair_candidate(
    candidate: RepairBatchCandidate,
    *,
    changed_chunks: tuple[ChangedChunkEvidence, ...],
    joins: tuple[RepairJoinEvidence, ...],
    post_master_windows: tuple[PostMasterWindowEvidence, ...],
    technical_fingerprint: str,
    technical_accepted: bool,
) -> RepairCandidateQualification:
    """Require changed chunks, both adjacent joins, mapped windows, and technical QA."""
    expected_joins = set(candidate.plan.affected_join_ids)
    actual_joins = {item.join_id for item in joins}
    if actual_joins != expected_joins or len(actual_joins) != len(joins):
        raise ValidationError("Repair candidate join evidence is incomplete")
    expected_chunks = set(candidate.plan.affected_chunk_ids)
    actual_chunks = {item.chunk_id for item in changed_chunks}
    if actual_chunks != expected_chunks or len(actual_chunks) != len(changed_chunks):
        raise ValidationError("Repair candidate changed-chunk evidence is incomplete")
    expected_windows = {
        (item.start_frame, item.end_frame, item.repair_ids)
        for item in candidate.plan.qa_windows
    }
    actual_windows = {
        (item.raw_start_frame, item.raw_end_frame, item.repair_ids)
        for item in post_master_windows
    }
    if expected_windows != actual_windows:
        raise ValidationError("Repair candidate post-master evidence is incomplete")
    if _SHA256.fullmatch(technical_fingerprint) is None:
        raise ValidationError("Repair candidate technical fingerprint must be SHA-256")
    ordered_chunks = tuple(sorted(changed_chunks, key=lambda item: item.chunk_id))
    accepted = (
        technical_accepted
        and all(item.accepted for item in joins)
        and all(item.verification.accepted for item in post_master_windows)
        and all(item.verification.accepted for item in changed_chunks)
    )
    return RepairCandidateQualification(
        candidate.output_digest,
        ordered_chunks,
        tuple(sorted(joins, key=lambda item: item.join_id)),
        tuple(
            sorted(
                post_master_windows, key=lambda item: (item.start_frame, item.end_frame)
            )
        ),
        technical_fingerprint,
        accepted,
    )


def repair_candidate_report(
    value: RepairCandidateQualification,
) -> dict[str, object]:
    """Serialize candidate-only evidence with a state a release cannot accept."""
    return {
        **runtime_metadata("speech-repair-candidate-verification"),
        "fingerprint": value.fingerprint,
        "candidate_digest": value.candidate_digest,
        "state": value.state.value,
        "accepted": value.accepted,
        "changed_chunks": [
            {
                "chunk_id": item.chunk_id,
                "verification_fingerprint": item.verification.fingerprint,
            }
            for item in value.changed_chunks
        ],
        "joins": [asdict(item) for item in value.joins],
        "post_master_windows": [
            {
                "raw_start_frame": item.raw_start_frame,
                "raw_end_frame": item.raw_end_frame,
                "start_frame": item.start_frame,
                "end_frame": item.end_frame,
                "repair_ids": list(item.repair_ids),
                "verification_fingerprint": item.verification.fingerprint,
            }
            for item in value.post_master_windows
        ],
        "technical_fingerprint": value.technical_fingerprint,
    }


@dataclass(frozen=True, slots=True)
class RepairAnchorInput:
    """One independently synthesized/cropped anchor with original-frame offset."""

    audio: Path
    expected_tokens: tuple[str, ...]
    original_frame_offset: int

    def __post_init__(self) -> None:
        if not self.expected_tokens or self.original_frame_offset < 0:
            raise ValidationError("Repair anchor input is incomplete")


@dataclass(frozen=True, slots=True)
class SentenceRepairAlignment:
    """Verified original anchor bracket plus verified generated target timing."""

    before_anchor: EnsembleAnalysis
    after_anchor: EnsembleAnalysis
    generated_target_evidence_fingerprint: str
    original_start_frame: int
    original_end_frame: int

    def __post_init__(self) -> None:
        if (
            self.original_start_frame < 0
            or self.original_end_frame <= self.original_start_frame
        ):
            raise ValidationError("Sentence repair anchor bracket is invalid")
        if _SHA256.fullmatch(self.generated_target_evidence_fingerprint) is None:
            raise ValidationError(
                "Generated target evidence fingerprint must be SHA-256"
            )

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("sentence-repair-alignment-v1", self)


async def bracket_sentence_repair(
    ensemble: SpeechAnalysisEnsemble,
    *,
    before: RepairAnchorInput,
    after: RepairAnchorInput,
    generated: CarrierExtractionEvidence,
    language: str = "en",
) -> SentenceRepairAlignment:
    """Use accepted original anchors and independently verified generated timing."""
    before_analysis, after_analysis = await _analyze_repair_anchors(
        ensemble,
        before=before,
        after=after,
        language=language,
    )
    before_alignment = before_analysis.forced_alignment
    after_alignment = after_analysis.forced_alignment
    if before_alignment is None or after_alignment is None:
        raise SpeechAnalysisError("Repair anchors lack authorized forced timing")
    if not generated.extracted.verification.accepted:
        raise SpeechAnalysisError(
            "Generated replacement did not pass crop verification"
        )
    start = before.original_frame_offset + before_alignment.units[-1].end_frame
    end = after.original_frame_offset + after_alignment.units[0].start_frame
    return SentenceRepairAlignment(
        before_analysis,
        after_analysis,
        generated.fingerprint,
        start,
        end,
    )


async def _analyze_repair_anchors(
    ensemble: SpeechAnalysisEnsemble,
    *,
    before: RepairAnchorInput,
    after: RepairAnchorInput,
    language: str,
) -> tuple[EnsembleAnalysis, EnsembleAnalysis]:
    analyses = await asyncio.gather(
        ensemble.analyze(
            before.audio,
            expected_tokens=before.expected_tokens,
            clip_class=ClipClass.REPAIRED_REGION,
            scope=VerificationScope.CANDIDATE,
            language=language,
            high_risk=True,
        ),
        ensemble.analyze(
            after.audio,
            expected_tokens=after.expected_tokens,
            clip_class=ClipClass.REPAIRED_REGION,
            scope=VerificationScope.CANDIDATE,
            language=language,
            high_risk=True,
        ),
    )
    if any(not analysis.verification.accepted for analysis in analyses):
        raise SpeechAnalysisError(
            "Original repair anchors failed independent consensus"
        )
    return analyses


def plan_repair_batch(
    replacements: tuple[ApprovedReplacementTake, ...],
    *,
    sample_rate: int,
    qa_margin_frames: int,
    splice_policy_fingerprint: str,
) -> RepairBatchPlan:
    """Resolve overlap, order, joins, and unioned QA windows before rebuilding."""
    if not replacements or sample_rate <= 0 or qa_margin_frames < 0:
        raise ValidationError("Repair batch inputs are incomplete")
    base_digests = {item.base_audio_digest for item in replacements}
    if len(base_digests) != 1:
        raise ValidationError("Repair takes do not share one exact base audio digest")
    if _SHA256.fullmatch(splice_policy_fingerprint) is None:
        raise ValidationError("Splice policy fingerprint must be SHA-256")
    ordered = tuple(
        sorted(
            replacements,
            key=lambda item: (
                item.start_frame,
                item.end_frame,
                item.chunk_id,
                item.repair_id,
            ),
        )
    )
    identities = tuple((item.repair_id, item.take) for item in ordered)
    if len(set(identities)) != len(identities):
        raise ValidationError("Repair batch contains a duplicate approved take")
    for previous, current in pairwise(ordered):
        if current.start_frame < previous.end_frame:
            raise ValidationError("Repair batch frame intervals overlap")
    windows = _coalesced_windows(ordered, margin=qa_margin_frames)
    chunks = tuple(dict.fromkeys(item.chunk_id for item in ordered))
    joins = tuple(
        sorted(
            {
                join
                for item in ordered
                for join in (f"{item.chunk_id}:before", f"{item.chunk_id}:after")
            }
        )
    )
    return RepairBatchPlan(
        next(iter(base_digests)),
        sample_rate,
        ordered,
        chunks,
        joins,
        windows,
        splice_policy_fingerprint,
    )


def assemble_repair_batch(
    plan: RepairBatchPlan,
    *,
    base_audio: Path,
    destination: Path,
    cache: RepairStageCache | None = None,
) -> RepairBatchCandidate:
    """Revalidate inputs and reconstruct one candidate from base coordinates once."""
    if plan.state is not AnalyzedArtifactState.REPAIR_CANDIDATE:
        raise ValidationError("Only a repair-candidate plan may be assembled")
    if sha256_file(base_audio) != plan.base_audio_digest:
        raise ArtifactError("Repair batch base audio digest changed")
    _validate_sample_rate(base_audio, plan.sample_rate)
    for item in plan.replacements:
        if sha256_file(item.replacement_audio) != item.replacement_digest:
            raise ArtifactError(f"Approved replacement changed: {item.repair_id}")
    event: RepairCacheEvent | None = None
    if cache is not None:
        key = cache.key("reconstruction", {"batch_plan": plan.fingerprint})
        event = cache.restore_audio(
            chunk_id="batch",
            stage="reconstruction",
            key=key,
            destination=destination,
        )
        if event.hit:
            return RepairBatchCandidate(
                plan,
                destination,
                sha256_file(destination),
                (),
                event,
            )
    evidence = splice_wav_regions(
        base_audio,
        tuple(
            FrameReplacement(
                item.repair_id,
                item.start_frame,
                item.end_frame,
                item.replacement_audio,
            )
            for item in plan.replacements
        ),
        destination,
        overwrite=True,
    )
    if cache is not None:
        cache.store_audio(stage="reconstruction", key=key, source=destination)
    return RepairBatchCandidate(
        plan,
        destination,
        sha256_file(destination),
        evidence,
        event,
    )


def _coalesced_windows(
    replacements: tuple[ApprovedReplacementTake, ...],
    *,
    margin: int,
) -> tuple[RepairQaWindow, ...]:
    windows: list[RepairQaWindow] = []
    for item in replacements:
        start = max(0, item.start_frame - margin)
        end = item.end_frame + margin
        if windows and start <= windows[-1].end_frame:
            previous = windows[-1]
            windows[-1] = RepairQaWindow(
                previous.start_frame,
                max(previous.end_frame, end),
                tuple(sorted({*previous.repair_ids, item.repair_id})),
            )
        else:
            windows.append(RepairQaWindow(start, end, (item.repair_id,)))
    return tuple(windows)


def _validate_sample_rate(path: Path, expected: int) -> None:
    try:
        with wave.open(str(path), "rb") as reader:
            actual = reader.getframerate()
    except (OSError, EOFError, wave.Error) as error:
        raise ArtifactError(f"Cannot inspect repair WAV {path}: {error}") from error
    if actual != expected:
        raise ArtifactError("Repair batch sample rate differs from its plan")


__all__ = [
    "AnalyzedArtifactState",
    "ApprovedReplacementTake",
    "ChangedChunkEvidence",
    "PostMasterWindowEvidence",
    "RepairAnchorInput",
    "RepairBatchCandidate",
    "RepairBatchPlan",
    "RepairCandidateQualification",
    "RepairJoinEvidence",
    "RepairQaWindow",
    "SentenceRepairAlignment",
    "assemble_repair_batch",
    "bracket_sentence_repair",
    "plan_repair_batch",
    "qualify_repair_candidate",
    "repair_candidate_output_path",
    "repair_candidate_report",
]
