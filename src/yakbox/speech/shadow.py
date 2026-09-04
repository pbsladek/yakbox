"""Read-only comparison between current Whisper QA and proposed consensus."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from yakbox.contracts import runtime_metadata
from yakbox.speech.analysis_models import ConsensusOutcome, ConsensusResult


class ShadowClassification(StrEnum):
    AGREEMENT = "agreement"
    ADDITIONAL_DEFECT = "additional_defect"
    FALSE_REJECTION_CANDIDATE = "false_rejection_candidate"
    FALSE_ACCEPTANCE_CANDIDATE = "false_acceptance_candidate"
    BASELINE_REJECTION_REMOVED = "baseline_rejection_removed"


class ShadowGroundTruth(StrEnum):
    KNOWN_CLEAN = "known_clean"
    KNOWN_DEFECTIVE = "known_defective"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ShadowComparison:
    """One non-gating comparison with stable evidence references."""

    audio_digest: str
    policy_fingerprint: str
    baseline_accepted: bool
    baseline_reason_codes: tuple[str, ...]
    consensus: ConsensusResult
    ground_truth: ShadowGroundTruth
    classification: ShadowClassification

    def to_dict(self) -> dict[str, object]:
        return {
            **runtime_metadata("speech-analysis-shadow"),
            "audio_digest": self.audio_digest,
            "policy_fingerprint": self.policy_fingerprint,
            "baseline": {
                "accepted": self.baseline_accepted,
                "reason_codes": list(self.baseline_reason_codes),
            },
            "proposed": {
                "accepted": self.consensus.outcome is ConsensusOutcome.ACCEPTED,
                "consensus_fingerprint": self.consensus.fingerprint,
                "reason_codes": list(self.consensus.reason_codes),
            },
            "ground_truth": self.ground_truth.value,
            "classification": self.classification.value,
        }


def compare_shadow_decision(
    *,
    audio_digest: str,
    baseline_accepted: bool,
    baseline_reason_codes: tuple[str, ...],
    consensus: ConsensusResult,
    ground_truth: ShadowGroundTruth = ShadowGroundTruth.UNKNOWN,
) -> ShadowComparison:
    """Classify an ensemble decision without changing the current build outcome."""
    proposed_accepted = consensus.outcome is ConsensusOutcome.ACCEPTED
    if baseline_accepted == proposed_accepted:
        classification = ShadowClassification.AGREEMENT
    elif baseline_accepted:
        classification = (
            ShadowClassification.ADDITIONAL_DEFECT
            if ground_truth is ShadowGroundTruth.KNOWN_DEFECTIVE
            else ShadowClassification.FALSE_REJECTION_CANDIDATE
        )
    else:
        classification = (
            ShadowClassification.FALSE_ACCEPTANCE_CANDIDATE
            if ground_truth is ShadowGroundTruth.KNOWN_DEFECTIVE
            else ShadowClassification.BASELINE_REJECTION_REMOVED
        )
    return ShadowComparison(
        audio_digest=audio_digest,
        policy_fingerprint=consensus.policy_fingerprint,
        baseline_accepted=baseline_accepted,
        baseline_reason_codes=baseline_reason_codes,
        consensus=consensus,
        ground_truth=ground_truth,
        classification=classification,
    )


__all__ = [
    "ShadowClassification",
    "ShadowComparison",
    "ShadowGroundTruth",
    "compare_shadow_decision",
]
