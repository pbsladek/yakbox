"""Versioned phoneme forced-alignment inspection service."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from yakbox.contracts import runtime_metadata
from yakbox.local_phoneme_alignment import Wav2Vec2CtcPhonemeAligner
from yakbox.speech.phonemes import (
    PhonemeAlignmentDecision,
    PhonemeAlignmentResult,
    validate_phoneme_alignment,
)


@dataclass(frozen=True, slots=True)
class PhonemeInspection:
    """Privacy-safe forced-alignment evidence for one expected utterance."""

    audio: Path
    expected_sha256: str
    result: PhonemeAlignmentResult
    decision: PhonemeAlignmentDecision

    @property
    def accepted(self) -> bool:
        """Return whether the path and final phoneme boundary passed."""
        return self.decision.accepted

    def to_dict(self) -> dict[str, object]:
        """Serialize the versioned phoneme inspection contract."""
        return {
            **runtime_metadata("phoneme-inspection"),
            "audio": str(self.audio),
            "expected_sha256": self.expected_sha256,
            "accepted": self.accepted,
            "reason_codes": list(self.decision.reason_codes),
            "minimum_confidence": self.decision.confidence,
            "start_seconds": self.decision.start_seconds,
            "end_seconds": self.decision.end_seconds,
            "alignment": {
                "backend": self.result.backend,
                "model": self.result.model,
                "fingerprint": self.result.fingerprint,
                "language": self.result.language,
                "issues": list(self.result.issues),
                "phonemes": [asdict(item) for item in self.result.phonemes],
            },
        }


async def inspect_phonemes(
    audio: Path,
    expected_text: str,
    *,
    language: str,
    model: str,
    revision: str | None,
    minimum_confidence: float = 0.20,
    aligner: Wav2Vec2CtcPhonemeAligner | None = None,
) -> PhonemeInspection:
    """Force expected phonemes onto audio and apply confidence/timing gates."""
    resolved = aligner or Wav2Vec2CtcPhonemeAligner(
        model=model,
        revision=revision,
    )
    result = await resolved.align(audio, expected_text, language=language)
    return PhonemeInspection(
        audio=audio.resolve(),
        expected_sha256=hashlib.sha256(expected_text.encode()).hexdigest(),
        result=result,
        decision=validate_phoneme_alignment(
            result,
            minimum_confidence=minimum_confidence,
        ),
    )
