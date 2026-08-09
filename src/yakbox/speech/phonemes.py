"""Backend-neutral phoneme forced-alignment contracts and gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

_IPA_VOWELS = frozenset(
    "aeiouyɐ"
    "\N{LATIN SMALL LETTER ALPHA}"
    "ɒɔɘəɚɛɜɝɞɟɨ"
    "\N{LATIN LETTER SMALL CAPITAL I}"
    "\N{LATIN SMALL LETTER TURNED M}"
    "ɵʉʊʌ"
    "\N{LATIN LETTER SMALL CAPITAL Y}"
    "æœɶ"
)


@dataclass(frozen=True, slots=True)
class PhonemeToken:
    """One expected phoneme with forced audio timing and path confidence."""

    symbol: str
    start_seconds: float
    end_seconds: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PhonemeAlignmentResult:
    """Expected phonemes forced onto an acoustic model's CTC emissions."""

    phonemes: tuple[PhonemeToken, ...]
    backend: str
    model: str
    fingerprint: str
    language: str
    issues: tuple[str, ...] = ()


@runtime_checkable
class PhonemeAligner(Protocol):
    """Optional local forced aligner used to validate short audio edges."""

    @property
    def fingerprint(self) -> str:
        """Return a stable backend, model, and phonemizer fingerprint."""
        ...

    async def align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
    ) -> PhonemeAlignmentResult:
        """Force the expected pronunciation onto an audio file."""
        ...


@dataclass(frozen=True, slots=True)
class PhonemeAlignmentDecision:
    """Hard-gate decision over phoneme timing and confidence evidence."""

    accepted: bool
    reason_codes: tuple[str, ...]
    start_seconds: float | None = None
    end_seconds: float | None = None
    confidence: float | None = None


def validate_phoneme_alignment(
    result: PhonemeAlignmentResult,
    *,
    minimum_confidence: float,
) -> PhonemeAlignmentDecision:
    """Reject invalid paths or weak final phoneme-boundary evidence."""
    reasons = list(result.issues)
    if not result.phonemes:
        reasons.append("phoneme_alignment_missing")
        return PhonemeAlignmentDecision(False, tuple(dict.fromkeys(reasons)))
    previous_end = 0.0
    for item in result.phonemes:
        if (
            not item.symbol
            or not all(
                math.isfinite(value)
                for value in (item.start_seconds, item.end_seconds, item.confidence)
            )
            or item.start_seconds < previous_end
            or item.end_seconds <= item.start_seconds
            or not 0 <= item.confidence <= 1
        ):
            reasons.append("invalid_phoneme_timing")
        previous_end = max(previous_end, item.end_seconds)
    confidence = _final_boundary_confidence(result.phonemes)
    if final_phoneme_is_consonant(result) and confidence < minimum_confidence:
        reasons.append("low_phoneme_confidence")
    return PhonemeAlignmentDecision(
        not reasons,
        tuple(dict.fromkeys(reasons)),
        start_seconds=result.phonemes[0].start_seconds,
        end_seconds=result.phonemes[-1].end_seconds,
        confidence=confidence,
    )


def _final_boundary_confidence(phonemes: tuple[PhonemeToken, ...]) -> float:
    final = phonemes[-1]
    if not _is_consonant(final.symbol):
        return final.confidence
    cluster = [final.confidence]
    for item in reversed(phonemes[:-1]):
        if not _is_consonant(item.symbol):
            break
        cluster.append(item.confidence)
    return sum(cluster) / len(cluster)


def final_phoneme_is_consonant(result: PhonemeAlignmentResult) -> bool:
    """Return whether the expected utterance ends on a consonant phone."""
    return bool(result.phonemes and _is_consonant(result.phonemes[-1].symbol))


def _is_consonant(symbol: str) -> bool:
    return not any(character.casefold() in _IPA_VOWELS for character in symbol)
