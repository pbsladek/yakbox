"""Human- and machine-readable local Whisper inspection reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

from yakbox.audio.crop import (
    inspect_signal_quality,
    inspect_speech_islands,
    wav_duration_seconds,
)
from yakbox.contracts import runtime_metadata
from yakbox.errors import ValidationError
from yakbox.local_alignment import MlxWhisperAligner
from yakbox.speech.alignment import (
    AlignmentResult,
    lexical_tokens,
    validate_extracted_alignment,
)
from yakbox.whisper_qa import (
    AlignmentEvaluation,
    WhisperClipType,
    alignment_evidence,
    classify_clip_type,
    evaluate_alignment,
)


@dataclass(frozen=True, slots=True)
class WhisperInspection:
    """Versioned transcription, timing, quality, and acoustic evidence."""

    path: Path
    expected_text: str | None
    result: AlignmentResult
    duration_seconds: float
    token_diff: tuple[dict[str, object], ...]
    reason_codes: tuple[str, ...]
    confidence: AlignmentEvaluation

    @property
    def accepted(self) -> bool:
        """Return whether structural quality and optional exact text pass."""
        return not self.reason_codes

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete explicit inspection result."""
        acoustic = inspect_speech_islands(self.path)
        signal = inspect_signal_quality(self.path)
        first = self.result.tokens[0].start_seconds if self.result.tokens else None
        last = self.result.tokens[-1].end_seconds if self.result.tokens else None
        evidence_start = self.result.clip_start_seconds or 0.0
        evidence_end = self.result.clip_end_seconds or self.duration_seconds
        return {
            **runtime_metadata("whisper-inspection"),
            "path": str(self.path),
            "duration_seconds": self.duration_seconds,
            "expected_text": self.expected_text,
            "recognized_text": self.result.transcript,
            "recognized_tokens": [token.text for token in self.result.tokens],
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "token_diff": list(self.token_diff),
            "backend": self.result.backend,
            "model": self.result.model,
            "fingerprint": self.result.fingerprint,
            "language": self.result.language,
            "timing_source": self.result.timing_source,
            "clip_type": self.confidence.clip_type.value,
            "confidence_profile": asdict(self.confidence.profile),
            "minimum_word_confidence": self.confidence.minimum_word_confidence,
            "decode_evidence": alignment_evidence(self.result),
            "words": [asdict(token) for token in self.result.tokens],
            "segments": [asdict(segment) for segment in self.result.segments],
            "parser_issues": list(self.result.issues),
            "speech_regions": [asdict(region) for region in self.result.speech_regions],
            "speech_islands": asdict(acoustic),
            "signal_quality": asdict(signal),
            "edge_speech": {
                "before_first_word_ms": (
                    max(0.0, first - evidence_start) * 1_000
                    if first is not None
                    else None
                ),
                "after_last_word_ms": (
                    max(0.0, evidence_end - last) * 1_000 if last is not None else None
                ),
            },
        }


async def inspect_with_whisper(
    path: Path,
    *,
    expected_text: str | None,
    language: str,
    model: str,
    revision: str | None,
    clip_type: WhisperClipType | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> WhisperInspection:
    """Run strict local alignment and explain every resulting gate."""
    aligner = MlxWhisperAligner(
        model=model,
        revision=revision,
        decode_consensus=True,
        prompt_sensitivity=True,
    )
    expected = expected_text or ""
    if (start_seconds is None) != (end_seconds is None):
        raise ValidationError("Both --start and --end are required for a window")
    result = (
        await aligner.align_window(
            path,
            expected,
            language=language,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        if start_seconds is not None and end_seconds is not None
        else await aligner.align(path, expected, language=language)
    )
    expected_tokens = lexical_tokens(expected)
    recognized_tokens = tuple(token.text for token in result.tokens)
    resolved_type = clip_type or classify_clip_type(expected_text)
    confidence = evaluate_alignment(
        result,
        clip_type=resolved_type,
        expected_text=expected_text,
    )
    reasons = list(confidence.reason_codes)
    if expected_text is not None:
        decision = validate_extracted_alignment(
            result,
            target_text=expected,
            minimum_confidence=0.2,
            maximum_extra_speech_ms=60,
            minimum_duration_seconds=0.06 * max(1, len(expected_tokens)),
            maximum_duration_seconds=1.6 + 0.75 * (len(expected_tokens) - 1),
        )
        reasons.extend(decision.reason_codes)
        if recognized_tokens != expected_tokens:
            reasons.append("expected_transcript_mismatch")
    return WhisperInspection(
        path=path.resolve(),
        expected_text=expected_text,
        result=result,
        duration_seconds=wav_duration_seconds(path),
        token_diff=_token_diff(expected_tokens, recognized_tokens),
        reason_codes=tuple(dict.fromkeys(reasons)),
        confidence=confidence,
    )


def _token_diff(
    expected: tuple[str, ...], recognized: tuple[str, ...]
) -> tuple[dict[str, object], ...]:
    matcher = SequenceMatcher(a=expected, b=recognized, autojunk=False)
    return tuple(
        _diff_entry(expected, recognized, opcode)
        for opcode in matcher.get_opcodes()
        if opcode[0] != "equal"
    )


def _diff_entry(
    expected: tuple[str, ...],
    recognized: tuple[str, ...],
    opcode: tuple[str, int, int, int, int],
) -> dict[str, object]:
    operation, expected_start, expected_end, recognized_start, recognized_end = opcode
    return {
        "operation": operation,
        "expected_start": expected_start,
        "expected_end": expected_end,
        "recognized_start": recognized_start,
        "recognized_end": recognized_end,
        "expected": list(expected[expected_start:expected_end]),
        "recognized": list(recognized[recognized_start:recognized_end]),
    }
