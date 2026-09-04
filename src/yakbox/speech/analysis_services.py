"""Separate recognition and forced-alignment service boundaries."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from yakbox.speech.analysis_models import (
    AlignmentPurpose,
    AudioSpan,
    ForcedAlignmentResult,
    RecognitionResult,
    VerifiedTextSpan,
)


@runtime_checkable
class SpeechRecognizer(Protocol):
    """Independent recognizer that never receives expected transcript text."""

    @property
    def fingerprint(self) -> str: ...

    async def recognize(
        self,
        audio: Path,
        *,
        language: str,
        span: AudioSpan | None = None,
    ) -> RecognitionResult: ...


@runtime_checkable
class ForcedAligner(Protocol):
    """Timing-only aligner for already-authorized text."""

    @property
    def fingerprint(self) -> str: ...

    async def force_align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
        purpose: AlignmentPurpose,
        verified_span: VerifiedTextSpan | None = None,
        span: AudioSpan | None = None,
    ) -> ForcedAlignmentResult: ...


class FakeSpeechRecognizer:
    """Deterministic scripted recognizer for offline contract tests."""

    def __init__(
        self,
        fingerprint: str,
        result_factory: Callable[[Path, str, AudioSpan | None], RecognitionResult],
    ) -> None:
        self._fingerprint = fingerprint
        self._result_factory = result_factory

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    async def recognize(
        self,
        audio: Path,
        *,
        language: str,
        span: AudioSpan | None = None,
    ) -> RecognitionResult:
        return self._result_factory(audio, language, span)


class FakeForcedAligner:
    """Deterministic scripted forced aligner for offline contract tests."""

    def __init__(
        self,
        fingerprint: str,
        result_factory: Callable[
            [
                Path,
                str,
                str,
                AlignmentPurpose,
                VerifiedTextSpan | None,
                AudioSpan | None,
            ],
            ForcedAlignmentResult,
        ],
    ) -> None:
        self._fingerprint = fingerprint
        self._result_factory = result_factory

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    async def force_align(
        self,
        audio: Path,
        expected_text: str,
        *,
        language: str,
        purpose: AlignmentPurpose,
        verified_span: VerifiedTextSpan | None = None,
        span: AudioSpan | None = None,
    ) -> ForcedAlignmentResult:
        if purpose is not AlignmentPurpose.NON_AUTHORITATIVE and verified_span is None:
            raise ValueError("Authoritative forced alignment requires a verified span")
        return self._result_factory(
            audio,
            expected_text,
            language,
            purpose,
            verified_span,
            span,
        )


__all__ = [
    "FakeForcedAligner",
    "FakeSpeechRecognizer",
    "ForcedAligner",
    "SpeechRecognizer",
]
