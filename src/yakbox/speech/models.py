from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import NewType

from yakbox.errors import ValidationError

CurrencyCode = NewType("CurrencyCode", str)
PricingSourceId = NewType("PricingSourceId", str)


class AudioFormat(StrEnum):
    WAV = "wav"
    MP3 = "mp3"


@dataclass(frozen=True, slots=True)
class ChatterboxSynthesisOptions:
    cfg_weight: float | None = None
    exaggeration: float | None = None
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class SpeechSynthesisRequest:
    text: str
    voice: str
    backend: str = "fake"
    profile: str | None = None
    output_format: AudioFormat = AudioFormat.WAV
    sample_rate: int | None = None
    title: str | None = None
    use_hd: bool = False
    precision: str | None = None
    apply_custom_pronunciations: bool = False
    project: str | None = None
    reference_audio: Path | None = None
    chatterbox: ChatterboxSynthesisOptions | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValidationError("Speech text must not be empty")
        if not self.voice.strip():
            raise ValidationError("A voice is required")
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValidationError("Sample rate must be positive")


@dataclass(frozen=True, slots=True)
class SpeechTransformationRequest:
    input_path: Path
    voice: str
    backend: str = "local"
    profile: str | None = None
    reference_audio: Path | None = None

    def __post_init__(self) -> None:
        if not self.input_path.is_file():
            raise ValidationError(f"Input audio does not exist: {self.input_path}")
        if not self.voice.strip():
            raise ValidationError("A voice is required")


@dataclass(frozen=True, slots=True)
class SpeechArtifact:
    path: Path
    backend: str
    voice: str
    output_format: AudioFormat
    bytes_written: int
    sha256: str
    duration_seconds: float | None = None
    sample_rate: int | None = None
    request_id: str | None = None
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class HostedUsageBudget:
    max_submitted_characters: int | None = None
    max_provider_requests: int | None = None
    max_estimated_spend: Decimal | None = None
    currency: CurrencyCode | None = None
    pricing_source: PricingSourceId | None = None
    confirm_above_characters: int | None = None
    confirm_above_requests: int | None = None

    def __post_init__(self) -> None:
        integers = (
            self.max_submitted_characters,
            self.max_provider_requests,
            self.confirm_above_characters,
            self.confirm_above_requests,
        )
        if any(value is not None and value < 0 for value in integers):
            raise ValidationError("Hosted usage limits cannot be negative")
        if self.max_estimated_spend is not None:
            if not self.max_estimated_spend.is_finite() or self.max_estimated_spend < 0:
                raise ValidationError(
                    "Hosted spending limit must be finite and non-negative"
                )
            if self.currency is None or self.pricing_source is None:
                raise ValidationError(
                    "A spending limit requires currency and pricing source"
                )


@dataclass(frozen=True, slots=True)
class HostedUsageSnapshot:
    logical_items: int = 0
    provider_attempts: int = 0
    submitted_characters: int = 0
    estimated_spend: Decimal | None = None
    currency: CurrencyCode | None = None
    ambiguous_attempts: int = 0
