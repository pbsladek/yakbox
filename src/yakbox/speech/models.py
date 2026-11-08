from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from yakbox.errors import ValidationError

CURRENCY_CODE_LENGTH = 3


class CurrencyCode(str):
    """Validated three-letter uppercase currency code."""

    def __new__(cls, value: str) -> CurrencyCode:
        normalized = value.strip().upper()
        if (
            len(normalized) != CURRENCY_CODE_LENGTH
            or not normalized.isascii()
            or not normalized.isalpha()
        ):
            raise ValidationError("Currency code must contain three ASCII letters")
        return super().__new__(cls, normalized)


class PricingSourceId(str):
    """Validated non-empty identifier for the source of hosted pricing."""

    def __new__(cls, value: str) -> PricingSourceId:
        normalized = value.strip()
        if not normalized:
            raise ValidationError("Pricing source must not be empty")
        return super().__new__(cls, normalized)


class AudioFormat(StrEnum):
    WAV = "wav"
    MP3 = "mp3"


class Precision(StrEnum):
    """Provider-neutral audio sample precision."""

    MULAW = "MULAW"
    PCM_16 = "PCM_16"
    PCM_24 = "PCM_24"
    PCM_32 = "PCM_32"


@dataclass(frozen=True, slots=True)
class ChatterboxSynthesisOptions:
    """Per-request generation controls for Chatterbox speech synthesis."""

    cfg_weight: float | None = None
    exaggeration: float | None = None
    seed: int | None = None
    variant: str = "original"
    model_directory: Path | None = None
    model_fingerprint: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None

    def __post_init__(self) -> None:
        if self.variant not in {"original", "turbo", "mlx-turbo"}:
            raise ValidationError("Unknown Chatterbox variant")
        if (self.model_directory is None) != (self.model_fingerprint is None):
            raise ValidationError(
                "Local models require a directory and recorded fingerprint"
            )
        if (
            self.model_fingerprint is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.model_fingerprint) is None
        ):
            raise ValidationError("Model fingerprint must be SHA-256")
        if self.variant == "mlx-turbo" and self.model_directory is None:
            raise ValidationError("MLX Turbo requires a verified local model snapshot")
        if self.variant != "original" and any(
            value not in (None, 0) for value in (self.cfg_weight, self.exaggeration)
        ):
            raise ValidationError("Turbo variants do not support CFG or exaggeration")
        for name, value in (
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("repetition_penalty", self.repetition_penalty),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValidationError(f"Chatterbox {name} must be positive and finite")
        if self.top_p is not None and self.top_p > 1:
            raise ValidationError("Chatterbox top_p must be at most 1")

    def to_dict(self) -> dict[str, object]:
        """Serialize generation controls and optional local snapshot identity."""
        result: dict[str, object] = {
            "cfg_weight": self.cfg_weight,
            "exaggeration": self.exaggeration,
            "seed": self.seed,
        }
        if self.variant != "original" or self.model_directory is not None:
            result.update(
                variant=self.variant,
                model_directory=str(self.model_directory)
                if self.model_directory
                else None,
                model_fingerprint=self.model_fingerprint,
            )
        for name in ("temperature", "top_p", "repetition_penalty"):
            if (value := getattr(self, name)) is not None:
                result[name] = value
        return result

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ChatterboxSynthesisOptions:
        """Decode worker options through the shared validation boundary."""
        numbers: dict[str, float | None] = {}
        for name in (
            "cfg_weight",
            "exaggeration",
            "temperature",
            "top_p",
            "repetition_penalty",
        ):
            item = value.get(name)
            if item is not None and (
                isinstance(item, bool) or not isinstance(item, (int, float))
            ):
                raise ValidationError("Invalid Chatterbox numeric option")
            numbers[name] = float(item) if isinstance(item, (int, float)) else None
        seed = value.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValidationError("Invalid Chatterbox seed")
        path = value.get("model_directory")
        fingerprint = value.get("model_fingerprint")
        if (path is not None and not isinstance(path, str)) or (
            fingerprint is not None and not isinstance(fingerprint, str)
        ):
            raise ValidationError("Invalid local model identity")
        return cls(
            **numbers,
            seed=seed,
            variant=str(value.get("variant", "original")),
            model_directory=Path(path) if isinstance(path, str) else None,
            model_fingerprint=fingerprint if isinstance(fingerprint, str) else None,
        )


@dataclass(frozen=True, slots=True)
class SpeechSynthesisRequest:
    """Provider-neutral request for generating speech from text."""

    text: str
    voice: str
    backend: str = "fake"
    profile: str | None = None
    output_format: AudioFormat = AudioFormat.WAV
    sample_rate: int | None = None
    title: str | None = None
    use_hd: bool = False
    precision: Precision | None = None
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
    """Provider-neutral request for transforming an existing audio file."""

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
    """Persisted speech output with backend provenance and integrity metadata."""

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
    """Hard limits and confirmation thresholds for billable hosted work."""

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
class SpeechBackendOptions:
    """Typed configuration for opening a speech backend."""

    isolated_local: bool = False
    device: str | None = None
    local_worker_timeout_seconds: float = 3_600
    local_threads_per_process: int = 1
    local_worker_log_path: Path | None = None


@dataclass(frozen=True, slots=True)
class HostedUsageSnapshot:
    """Observed hosted requests, submitted text, and estimated spend totals."""

    logical_items: int = 0
    provider_attempts: int = 0
    submitted_characters: int = 0
    estimated_spend: Decimal | None = None
    currency: CurrencyCode | None = None
    ambiguous_attempts: int = 0
