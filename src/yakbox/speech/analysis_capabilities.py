"""Versioned capability resolution for registered speech-analysis engines."""

from __future__ import annotations

from dataclasses import dataclass

from yakbox.errors import ValidationError
from yakbox.speech.analysis_fingerprints import semantic_fingerprint


@dataclass(frozen=True, slots=True)
class AnalysisCapabilities:
    """Operations one engine is qualified to perform for one language."""

    recognition: bool
    forced_alignment: bool
    word_timing: bool
    character_timing: bool
    batching: bool
    maximum_duration_frames: int
    sample_rate: int = 16_000

    def __post_init__(self) -> None:
        if self.maximum_duration_frames <= 0 or self.sample_rate <= 0:
            raise ValidationError("Analysis capability durations must be positive")
        if self.forced_alignment and self.recognition:
            raise ValidationError(
                "One registered engine role cannot vote and force-align"
            )


@dataclass(frozen=True, slots=True)
class EngineLanguageCapabilities:
    """Capabilities for one canonical Yakbox language identifier."""

    language: str
    upstream_language: str
    capabilities: AnalysisCapabilities


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """Versioned language matrix for one logical analysis engine."""

    engine: str
    backend: str
    languages: tuple[EngineLanguageCapabilities, ...]

    def resolve(self, language: str) -> EngineLanguageCapabilities:
        matches = tuple(item for item in self.languages if item.language == language)
        if len(matches) != 1:
            raise ValidationError(
                f"Speech-analysis engine {self.engine!r} does not support {language!r}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class CapabilityMatrix:
    """Complete, fingerprinted engine capability registry."""

    version: int
    engines: tuple[EngineCapabilities, ...]

    def __post_init__(self) -> None:
        if self.version < 1 or not self.engines:
            raise ValidationError("Capability matrix must be versioned and non-empty")
        names = tuple(engine.engine for engine in self.engines)
        if len(names) != len(set(names)):
            raise ValidationError("Capability matrix engine names must be unique")

    @property
    def fingerprint(self) -> str:
        return semantic_fingerprint("speech-capability-matrix-v1", self)

    def resolve_recognizer(
        self, engine: str, language: str
    ) -> EngineLanguageCapabilities:
        resolved = self._engine(engine).resolve(language)
        if not resolved.capabilities.recognition:
            raise ValidationError(
                f"Speech-analysis engine {engine!r} is not a recognizer"
            )
        return resolved

    def resolve_forced_aligner(
        self, engine: str, language: str
    ) -> EngineLanguageCapabilities:
        resolved = self._engine(engine).resolve(language)
        if not resolved.capabilities.forced_alignment:
            raise ValidationError(
                f"Speech-analysis engine {engine!r} is not a forced aligner"
            )
        return resolved

    def _engine(self, name: str) -> EngineCapabilities:
        matches = tuple(engine for engine in self.engines if engine.engine == name)
        if len(matches) != 1:
            raise ValidationError(f"Unknown speech-analysis engine: {name}")
        return matches[0]


def default_capability_matrix() -> CapabilityMatrix:
    """Return the English-only Phase 0 capability declaration."""
    minute = 60 * 16_000
    return CapabilityMatrix(
        version=1,
        engines=(
            EngineCapabilities(
                "whisper",
                "mlx-whisper",
                (
                    EngineLanguageCapabilities(
                        "en",
                        "en",
                        AnalysisCapabilities(
                            True, False, True, False, False, 30 * minute
                        ),
                    ),
                ),
            ),
            EngineCapabilities(
                "parakeet",
                "parakeet-mlx",
                (
                    EngineLanguageCapabilities(
                        "en",
                        "en",
                        AnalysisCapabilities(
                            True, False, True, False, True, 24 * minute
                        ),
                    ),
                ),
            ),
            EngineCapabilities(
                "qwen",
                "mlx-audio-qwen3-asr",
                (
                    EngineLanguageCapabilities(
                        "en",
                        "English",
                        AnalysisCapabilities(
                            True, False, False, False, True, 30 * minute
                        ),
                    ),
                ),
            ),
            EngineCapabilities(
                "qwen-forced",
                "mlx-audio-qwen3-forced",
                (
                    EngineLanguageCapabilities(
                        "en",
                        "English",
                        AnalysisCapabilities(False, True, True, True, True, 5 * minute),
                    ),
                ),
            ),
        ),
    )


__all__ = [
    "AnalysisCapabilities",
    "CapabilityMatrix",
    "EngineCapabilities",
    "EngineLanguageCapabilities",
    "default_capability_matrix",
]
