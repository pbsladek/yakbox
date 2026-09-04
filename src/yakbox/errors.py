"""Public application exceptions with stable machine-readable codes."""

from __future__ import annotations

from typing import ClassVar


class YakboxError(Exception):
    """Base class for expected yakbox failures."""

    code: ClassVar[str] = "yakbox_error"


class ValidationError(YakboxError):
    """User input or configuration is invalid."""

    code = "validation_error"


class ConfigurationError(YakboxError):
    """Required configuration is missing or inconsistent."""

    code = "configuration_error"


class BackendUnavailableError(YakboxError):
    """A requested backend is not installed or configured."""

    code = "backend_unavailable"


class BuildError(YakboxError):
    """An audiobook build could not complete."""

    code = "build_error"


class ArtifactError(YakboxError):
    """An artifact operation could not be completed safely."""

    code = "artifact_error"


class ModelIntegrityError(YakboxError):
    """A registered local model does not match its immutable record."""

    code = "model_integrity_error"


class WorkerProtocolError(YakboxError):
    """An analysis worker request or response violates its protocol."""

    code = "worker_protocol_error"


class SpeechAnalysisError(YakboxError):
    """Speech evidence cannot satisfy the configured analysis policy."""

    code = "speech_analysis_error"


def stable_error_code(error: BaseException) -> str:
    """Return a compatibility-safe error code for an expected failure."""
    if isinstance(error, YakboxError):
        return error.code
    if isinstance(error, OSError):
        return "io_error"
    if isinstance(error, ValueError):
        return "invalid_value"
    return "internal_error"
