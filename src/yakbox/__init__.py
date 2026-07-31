"""Yakbox audiobook build system."""

from importlib.metadata import PackageNotFoundError, version

from yakbox.errors import (
    ArtifactError,
    BackendUnavailableError,
    BuildError,
    ConfigurationError,
    ValidationError,
    YakboxError,
)

try:
    __version__ = version("yakbox")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.1.0"

__all__ = [
    "ArtifactError",
    "BackendUnavailableError",
    "BuildError",
    "ConfigurationError",
    "ValidationError",
    "YakboxError",
    "__version__",
]
