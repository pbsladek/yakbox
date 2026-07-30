"""Public application exceptions."""


class YakboxError(Exception):
    """Base class for expected yakbox failures."""


class ValidationError(YakboxError):
    """User input or configuration is invalid."""


class ConfigurationError(YakboxError):
    """Required configuration is missing or inconsistent."""


class BackendUnavailableError(YakboxError):
    """A requested backend is not installed or configured."""


class BuildError(YakboxError):
    """An audiobook build could not complete."""


class ArtifactError(YakboxError):
    """An artifact operation could not be completed safely."""
