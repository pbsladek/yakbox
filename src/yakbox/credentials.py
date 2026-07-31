"""Credential precedence and provenance for hosted operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CredentialSource(StrEnum):
    """Supported credential sources in descending precedence order."""

    EXPLICIT = "explicit"
    ENVIRONMENT = "environment"
    KEYRING = "keyring"
    LEGACY_CONFIG = "legacy_config"


@dataclass(frozen=True, slots=True)
class ResolvedCredential:
    """A credential value together with the source that supplied it."""

    value: str
    source: CredentialSource
    profile: str | None = None


def resolve_resemble_credential(
    *,
    explicit: str | None,
    environment: str | None,
    keyring: str | None,
    legacy_config: str | None,
    profile: str,
) -> ResolvedCredential | None:
    """Resolve a Resemble credential without losing source provenance."""
    candidates = (
        (explicit, CredentialSource.EXPLICIT, None),
        (environment, CredentialSource.ENVIRONMENT, None),
        (keyring, CredentialSource.KEYRING, profile),
        (legacy_config, CredentialSource.LEGACY_CONFIG, None),
    )
    for value, source, selected_profile in candidates:
        if value:
            return ResolvedCredential(value, source, selected_profile)
    return None
