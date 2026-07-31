from __future__ import annotations

import pytest

from yakbox.credentials import CredentialSource, resolve_resemble_credential


@pytest.mark.parametrize(
    ("values", "expected_source", "expected_value"),
    [
        (
            ("argument", "environment", "keyring", "legacy"),
            CredentialSource.EXPLICIT,
            "argument",
        ),
        (
            (None, "environment", "keyring", "legacy"),
            CredentialSource.ENVIRONMENT,
            "environment",
        ),
        ((None, None, "keyring", "legacy"), CredentialSource.KEYRING, "keyring"),
        ((None, None, None, "legacy"), CredentialSource.LEGACY_CONFIG, "legacy"),
        ((None, None, None, None), None, None),
    ],
)
def test_credential_precedence_preserves_source(
    values: tuple[str | None, str | None, str | None, str | None],
    expected_source: CredentialSource | None,
    expected_value: str | None,
) -> None:
    resolved = resolve_resemble_credential(
        explicit=values[0],
        environment=values[1],
        keyring=values[2],
        legacy_config=values[3],
        profile="studio",
    )

    assert (resolved.source if resolved else None) is expected_source
    assert (resolved.value if resolved else None) == expected_value
