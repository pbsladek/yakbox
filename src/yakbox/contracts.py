from __future__ import annotations

from datetime import UTC, datetime

from yakbox import __version__

SCHEMA_BASE = "https://yakbox.dev/schemas"


def schema_uri(name: str) -> str:
    """Return the stable absolute identifier for a packaged JSON schema."""
    return f"{SCHEMA_BASE}/{name}-v1.schema.json"


def utc_timestamp() -> str:
    """Return an RFC 3339-compatible UTC timestamp."""
    return datetime.now(UTC).isoformat()


def runtime_metadata(name: str, *, timestamp: str | None = None) -> dict[str, object]:
    """Metadata required on every generated, machine-readable document."""
    return {
        "$schema": schema_uri(name),
        "schema_version": 1,
        "yakbox_version": __version__,
        "timestamp": timestamp or utc_timestamp(),
    }
