"""Bundled Draft 2020-12 schemas for yakbox JSON contracts."""

from __future__ import annotations

import json
from importlib.resources import files


def load_schema(name: str, *, version: int = 1) -> dict[str, object]:
    """Load a bundled schema by contract name and explicit version."""
    if version < 1:
        raise ValueError("Schema version must be positive")
    resource = files(__package__).joinpath(f"{name}-v{version}.schema.json")
    value = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Schema {name!r} is not a JSON object")
    return value


def schema_names(*, version: int = 1) -> tuple[str, ...]:
    """Return bundled contract names for one explicit schema version."""
    if version < 1:
        raise ValueError("Schema version must be positive")
    suffix = f"-v{version}.schema.json"
    return tuple(
        sorted(
            resource.name.removesuffix(suffix)
            for resource in files(__package__).iterdir()
            if resource.name.endswith(suffix)
        )
    )


__all__ = ["load_schema", "schema_names"]
