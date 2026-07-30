"""Bundled Draft 2020-12 schemas for yakbox JSON contracts."""

from __future__ import annotations

import json
from importlib.resources import files


def load_schema(name: str) -> dict[str, object]:
    """Load a bundled schema by its unversioned contract name."""
    resource = files(__package__).joinpath(f"{name}-v1.schema.json")
    value = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Schema {name!r} is not a JSON object")
    return value


def schema_names() -> tuple[str, ...]:
    """Return the bundled version-1 schema contract names."""
    suffix = "-v1.schema.json"
    return tuple(
        sorted(
            resource.name.removesuffix(suffix)
            for resource in files(__package__).iterdir()
            if resource.name.endswith(suffix)
        )
    )


__all__ = ["load_schema", "schema_names"]
