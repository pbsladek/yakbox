"""Canonical fingerprints for speech-analysis evidence and policy inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path


def semantic_fingerprint(namespace: str, value: object) -> str:
    """Hash a typed semantic value without operational or display metadata."""
    payload = {
        "namespace": namespace,
        "value": _canonical(value),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def text_fingerprint(value: str) -> str:
    """Hash private text for use in evidence references."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: object) -> object:
    if value is None or isinstance(value, str | int | bool):
        result: object = value
    elif isinstance(value, float):
        result = value if not value.is_integer() else int(value)
    elif isinstance(value, Enum):
        result = value.value
    elif isinstance(value, Path):
        raise TypeError("Local paths are not valid semantic fingerprint inputs")
    elif is_dataclass(value) and not isinstance(value, type):
        result = {
            field.name: _canonical(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("operation_")
        }
    elif isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Fingerprint mappings require string keys")
        result = {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    elif isinstance(value, tuple | list):
        result = [_canonical(item) for item in value]
    else:
        raise TypeError(f"Unsupported fingerprint value: {type(value).__name__}")
    return result


__all__ = ["semantic_fingerprint", "text_fingerprint"]
