"""Configuration loading with explicit precedence and no database."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from yakbox.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class YakboxConfig:
    default_backend: str = "fake"


def default_config_path() -> Path:
    configured = os.environ.get("YAKBOX_CONFIG")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".config" / "yakbox" / "config.toml"


def load_config(path: Path | None = None) -> YakboxConfig:
    source = path or default_config_path()
    raw: dict[str, object] = {}
    if source.exists():
        try:
            raw = tomllib.loads(source.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ConfigurationError(
                f"Cannot read configuration {source}: {error}"
            ) from error
    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ConfigurationError("Configuration sections must be TOML tables")
    return YakboxConfig(
        default_backend=os.environ.get(
            "YAKBOX_BACKEND", str(defaults.get("backend", "fake"))
        ),
    )
