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
    legacy_resemble_api_key: str | None = None
    resemble_voice_uuid: str | None = None
    resemble_project_uuid: str | None = None
    cloud_concurrency: int = 5


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
    cloud = raw.get("cloud", {})
    defaults = raw.get("defaults", {})
    if not isinstance(cloud, dict) or not isinstance(defaults, dict):
        raise ConfigurationError("Configuration sections must be TOML tables")
    concurrency_value = os.environ.get(
        "YAKBOX_CLOUD_CONCURRENCY", str(cloud.get("concurrency", 5))
    )
    try:
        concurrency = int(concurrency_value)
    except ValueError as error:
        raise ConfigurationError("Cloud concurrency must be an integer") from error
    if concurrency < 1:
        raise ConfigurationError("Cloud concurrency must be at least 1")
    legacy_api_key = _optional_string(cloud.get("api_key"))
    return YakboxConfig(
        default_backend=os.environ.get(
            "YAKBOX_BACKEND", str(defaults.get("backend", "fake"))
        ),
        legacy_resemble_api_key=legacy_api_key,
        resemble_voice_uuid=(
            os.environ.get("YAKBOX_CLOUD_VOICE_UUID")
            or os.environ.get("RESEMBLE_VOICE_UUID")
            or _optional_string(cloud.get("voice_uuid"))
        ),
        resemble_project_uuid=(
            os.environ.get("YAKBOX_CLOUD_PROJECT_UUID")
            or os.environ.get("RESEMBLE_PROJECT_UUID")
            or _optional_string(cloud.get("project_uuid"))
        ),
        cloud_concurrency=concurrency,
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
