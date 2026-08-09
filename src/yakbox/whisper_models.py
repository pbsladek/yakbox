"""Explicit, local-first model management for MLX Whisper."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

from yakbox.errors import BackendUnavailableError

DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_WHISPER_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"


class _HuggingFaceHub(Protocol):
    def snapshot_download(
        self,
        *,
        repo_id: str,
        revision: str,
        local_files_only: bool = False,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class WhisperModelStatus:
    """Resolved cache and integrity state for one pinned Whisper model."""

    model: str
    revision: str | None
    installed: bool
    verified: bool
    local_path: Path | None
    size_bytes: int
    file_count: int
    fingerprint: str | None
    issues: tuple[str, ...]
    mlx_whisper_available: bool
    huggingface_hub_available: bool

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-safe representation."""
        return {
            **asdict(self),
            "local_path": str(self.local_path) if self.local_path else None,
            "issues": list(self.issues),
        }


def model_status(model: str, revision: str | None) -> WhisperModelStatus:
    """Inspect a local path or pinned Hugging Face cache without network access."""
    issues: list[str] = []
    local_path: Path | None = None
    candidate = Path(model).expanduser()
    hub_available = importlib.util.find_spec("huggingface_hub") is not None
    if candidate.exists():
        local_path = candidate.resolve()
    elif revision is None:
        issues.append("remote_revision_unpinned")
    elif not hub_available:
        issues.append("huggingface_hub_unavailable")
    else:
        try:
            local_path = Path(
                _hub().snapshot_download(
                    repo_id=model,
                    revision=revision,
                    local_files_only=True,
                )
            ).resolve()
        except Exception:  # noqa: BLE001 - cache misses vary by hub version.
            issues.append("model_not_cached")
    if local_path is not None:
        issues.extend(_verify_directory(local_path))
    files = _model_files(local_path) if local_path is not None else ()
    size = sum(path.stat().st_size for path in files if path.is_file())
    fingerprint = _directory_fingerprint(local_path, files) if local_path else None
    return WhisperModelStatus(
        model=model,
        revision=revision,
        installed=local_path is not None,
        verified=local_path is not None and not issues,
        local_path=local_path,
        size_bytes=size,
        file_count=len(files),
        fingerprint=fingerprint,
        issues=tuple(dict.fromkeys(issues)),
        mlx_whisper_available=importlib.util.find_spec("mlx_whisper") is not None,
        huggingface_hub_available=hub_available,
    )


def install_model(model: str, revision: str | None) -> WhisperModelStatus:
    """Explicitly download a pinned model, then verify its local snapshot."""
    candidate = Path(model).expanduser()
    if candidate.exists():
        return model_status(model, revision)
    if revision is None:
        raise BackendUnavailableError(
            "Remote MLX Whisper model installation requires a pinned revision"
        )
    try:
        _hub().snapshot_download(repo_id=model, revision=revision)
    except Exception as error:
        raise BackendUnavailableError(
            f"Could not install pinned MLX Whisper model {model}@{revision}"
        ) from error
    return model_status(model, revision)


def require_model_path(model: str, revision: str | None) -> Path:
    """Resolve only an already-installed, verified model; never download it."""
    status = model_status(model, revision)
    if status.local_path is None:
        identity = f"{model}@{revision}" if revision else model
        raise BackendUnavailableError(
            f"MLX Whisper model is not installed: {identity}; run "
            f"yakbox whisper models install --model {model!r}"
        )
    if not status.verified:
        raise BackendUnavailableError(
            "MLX Whisper model verification failed: " + ", ".join(status.issues)
        )
    return status.local_path


def _hub() -> _HuggingFaceHub:
    try:
        return cast(_HuggingFaceHub, importlib.import_module("huggingface_hub"))
    except ImportError as error:
        raise BackendUnavailableError(
            "Hugging Face model support is unavailable; install Yakbox with "
            "the alignment extra"
        ) from error


def _verify_directory(path: Path) -> tuple[str, ...]:
    issues: list[str] = []
    if not path.is_dir():
        return ("model_path_not_directory",)
    files = _model_files(path)
    if not files:
        return ("model_directory_empty",)
    if not (path / "config.json").is_file():
        issues.append("config_missing")
    if not any(item.suffix == ".safetensors" for item in files):
        issues.append("weights_missing")
    if any(
        item.name.endswith(".incomplete") or item.stat().st_size == 0 for item in files
    ):
        issues.append("model_file_incomplete")
    try:
        raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            issues.append("config_invalid")
    except OSError, UnicodeDecodeError, json.JSONDecodeError:
        if "config_missing" not in issues:
            issues.append("config_invalid")
    return tuple(issues)


def _model_files(path: Path | None) -> tuple[Path, ...]:
    if path is None or not path.is_dir():
        return ()
    return tuple(sorted(item for item in path.rglob("*") if item.is_file()))


def _directory_fingerprint(path: Path, files: tuple[Path, ...]) -> str:
    payload = [
        (item.relative_to(path).as_posix(), item.stat().st_size) for item in files
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
