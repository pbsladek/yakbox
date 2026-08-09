"""Explicit offline model management for phoneme CTC alignment."""

from __future__ import annotations

import importlib
import importlib.util
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

from yakbox.errors import BackendUnavailableError
from yakbox.whisper_models import model_status as _base_status

DEFAULT_PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
DEFAULT_PHONEME_REVISION = "c43348bbaa5a77692c8e7bf3409d683474fdf2a4"


class _HuggingFaceHub(Protocol):
    def snapshot_download(
        self,
        *,
        repo_id: str,
        revision: str,
        local_files_only: bool = False,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class PhonemeModelStatus:
    """Local model, Python runtime, and external phonemizer readiness."""

    model: str
    revision: str | None
    installed: bool
    verified: bool
    local_path: Path | None
    size_bytes: int
    file_count: int
    fingerprint: str | None
    issues: tuple[str, ...]
    torch_available: bool
    transformers_available: bool
    huggingface_hub_available: bool
    espeak_ng_available: bool

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-safe status document."""
        return {
            **asdict(self),
            "local_path": str(self.local_path) if self.local_path else None,
            "issues": list(self.issues),
        }


def phoneme_model_status(
    model: str,
    revision: str | None,
) -> PhonemeModelStatus:
    """Inspect a phoneme model snapshot without making a network request."""
    status = _base_status(model, revision)
    issues = [item for item in status.issues if item != "weights_missing"]
    if status.local_path is not None:
        files = tuple(status.local_path.rglob("*"))
        if not any(
            item.is_file()
            and (item.suffix == ".safetensors" or item.name == "pytorch_model.bin")
            for item in files
        ):
            issues.append("weights_missing")
        if not (status.local_path / "vocab.json").is_file():
            issues.append("vocabulary_missing")
    return PhonemeModelStatus(
        model=status.model,
        revision=status.revision,
        installed=status.installed,
        verified=status.installed and not issues,
        local_path=status.local_path,
        size_bytes=status.size_bytes,
        file_count=status.file_count,
        fingerprint=status.fingerprint,
        issues=tuple(dict.fromkeys(issues)),
        torch_available=importlib.util.find_spec("torch") is not None,
        transformers_available=importlib.util.find_spec("transformers") is not None,
        huggingface_hub_available=status.huggingface_hub_available,
        espeak_ng_available=shutil.which("espeak-ng") is not None,
    )


def install_phoneme_model(
    model: str,
    revision: str | None,
) -> PhonemeModelStatus:
    """Explicitly download and verify one pinned phoneme model."""
    candidate = Path(model).expanduser()
    if candidate.exists():
        return phoneme_model_status(model, revision)
    if revision is None:
        raise BackendUnavailableError(
            "Remote phoneme model installation requires a pinned revision"
        )
    try:
        _hub().snapshot_download(repo_id=model, revision=revision)
    except Exception as error:
        raise BackendUnavailableError(
            f"Could not install pinned phoneme model {model}@{revision}"
        ) from error
    return phoneme_model_status(model, revision)


def require_phoneme_model_path(model: str, revision: str | None) -> Path:
    """Resolve an installed verified phoneme model without downloading it."""
    status = phoneme_model_status(model, revision)
    if status.local_path is None:
        raise BackendUnavailableError(
            f"Phoneme model is not installed: {model}@{revision or 'unpinned'}; "
            "run yakbox whisper phoneme-models install"
        )
    if not status.verified:
        raise BackendUnavailableError(
            "Phoneme model verification failed: " + ", ".join(status.issues)
        )
    return status.local_path


def _hub() -> _HuggingFaceHub:
    if importlib.util.find_spec("huggingface_hub") is None:
        raise BackendUnavailableError(
            "Hugging Face model support is unavailable; install Yakbox with "
            "the phoneme extra"
        )
    return cast(_HuggingFaceHub, importlib.import_module("huggingface_hub"))
