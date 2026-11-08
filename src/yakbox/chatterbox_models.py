"""Local Chatterbox snapshot provenance, verification, and stable identities."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

from yakbox._files import atomic_write_bytes, safe_child, sha256_file
from yakbox.contracts import runtime_metadata, schema_uri
from yakbox.errors import ModelIntegrityError, ValidationError

RECORD_NAME = "yakbox-chatterbox-model.toml"
VARIANTS = ("original", "turbo", "mlx-turbo")
_ALLOWED_SUFFIXES = {
    ".safetensors",
    ".json",
    ".txt",
    ".model",
    ".pt",
    ".jinja",
    ".md",
    ".yaml",
    ".yml",
}
type _FileState = tuple[str, int, int, int, int]
type _SnapshotState = tuple[Path, str, str | None, tuple[_FileState, ...]]


class SnapshotVerifier:
    """Reuse a worker's verification while every file's filesystem identity holds."""

    def __init__(self) -> None:
        self._previous: _SnapshotState | None = None

    def verify(self, directory: Path, *, variant: str, fingerprint: str | None) -> None:
        """Hash on first use and when files or the expected identity change."""
        root = directory.resolve()
        try:
            files = _files(root)
            files[RECORD_NAME] = safe_child(root, root / RECORD_NAME)
            state = []
            for name, path in files.items():
                info = path.stat()
                state.append(
                    (
                        name,
                        info.st_size,
                        info.st_mtime_ns,
                        info.st_ctime_ns,
                        info.st_ino,
                    )
                )
        except OSError as error:
            raise ModelIntegrityError("Cannot inspect Chatterbox snapshot") from error
        identity = (root, variant, fingerprint, tuple(state))
        if identity != self._previous:
            verify_snapshot(root, variant=variant, fingerprint=fingerprint)
            self._previous = identity


def _files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if ".cache" in path.relative_to(root).parts:
            continue  # Hugging Face's local download bookkeeping is not model data.
        if path.is_symlink():
            raise ModelIntegrityError("Model snapshot must not contain symlinks")
        if not path.is_file() or path.name == RECORD_NAME:
            continue
        if path.suffix not in _ALLOWED_SUFFIXES and path.name not in {
            "LICENSE",
            "NOTICE",
            ".gitattributes",
        }:
            raise ModelIntegrityError("Model snapshot contains unsupported files")
        files[path.relative_to(root).as_posix()] = safe_child(root, path)
    return files


def _required(files: dict[str, Path], variant: str) -> None:
    if variant not in VARIANTS:
        raise ModelIntegrityError("Unknown Chatterbox snapshot variant")
    required = {"ve.safetensors"}
    if variant == "original":
        required |= {
            "s3gen.safetensors",
            "t3_cfg.safetensors",
            "tokenizer.json",
            "conds.pt",
        }
    elif variant == "turbo":
        required |= {
            "s3gen_meanflow.safetensors",
            "t3_turbo_v1.safetensors",
            "tokenizer_config.json",
        }
    else:
        required = {
            "config.json",
            "conds.safetensors",
            "tokenizer_config.json",
            "s3tokenizer/model.safetensors",
        }
        if not any(
            "/" not in name
            and name.endswith(".safetensors")
            and name != "conds.safetensors"
            for name in files
        ):
            raise ModelIntegrityError("MLX snapshot has no safetensors weights")
    if required - files.keys():
        raise ModelIntegrityError("Chatterbox snapshot is missing required model files")
    _check_json_configs(files, variant)


def _check_json_configs(files: dict[str, Path], variant: str) -> None:
    for name, path in files.items():
        if name.endswith(".json"):
            try:
                raw = json.loads(path.read_bytes())
            except (ValueError, UnicodeError) as error:
                raise ModelIntegrityError("Invalid model JSON configuration") from error
            if _remote_code(raw):
                raise ModelIntegrityError("Model configuration requests remote code")
            if (
                variant == "mlx-turbo"
                and name == "config.json"
                and (
                    not isinstance(raw, dict)
                    or raw.get("model_type") != "chatterbox_turbo"
                )
            ):
                raise ModelIntegrityError("MLX snapshot must select chatterbox_turbo")


def _remote_code(value: object) -> bool:
    if isinstance(value, dict):
        return bool(value.get("auto_map") or value.get("trust_remote_code")) or any(
            _remote_code(item) for item in value.values()
        )
    return isinstance(value, list) and any(_remote_code(item) for item in value)


def _identity(value: dict[str, object]) -> str:
    content = {
        name: value[name]
        for name in ("variant", "repository", "revision", "license", "files")
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_header(value: dict[str, object]) -> None:
    required = {
        "$schema",
        "schema_version",
        "yakbox_version",
        "timestamp",
        "variant",
        "repository",
        "revision",
        "source_url",
        "license",
        "fingerprint",
        "files",
    }
    if (
        set(value) != required
        or value.get("$schema") != schema_uri("chatterbox-model")
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise ModelIntegrityError("Unsupported Chatterbox snapshot record")
    version, timestamp = value.get("yakbox_version"), value.get("timestamp")
    if (
        not isinstance(version, str)
        or not version
        or not isinstance(timestamp, str)
        or datetime.fromisoformat(timestamp).utcoffset() != timedelta(0)
    ):
        raise ModelIntegrityError("Invalid Chatterbox snapshot envelope")


def record_snapshot(
    directory: Path,
    *,
    variant: str,
    repository: str,
    revision: str,
    license_id: str | None = None,
) -> dict[str, object]:
    """Record an explicitly acquired, trusted model snapshot without downloading."""
    if variant not in VARIANTS or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValidationError("Select a Chatterbox variant and full immutable revision")
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None:
        raise ValidationError("Model repository must be owner/name")
    if license_id is None and variant != "mlx-turbo":
        license_id = "MIT"
    if license_id not in {"MIT", "Apache-2.0"}:
        raise ValidationError("Select the reviewed model license: MIT or Apache-2.0")
    root = directory.resolve()
    files = _files(root)
    _required(files, variant)
    value = {
        **runtime_metadata("chatterbox-model"),
        "variant": variant,
        "repository": repository,
        "revision": revision,
        "license": license_id,
        "source_url": f"https://huggingface.co/{repository}/tree/{revision}",
        "files": [
            {"path": name, "sha256": sha256_file(path), "size": path.stat().st_size}
            for name, path in files.items()
        ],
    }
    value["fingerprint"] = _identity(value)
    text = (
        "\n".join(
            f"{json.dumps(key)} = {json.dumps(item)}"
            for key, item in value.items()
            if key != "files"
        )
        + "\n"
    )
    for item in cast(list[dict[str, object]], value["files"]):
        text += (
            "\n[[files]]\n"
            + "\n".join(f"{key} = {json.dumps(entry)}" for key, entry in item.items())
            + "\n"
        )
    atomic_write_bytes(safe_child(root, root / RECORD_NAME), text.encode())
    return value


def verify_snapshot(
    directory: Path,
    *,
    variant: str,
    fingerprint: str | None = None,
) -> dict[str, object]:
    """Verify every recorded file and its immutable identity before local loading."""
    root = directory.resolve()
    try:
        value = tomllib.loads(
            safe_child(root, root / RECORD_NAME).read_text(encoding="utf-8")
        )
        _validate_header(value)
        revision = value.get("revision")
        repository = value.get("repository")
        if (
            not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or not isinstance(repository, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None
            or value.get("source_url")
            != f"https://huggingface.co/{repository}/tree/{revision}"
        ):
            raise ModelIntegrityError("Invalid Chatterbox snapshot provenance")
        files = _files(root)
        _required(files, variant)
        declared = value["files"]
        if (
            not isinstance(declared, list)
            or {item["path"] for item in declared} != files.keys()
            or len(declared) != len(files)
        ):
            raise ModelIntegrityError("Chatterbox snapshot file set differs")
        for item in declared:
            path = files[item["path"]]
            if (
                type(item["size"]) is not int
                or not isinstance(item["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
                or path.stat().st_size != item["size"]
                or sha256_file(path) != item["sha256"]
            ):
                raise ModelIntegrityError("Chatterbox snapshot file digest differs")
        identity = _identity(value)
        if (
            value["variant"] != variant
            or value["license"] not in {"MIT", "Apache-2.0"}
            or value["fingerprint"] != identity
            or (fingerprint is not None and fingerprint != identity)
        ):
            raise ModelIntegrityError("Chatterbox snapshot identity differs")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ModelIntegrityError("Cannot verify Chatterbox snapshot record") from error
    return value
