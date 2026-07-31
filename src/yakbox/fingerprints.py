"""Stable runtime fingerprints for artifact-producing tools and backends."""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

from yakbox import __version__
from yakbox.audiobook.manifest import BackendProfile


def backend_fingerprint(profile: BackendProfile) -> str:
    return _backend_fingerprint(profile.backend, profile.executor)


def backend_versions(profile: BackendProfile) -> dict[str, str]:
    return _backend_versions(profile.backend)


def backend_runtime_fingerprint(backend: str) -> str:
    return _backend_fingerprint(backend, None)


def _backend_fingerprint(backend: str, executor: str | None) -> str:
    packages = _backend_versions(backend)
    payload = {
        "backend": backend,
        "executor": executor,
        "packages": packages,
    }
    return _digest(payload)


def _backend_versions(backend: str) -> dict[str, str]:
    packages = ["yakbox"]
    if backend in {"local", "chatterbox", "chatterbox-local"}:
        packages.extend(("chatterbox-tts", "torch", "torchaudio"))
    elif backend in {"resemble", "cloud"}:
        packages.append("httpx")
    return {name: _package_version(name) for name in packages}


@lru_cache(maxsize=1)
def media_tool_fingerprint() -> str:
    return _digest(media_tool_versions())


def media_tool_versions() -> dict[str, str]:
    return {
        "yakbox": __version__,
        "ffmpeg": _command_version("ffmpeg"),
        "ffprobe": _command_version("ffprobe"),
    }


def _package_version(name: str) -> str:
    if name == "yakbox":
        return __version__
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


@lru_cache(maxsize=4)
def _command_version(command: str) -> str:
    try:
        result = subprocess.run(
            [command, "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    first_line = (result.stdout or result.stderr).splitlines()
    return first_line[0][:512] if first_line else f"exit-{result.returncode}"


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
