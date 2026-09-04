"""Redacted execution identities for direct and isolated model adapters."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path

from yakbox._files import sha256_file
from yakbox.speech.analysis_models import ExecutionIdentity


def execution_identity(
    *,
    worker_artifact: Path,
    dependency_lock: Path,
    device_class: str,
    determinism_mode: str = "greedy",
    decode_seeds: tuple[int, ...] = (),
) -> ExecutionIdentity:
    """Build an identity without usernames, machine IDs, or local paths."""
    return ExecutionIdentity(
        worker_artifact_digest=sha256_file(worker_artifact),
        lock_digest=sha256_file(dependency_lock),
        python_version=(
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        os_family=platform.system(),
        os_version=(
            platform.mac_ver()[0]
            if platform.system() == "Darwin"
            else platform.release()
        ),
        architecture=platform.machine(),
        mlx_version=_optional_package_version("mlx"),
        metal_version=None,
        device_class=device_class,
        determinism_mode=determinism_mode,
        decode_seeds=decode_seeds,
    )


def execution_identity_from_digests(
    *,
    worker_artifact_digest: str,
    lock_digest: str,
    device_class: str = "apple-silicon",
    determinism_mode: str = "greedy",
    decode_seeds: tuple[int, ...] = (),
) -> ExecutionIdentity:
    """Build the same redacted identity from supervisor-verified digests."""
    return ExecutionIdentity(
        worker_artifact_digest=worker_artifact_digest,
        lock_digest=lock_digest,
        python_version=(
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        os_family=platform.system(),
        os_version=(
            platform.mac_ver()[0]
            if platform.system() == "Darwin"
            else platform.release()
        ),
        architecture=platform.machine(),
        mlx_version=_optional_package_version("mlx"),
        metal_version=None,
        device_class=device_class,
        determinism_mode=determinism_mode,
        decode_seeds=decode_seeds,
    )


def _optional_package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


__all__ = ["execution_identity", "execution_identity_from_digests"]
