"""Safe filesystem primitives shared by application services."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from yakbox.errors import ArtifactError


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    resolved = path.resolve()
    before = _file_signature(resolved)
    digest = _sha256_snapshot(str(resolved), *before)
    after = _file_signature(resolved)
    if after != before:
        digest = _sha256_snapshot(str(resolved), *after)
    return digest


@lru_cache(maxsize=4_096)
def _sha256_snapshot(
    path: str,
    _size: int,
    _mtime_ns: int,
    _ctime_ns: int,
    _inode: int,
) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(path: Path) -> tuple[int, int, int, int]:
    status = path.stat()
    return (
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
        status.st_ino,
    )


def atomic_write_bytes(path: Path, data: bytes, *, overwrite: bool = False) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ArtifactError(f"Output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() and not overwrite:
            raise ArtifactError(f"Output already exists: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    path: Path, value: Mapping[str, object], *, overwrite: bool = True
) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    atomic_write_bytes(path, f"{payload}\n".encode(), overwrite=overwrite)


@contextmanager
def atomic_output_path(path: Path, *, overwrite: bool = False) -> Iterator[Path]:
    """Yield a sibling temporary path and atomically commit it on success."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        if not temporary.is_file():
            raise ArtifactError(f"Output writer did not create a file: {temporary}")
        commit_temporary_file(temporary, path, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def commit_temporary_file(
    temporary: Path,
    destination: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Durably commit an already-written sibling temporary file."""

    temporary = temporary.resolve()
    destination = destination.resolve()
    if temporary.parent != destination.parent:
        raise ArtifactError("Atomic commit requires a sibling temporary file")
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise ArtifactError(f"Output writer produced no data: {temporary}")
    # Windows requires a writable file descriptor for fsync().  Reopening the
    # completed artifact read-only works on POSIX but fails with EBADF on
    # Windows before the atomic replace can happen.
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    if overwrite:
        temporary.replace(destination)
    else:
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ArtifactError(f"Output already exists: {destination}") from error
        except OSError as error:
            raise ArtifactError(
                f"Filesystem cannot safely commit output: {destination}"
            ) from error
        temporary.unlink()
    _sync_directory(destination.parent)


def safe_child(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ArtifactError(f"Path escapes managed root {resolved_root}: {candidate}")
    return resolved


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
