from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from yakbox.errors import ArtifactError

_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def validate_output_name(name: str) -> str:
    if (
        not name
        or "\x00" in name
        or Path(name).is_absolute()
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
    ):
        raise ArtifactError(f"Unsafe output name: {name!r}")
    if Path(name).stem.casefold() in _RESERVED:
        raise ArtifactError(f"Reserved output name: {name!r}")
    return name


def slugify(value: str) -> str:
    slug = re.sub(r"[^\w]+", "-", value.casefold(), flags=re.UNICODE).strip("-")
    return slug[:80] or "speech"


def atomic_commit_bytes(
    destination: Path, data: bytes, *, overwrite: bool = False
) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        commit_file(temporary, destination, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def commit_file(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if overwrite:
        temporary.replace(destination)
    else:
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ArtifactError(f"Output already exists: {destination}") from error
        except OSError as error:
            raise ArtifactError(
                f"Filesystem cannot safely commit without overwrite: {destination}"
            ) from error
        temporary.unlink()
    _sync_directory(destination.parent)


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
