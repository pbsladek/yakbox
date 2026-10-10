from __future__ import annotations

import os
from pathlib import Path

import pytest

from yakbox._files import atomic_output_path, atomic_write_bytes, commit_temporary_file
from yakbox.errors import ArtifactError


def test_atomic_bytes_preserve_a_destination_created_during_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "evidence.json"
    real_replace = Path.replace
    real_link = os.link

    def racing_replace(source: Path, target: Path) -> Path:
        if target == destination:
            target.write_bytes(b"other writer")
        return real_replace(source, target)

    def racing_link(source: Path, target: Path) -> None:
        if target == destination:
            target.write_bytes(b"other writer")
        real_link(source, target)

    monkeypatch.setattr(Path, "replace", racing_replace)
    monkeypatch.setattr(os, "link", racing_link)

    with pytest.raises(ArtifactError, match="already exists"):
        atomic_write_bytes(destination, b"this writer")

    assert destination.read_bytes() == b"other writer"
    assert tuple(tmp_path.glob("*.part")) == ()


@pytest.mark.parametrize("payload", [b"", b"new bytes"])
def test_atomic_bytes_preserve_empty_output_and_explicit_overwrite(
    tmp_path: Path, payload: bytes
) -> None:
    destination = tmp_path / "output"
    atomic_write_bytes(destination, b"old bytes")
    atomic_write_bytes(destination, payload, overwrite=True)
    assert destination.read_bytes() == payload


def test_atomic_output_preserves_media_suffix(tmp_path: Path) -> None:
    destination = tmp_path / "audio.wav"

    with atomic_output_path(destination) as temporary:
        assert temporary.name.endswith(".part.wav")
        temporary.write_bytes(b"audio")

    assert destination.read_bytes() == b"audio"


def test_commit_reopens_completed_file_writable_for_windows_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = tmp_path / ".audio.wav.part"
    destination = tmp_path / "audio.wav"
    temporary.write_bytes(b"audio")
    modes: list[str] = []
    real_open = Path.open

    def recording_open(
        path: Path,
        mode: str = "r",
        *,
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> object:
        if path == temporary.resolve():
            modes.append(mode)
        return real_open(
            path,
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "open", recording_open)

    commit_temporary_file(temporary, destination)

    assert modes == ["rb+"]
    assert destination.read_bytes() == b"audio"
