from __future__ import annotations

from pathlib import Path

import pytest

from yakbox._files import atomic_output_path, commit_temporary_file


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
