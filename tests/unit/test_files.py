from __future__ import annotations

from pathlib import Path

from yakbox._files import commit_temporary_file


def test_commit_reopens_completed_file_writable_for_windows_fsync(
    tmp_path: Path,
    monkeypatch,
) -> None:
    temporary = tmp_path / ".audio.wav.part"
    destination = tmp_path / "audio.wav"
    temporary.write_bytes(b"audio")
    modes: list[str] = []
    real_open = Path.open

    def recording_open(path: Path, mode: str = "r", *args, **kwargs):
        if path == temporary.resolve():
            modes.append(mode)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)

    commit_temporary_file(temporary, destination)

    assert modes == ["rb+"]
    assert destination.read_bytes() == b"audio"
