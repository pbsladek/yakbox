from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from yakbox.errors import ArtifactError, BuildError
from yakbox.playback import play_audio


@pytest.mark.parametrize("platform,opener", [("darwin", "open"), ("linux", "xdg-open")])
def test_playback_passes_an_absolute_path_as_one_argument_without_a_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    opener: str,
) -> None:
    audio = tmp_path / 'a sample & $(echo unsafe) "é".wav'
    audio.write_bytes(b"generated audio")
    monkeypatch.setattr("yakbox.playback.sys.platform", platform)
    calls: list[tuple[str, ...]] = []

    def launch(
        arguments: tuple[str, ...],
        *,
        check: bool,
        stdout: int,
        stderr: int,
        timeout: int,
    ) -> None:
        assert check
        assert stdout == stderr == subprocess.DEVNULL
        assert timeout == 15
        calls.append(arguments)

    monkeypatch.setattr("yakbox.playback.subprocess.run", launch)
    play_audio(audio)
    assert calls == [(opener, str(audio.resolve()))]
    assert audio.read_bytes() == b"generated audio"


@pytest.mark.parametrize(
    "failure",
    [
        OSError("missing player"),
        subprocess.CalledProcessError(1, "player", stderr=b"unbounded diagnostics"),
        subprocess.TimeoutExpired("player", 15),
    ],
)
def test_playback_failure_preserves_audio_and_reports_its_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    audio = tmp_path / "preview.wav"
    audio.write_bytes(b"generated audio")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("yakbox.playback.subprocess.run", fail)
    with pytest.raises(BuildError, match="Generated audio is available") as error:
        play_audio(audio)
    assert str(audio) in str(error.value)
    assert "unbounded diagnostics" not in str(error.value)
    assert audio.read_bytes() == b"generated audio"


@pytest.mark.parametrize("exists", [False, True])
def test_playback_rejects_missing_audio_and_directories(
    tmp_path: Path, exists: bool
) -> None:
    path = tmp_path / "audio"
    if exists:
        path.mkdir()
    with pytest.raises(ArtifactError):
        play_audio(path)
