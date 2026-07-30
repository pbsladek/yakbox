from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from yakbox.audio.assemble import assemble_m4b
from yakbox.audio.inspect import AudioInspection, inspect_audio
from yakbox.audio.master import master_wav
from yakbox.errors import ArtifactError


def test_inspection_translates_timeout_and_malformed_provider_output(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "chapter.wav"
    audio.write_bytes(b"RIFF-short")

    with (
        patch("yakbox.audio.inspect.shutil.which", return_value="/ffprobe"),
        patch(
            "yakbox.audio.inspect.subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffprobe", 60),
        ),
        pytest.raises(ArtifactError, match="timed out"),
    ):
        inspect_audio(audio)

    malformed = subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=0,
        stdout='{"streams": []}',
        stderr="",
    )
    with (
        patch("yakbox.audio.inspect.shutil.which", return_value="/ffprobe"),
        patch("yakbox.audio.inspect.subprocess.run", return_value=malformed),
        pytest.raises(ArtifactError, match="Invalid FFprobe response"),
    ):
        inspect_audio(audio)


def test_mastering_rejects_missing_output_and_cleans_temporary_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "mastered.wav"
    source.write_bytes(b"RIFF-short")
    success_without_output = subprocess.CompletedProcess(
        args=["ffmpeg"],
        returncode=0,
        stdout="",
        stderr="",
    )

    with (
        patch("yakbox.audio.master.shutil.which", return_value="/ffmpeg"),
        patch(
            "yakbox.audio.master.subprocess.run",
            return_value=success_without_output,
        ),
        pytest.raises(ArtifactError, match="produced no output"),
    ):
        master_wav(source, destination)

    assert not destination.exists()
    assert tuple(tmp_path.glob(".mastered.*")) == ()


def test_failed_assembly_cleans_all_temporary_files(tmp_path: Path) -> None:
    chapter = tmp_path / "chapter.mp3"
    destination = tmp_path / "book.m4b"
    chapter.write_bytes(b"short")
    inspection = AudioInspection(
        schema_version=1,
        path=chapter,
        format_name="mp3",
        codec="mp3",
        duration_seconds=0.1,
        sample_rate=44_100,
        channels=1,
        bit_rate=192_000,
        size=chapter.stat().st_size,
        valid=True,
        issues=(),
    )
    failed = subprocess.CompletedProcess(
        args=["ffmpeg"],
        returncode=1,
        stdout="",
        stderr="failed",
    )

    with (
        patch("yakbox.audio.assemble.shutil.which", return_value="/ffmpeg"),
        patch("yakbox.audio.assemble.inspect_audio", return_value=inspection),
        patch("yakbox.audio.assemble.subprocess.run", return_value=failed),
        pytest.raises(ArtifactError, match="assembly failed"),
    ):
        assemble_m4b((chapter,), destination, title="Tiny book")

    assert not destination.exists()
    assert tuple(tmp_path.glob(".yakbox-concat-*")) == ()
    assert tuple(tmp_path.glob(".book.*")) == ()
