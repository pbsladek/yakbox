from __future__ import annotations

import importlib
import json
import math
import struct
import subprocess
import wave
import zlib
from collections.abc import Callable
from pathlib import Path

import pytest

from yakbox.audio.crop import wav_duration_seconds
from yakbox.audio.inspect import AudioQualityPolicy, inspect_audio
from yakbox.audio.master import encode_mp3, master_wav
from yakbox.audio.processes import run_command
from yakbox.audiobook.build import _normalize_synthesis_chunks
from yakbox.audiobook.planner import BuildStage, PlanNode
from yakbox.errors import ArtifactError
from yakbox.speech.canonical_audio import CanonicalAudioPreparer

inspection_module = importlib.import_module("yakbox.audio.inspect")
module = importlib.import_module("yakbox.audiobook.build")


def write_pcm(path: Path, *, seconds: float = 1, rate: int = 24_000) -> None:
    samples = [
        round(4000 * math.sin(2 * math.pi * 440 * index / rate))
        for index in range(round(rate * seconds))
    ]
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def test_duration_reads_at_most_two_frames_and_detects_truncation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "audio.wav"
    write_pcm(audio, seconds=5)
    original = wave.Wave_read.readframes
    counts: list[int] = []

    def read(reader: wave.Wave_read, count: int) -> bytes:
        counts.append(count)
        assert count <= 2
        return original(reader, count)

    monkeypatch.setattr(wave.Wave_read, "readframes", read)
    assert wav_duration_seconds(audio) == 5
    assert counts == [2]
    with audio.open("r+b") as stream:
        stream.truncate(audio.stat().st_size - 2)
    with pytest.raises(ArtifactError, match="truncated"):
        wav_duration_seconds(audio)


def test_duration_rejects_empty_and_partial_pcm_frames(tmp_path: Path) -> None:
    audio = tmp_path / "empty.wav"
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24_000)
        writer.writeframes(b"")
    with pytest.raises(ArtifactError, match="empty"):
        wav_duration_seconds(audio)
    payload = bytearray(audio.read_bytes())
    payload[40:44] = struct.pack("<I", 3)
    audio.write_bytes(payload + b"\x01\x00\x01")
    with pytest.raises(ArtifactError, match="invalid frame"):
        wav_duration_seconds(audio)


def test_sub_quarter_second_silence_limits_are_enforced(tmp_path: Path) -> None:
    audio = tmp_path / "edges.wav"
    write_pcm(audio)
    with wave.open(str(audio), "rb") as reader:
        content = reader.readframes(reader.getnframes())
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24_000)
        writer.writeframes(b"\0" * 9600 + content + b"\0" * 9600)
    value = inspect_audio(
        audio,
        quality=AudioQualityPolicy(
            maximum_leading_silence_seconds=0.1,
            maximum_trailing_silence_seconds=0.1,
        ),
    )
    assert not value.valid
    assert value.leading_silence_seconds == pytest.approx(0.2, abs=0.001)
    assert value.trailing_silence_seconds == pytest.approx(0.2, abs=0.001)
    assert len(value.issues) == 2


def test_policy_changes_reuse_probe_and_signal_measurements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "cache.wav"
    write_pcm(audio)
    original = inspection_module.run_command
    commands: list[str] = []

    def run(
        command: list[str],
        *,
        timeout: float,
        stderr_line: Callable[[str], None] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command[0])
        return original(command, timeout=timeout, stderr_line=stderr_line)

    monkeypatch.setattr(inspection_module, "run_command", run)
    plain = inspect_audio(audio)
    measured = inspect_audio(audio, quality=AudioQualityPolicy())
    strict = inspect_audio(audio, quality=AudioQualityPolicy(minimum_loudness_lufs=-5))
    assert plain.integrated_loudness_lufs is None
    assert measured.valid and not strict.valid
    assert measured.integrated_loudness_lufs == strict.integrated_loudness_lufs
    assert commands == ["ffprobe", "ffmpeg"]
    write_pcm(audio, seconds=2)
    assert inspect_audio(audio, quality=AudioQualityPolicy()).duration_seconds == 2
    assert commands == ["ffprobe", "ffmpeg", "ffprobe", "ffmpeg"]


def test_quality_and_mastering_select_the_probed_audio_stream(tmp_path: Path) -> None:
    audio = tmp_path / "two-tracks.m4b"
    result = run_command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=24000:duration=3",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=24000:cl=stereo",
            "-t",
            "3",
            "-map",
            "0:a:0",
            "-map",
            "1:a:0",
            "-disposition:a:0",
            "0",
            "-disposition:a:1",
            "default",
            "-c:a",
            "aac",
            str(audio),
        ],
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    value = inspect_audio(audio, quality=AudioQualityPolicy(minimum_loudness_lufs=-30))
    assert value.channels == 1
    assert value.valid
    output = tmp_path / "master.wav"
    master_wav(audio, output, normalize=False)
    assert inspect_audio(
        output,
        quality=AudioQualityPolicy(
            minimum_loudness_lufs=-30,
        ),
    ).valid
    prepared = CanonicalAudioPreparer(tmp_path / "canonical").prepare(audio)
    assert inspect_audio(
        prepared.path,
        quality=AudioQualityPolicy(
            minimum_loudness_lufs=-30,
        ),
    ).valid


def test_normalization_only_converts_incompatible_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [tmp_path / f"chunk-{index}.wav" for index in range(3)]
    for path, rate, width in zip(paths, (16000, 16000, 24000), (3, 3, 2), strict=True):
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(width)
            writer.setframerate(rate)
            writer.writeframes(b"\0" * rate * width)
    before = [path.read_bytes() for path in paths[:2]]
    original = module.master_wav
    converted: list[Path] = []

    def convert(
        source: Path,
        destination: Path,
        *,
        sample_rate: int,
        normalize: bool,
        overwrite: bool,
    ) -> None:
        converted.append(source)
        original(
            source,
            destination,
            sample_rate=sample_rate,
            normalize=normalize,
            overwrite=overwrite,
        )

    monkeypatch.setattr(module, "master_wav", convert)
    node = PlanNode(
        id="one:synthesize",
        stage=BuildStage.SYNTHESIZE,
        chapter_id="one",
        fingerprint="a" * 64,
        dependencies=(),
        output=tmp_path / "chapter.wav",
        chunks=("a", "b", "c"),
    )
    assert _normalize_synthesis_chunks(node, paths, preferred_sample_rate=16000) == (
        1,
        3,
        16000,
    )
    assert converted == paths[2:]
    assert [path.read_bytes() for path in paths[:2]] == before


def test_mp3_cover_maps_one_audio_stream_and_one_attached_picture(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio.wav"
    write_pcm(audio)
    cover = tmp_path / "cover.png"

    def chunk(kind: bytes, value: bytes) -> bytes:
        return (
            struct.pack(">I", len(value))
            + kind
            + value
            + struct.pack(">I", zlib.crc32(kind + value))
        )

    # Generate a red RGB fixture; no third-party image asset is needed.
    cover.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 16) * 16))
        + chunk(b"IEND", b"")
    )
    delivery = tmp_path / "delivery.mp3"
    encode_mp3(audio, delivery, cover=cover)
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,codec_type",
            "-of",
            "json",
            str(delivery),
        ],
        timeout=10,
    )
    streams = json.loads(result.stdout)["streams"]
    assert sorted(item["codec_type"] for item in streams) == ["audio", "video"]
    assert (
        next(item for item in streams if item["codec_type"] == "video")["codec_name"]
        == "png"
    )
