from __future__ import annotations

import asyncio
import math
import struct
import subprocess
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import yakbox.local_phoneme_alignment as phonemes
from yakbox.errors import BackendUnavailableError, BuildError


def _wave(path: Path, *, channels: int = 1, width: int = 2) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(16_000)
        writer.writeframes(b"\0" * 16_000 * channels * width)


@pytest.mark.asyncio
@pytest.mark.parametrize("accelerated", [False, True])
async def test_phoneme_adapter_aligns_with_offline_models_and_reuses_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accelerated: bool
) -> None:
    audio = tmp_path / "speech.wav"
    _wave(audio)
    (tmp_path / "vocab.json").write_text('{"<pad>":0,"a":1,"b":2}')
    model = Mock()
    model.config.pad_token_id = 0
    model.to.return_value = model
    model.return_value.logits = [object()]
    processor = Mock(
        return_value=SimpleNamespace(
            input_values=SimpleNamespace(to=lambda _device: object())
        )
    )
    loader = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
        AutoModelForCTC=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )
    scores = [
        [-9.0, 0.0, -9.0],
        [-0.1, -9.0, -9.0],
        [-9.0, -9.0, 0.0],
        [-0.1, -9.0, -9.0],
    ]
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: accelerated)),
        inference_mode=nullcontext,
        log_softmax=lambda _logits, **_kwargs: SimpleNamespace(
            cpu=lambda: SimpleNamespace(tolist=lambda: scores)
        ),
    )
    monkeypatch.setattr(phonemes, "require_phoneme_model_path", lambda *_args: tmp_path)
    monkeypatch.setattr(
        phonemes.importlib,
        "import_module",
        lambda name: torch if name == "torch" else loader,
    )
    monkeypatch.setattr(phonemes, "_phonemize", lambda *_args, **_kwargs: "ab")
    aligner = phonemes.open_phoneme_aligner(
        "wav2vec2-ctc", model="reviewed", revision="pinned"
    )
    first = await aligner.align(audio, "ab", language="en")
    second = await aligner.align(audio, "ab", language="en")
    assert first == second
    assert [token.symbol for token in first.phonemes] == ["a", "b"]
    assert first.phonemes[0].end_seconds == 0.25
    assert first.phonemes[1].start_seconds == 0.5
    assert first.phonemes[0].confidence == 1
    assert first.fingerprint == aligner.fingerprint
    loader.AutoProcessor.from_pretrained.assert_called_once_with(
        str(tmp_path), local_files_only=True
    )
    loader.AutoModelForCTC.from_pretrained.assert_called_once_with(
        str(tmp_path), local_files_only=True
    )
    model.to.assert_called_once_with("mps" if accelerated else "cpu")
    model.eval.assert_called_once()


def test_missing_phoneme_runtime_is_a_domain_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(_name: str) -> None:
        raise ImportError("unavailable")

    monkeypatch.setattr(phonemes.importlib, "import_module", missing)
    aligner = phonemes.Wav2Vec2CtcPhonemeAligner(model="fixture", revision=None)
    with pytest.raises(BackendUnavailableError, match="extra phoneme"):
        aligner._load_runtime(tmp_path)
    with pytest.raises(BackendUnavailableError, match="Unsupported"):
        phonemes.open_phoneme_aligner("unsupported", model="fixture", revision=None)


@pytest.mark.asyncio
async def test_phoneme_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    async def stalled(*_args: object) -> None:
        await asyncio.Future()

    monkeypatch.setattr(phonemes.asyncio, "to_thread", stalled)
    aligner = phonemes.Wav2Vec2CtcPhonemeAligner(
        model="fixture", revision=None, timeout_seconds=0.01
    )
    with pytest.raises(BuildError, match="exceeded"):
        await aligner.align(Path("unused.wav"), "word", language="en")


def test_phonemizer_uses_fixed_argv_and_normalizes_ipa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock(
        return_value=subprocess.CompletedProcess(
            [],
            0,
            stdout=" \u02c8a\u02ccb\u200d ",
        )
    )
    monkeypatch.setattr(phonemes.subprocess, "run", run)
    assert (
        phonemes._phonemize(
            "a; $(unsafe)", language=" EN ", executable="espeak-ng", timeout_seconds=3
        )
        == "ab"
    )
    assert run.call_args.args[0] == [
        "espeak-ng",
        "-q",
        "--ipa",
        "-v",
        "en",
        "a; $(unsafe)",
    ]
    assert run.call_args.kwargs["timeout"] == 3
    assert "shell" not in run.call_args.kwargs


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError(),
        subprocess.CalledProcessError(1, "espeak"),
        subprocess.TimeoutExpired("espeak", 1),
    ],
)
def test_phonemizer_failures_are_translated(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    monkeypatch.setattr(phonemes.subprocess, "run", Mock(side_effect=failure))
    expected = (
        BackendUnavailableError
        if isinstance(failure, FileNotFoundError)
        else BuildError
    )
    with pytest.raises(expected):
        phonemes._phonemize(
            "hello", language="", executable="espeak-ng", timeout_seconds=1
        )


def test_phonemizer_refuses_empty_pronunciation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        phonemes.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 0, stdout="  ")),
    )
    with pytest.raises(BuildError, match="no expected phonemes"):
        phonemes._phonemize(
            "hello", language="en", executable="espeak-ng", timeout_seconds=1
        )


@pytest.mark.parametrize("content", ["not-json", "[]", '{"a":"invalid"}'])
def test_phoneme_model_vocabulary_is_validated(tmp_path: Path, content: str) -> None:
    (tmp_path / "vocab.json").write_text(content)
    with pytest.raises(BackendUnavailableError, match="vocabulary is invalid"):
        phonemes._load_vocabulary(tmp_path)


def test_phoneme_pcm_decoding_downmixes_channels(tmp_path: Path) -> None:
    audio = tmp_path / "stereo.wav"
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(8_000)
        writer.writeframes(struct.pack("<hhhh", -32768, 0, 32767, 32767))
    samples, rate = phonemes._read_pcm16_mono(audio)
    assert rate == 8000
    assert samples == [-0.5, 32767 / 32768]
    assert phonemes._resample([0, 1], 8000, 16000) == [0, 0.5, 1, 1]
    assert phonemes._resample(samples, rate, rate) is samples


def test_phoneme_pcm_rejects_missing_or_wrong_precision(tmp_path: Path) -> None:
    path = tmp_path / "speech.wav"
    with pytest.raises(BuildError, match="Cannot read"):
        phonemes._read_pcm16_mono(path)
    _wave(path, width=1)
    with pytest.raises(BuildError, match="16-bit PCM"):
        phonemes._read_pcm16_mono(path)


@pytest.mark.parametrize(
    ("samples", "source", "target"),
    [([], 8000, 16000), ([1.0], 0, 16000), ([1.0], 16000, 0)],
)
def test_phoneme_resampling_rejects_invalid_metadata(
    samples: list[float], source: int, target: int
) -> None:
    with pytest.raises(BuildError, match="invalid sample-rate"):
        phonemes._resample(samples, source, target)


@pytest.mark.parametrize(
    ("scores", "targets", "reason"),
    [
        ([], (1,), "requires emissions"),
        ([[0.0, 0.0]], (), "requires emissions"),
        ([[0.0, 0.0]], (1, 1), "too short"),
        ([[-math.inf, -math.inf]] * 2, (1,), "No valid CTC path"),
    ],
)
def test_ctc_alignment_rejects_impossible_paths(
    scores: list[list[float]], targets: tuple[int, ...], reason: str
) -> None:
    with pytest.raises(BuildError, match=reason):
        phonemes._ctc_forced_alignment(scores, targets, blank_id=0)
