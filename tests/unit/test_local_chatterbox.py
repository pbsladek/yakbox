from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from yakbox.local import LocalChatterboxService
from yakbox.speech import (
    AudioFormat,
    ChatterboxSynthesisOptions,
    SpeechSynthesisRequest,
    SpeechTransformationRequest,
)


class _Model:
    sr = 24_000

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def generate(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return object()


class _Factory:
    model = _Model()
    loads = 0

    @classmethod
    def from_pretrained(cls, *, device: str) -> _Model:
        assert device == "cpu"
        cls.loads += 1
        return cls.model


class _AudioWriter:
    fail = False

    @classmethod
    def save(cls, path: str, _waveform: object, sample_rate: int) -> None:
        assert sample_rate == 24_000
        Path(path).write_bytes(b"tiny-wav")
        if cls.fail:
            raise RuntimeError("simulated writer failure")


class _Torch:
    seeds: ClassVar[list[int]] = []

    @classmethod
    def manual_seed(cls, seed: int) -> None:
        cls.seeds.append(seed)


@pytest.fixture(autouse=True)
def _reset_fakes() -> None:
    _Factory.model = _Model()
    _Factory.loads = 0
    _AudioWriter.fail = False
    _Torch.seeds = []


@pytest.fixture
def mocked_chatterbox(monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.import_module

    def load(name: str, package: str | None = None) -> object:
        if name == "torchaudio":
            return _AudioWriter
        if name == "chatterbox.tts":
            return SimpleNamespace(ChatterboxTTS=_Factory)
        if name == "chatterbox.vc":
            return SimpleNamespace(ChatterboxVC=_Factory)
        if name == "torch":
            return _Torch
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", load)


@pytest.mark.asyncio
async def test_short_local_tts_is_typed_cached_and_atomic(
    tmp_path: Path, mocked_chatterbox: None
) -> None:
    service = LocalChatterboxService(device="cpu")
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")

    first = await service.synthesize_to_file(
        SpeechSynthesisRequest(
            text="Hi.",
            voice="narrator",
            backend="chatterbox-local",
            output_format=AudioFormat.WAV,
            reference_audio=reference,
        ),
        tmp_path / "first.wav",
    )
    await service.synthesize_to_file(
        SpeechSynthesisRequest(text="Yo.", voice="narrator"),
        tmp_path / "second.wav",
    )

    assert first.path.read_bytes() == b"tiny-wav"
    assert first.sample_rate == 24_000
    assert _Factory.loads == 1
    assert _Factory.model.calls[0] == (
        ("Hi.",),
        {"audio_prompt_path": str(reference)},
    )


@pytest.mark.asyncio
async def test_short_local_voice_conversion_uses_reference(
    tmp_path: Path, mocked_chatterbox: None
) -> None:
    source = tmp_path / "source.wav"
    reference = tmp_path / "target.wav"
    source.write_bytes(b"source")
    reference.write_bytes(b"target")

    artifact = await LocalChatterboxService(device="cpu").transform_to_file(
        SpeechTransformationRequest(
            input_path=source,
            voice="character",
            reference_audio=reference,
        ),
        tmp_path / "converted.wav",
    )

    assert artifact.path.read_bytes() == b"tiny-wav"
    assert _Factory.model.calls == [
        ((), {"audio": str(source), "target_voice_path": str(reference)})
    ]


@pytest.mark.asyncio
async def test_short_local_tts_applies_typed_chatterbox_controls(
    tmp_path: Path,
    mocked_chatterbox: None,
) -> None:
    await LocalChatterboxService(device="cpu").synthesize_to_file(
        SpeechSynthesisRequest(
            text="Hi.",
            voice="narrator",
            chatterbox=ChatterboxSynthesisOptions(
                cfg_weight=0.4,
                exaggeration=0.7,
                seed=42,
            ),
        ),
        tmp_path / "controlled.wav",
    )

    assert _Factory.model.calls == [
        (("Hi.",), {"cfg_weight": 0.4, "exaggeration": 0.7})
    ]
    assert _Torch.seeds == [42]


@pytest.mark.asyncio
async def test_local_writer_failure_leaves_no_partial_destination(
    tmp_path: Path, mocked_chatterbox: None
) -> None:
    destination = tmp_path / "failed.wav"
    _AudioWriter.fail = True

    with pytest.raises(RuntimeError, match="writer failure"):
        await LocalChatterboxService(device="cpu").synthesize_to_file(
            SpeechSynthesisRequest(text="Hi.", voice="narrator"),
            destination,
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob("*.part"))
