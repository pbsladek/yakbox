from __future__ import annotations

import importlib
import sys
import wave
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar, Protocol, cast

import pytest

from yakbox.errors import ArtifactError
from yakbox.local import LocalChatterboxService, _import_chatterbox_module
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
        self.prepare_calls: list[tuple[str, float]] = []
        self.conds: object = object()

    def generate(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return object()

    def prepare_conditionals(self, path: str, *, exaggeration: float) -> None:
        self.prepare_calls.append((path, exaggeration))
        self.conds = (path, exaggeration)


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
    frames = 2_400

    @classmethod
    def save(cls, path: str, _waveform: object, sample_rate: int) -> None:
        assert sample_rate == 24_000
        assert Path(path).suffix == ".wav"
        with wave.open(path, "wb") as writer:
            writer.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
            writer.writeframes(b"\0\0" * cls.frames)
        if cls.fail:
            raise RuntimeError("simulated writer failure")


class _Torch:
    seeds: ClassVar[list[int]] = []
    cuda = SimpleNamespace(is_available=lambda: False)
    backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))

    @classmethod
    def manual_seed(cls, seed: int) -> None:
        cls.seeds.append(seed)


class _ResourceCompat(Protocol):
    def resource_filename(self, package: str, resource: str) -> str: ...


class _ImportedTTS(Protocol):
    ChatterboxTTS: type[_Factory]


@pytest.fixture(autouse=True)
def _reset_fakes() -> None:
    _Factory.model = _Model()
    _Factory.loads = 0
    _AudioWriter.fail = False
    _AudioWriter.frames = 2_400
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
    _ = mocked_chatterbox
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

    assert first.path.stat().st_size > 44
    assert first.sample_rate == 24_000
    assert _Factory.loads == 2
    assert _Factory.model.calls[0] == (
        ("Hi.",),
        {},
    )
    assert _Factory.model.prepare_calls == [(str(reference), 0.5)]


@pytest.mark.asyncio
async def test_short_local_voice_conversion_uses_reference(
    tmp_path: Path, mocked_chatterbox: None
) -> None:
    _ = mocked_chatterbox
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

    assert artifact.path.stat().st_size > 44
    assert _Factory.model.calls == [
        ((), {"audio": str(source), "target_voice_path": str(reference)})
    ]


@pytest.mark.asyncio
async def test_short_local_tts_applies_typed_chatterbox_controls(
    tmp_path: Path,
    mocked_chatterbox: None,
) -> None:
    _ = mocked_chatterbox
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
async def test_auto_device_resolves_to_cpu_when_accelerators_are_unavailable(
    tmp_path: Path, mocked_chatterbox: None
) -> None:
    _ = mocked_chatterbox

    await LocalChatterboxService(device="auto").synthesize_to_file(
        SpeechSynthesisRequest(text="Hi.", voice="narrator"),
        tmp_path / "auto.wav",
    )

    assert _Factory.loads == 1


@pytest.mark.asyncio
async def test_local_writer_failure_leaves_no_partial_destination(
    tmp_path: Path, mocked_chatterbox: None
) -> None:
    _ = mocked_chatterbox
    destination = tmp_path / "failed.wav"
    _AudioWriter.fail = True

    with pytest.raises(RuntimeError, match="writer failure"):
        await LocalChatterboxService(device="cpu").synthesize_to_file(
            SpeechSynthesisRequest(text="Hi.", voice="narrator"),
            destination,
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob("*part*"))


@pytest.mark.asyncio
async def test_generation_limit_output_is_rejected_before_commit(
    tmp_path: Path, mocked_chatterbox: None
) -> None:
    _ = mocked_chatterbox
    destination = tmp_path / "truncated.wav"
    _AudioWriter.frames = 24_000 * 38

    with pytest.raises(ArtifactError, match="generation limit"):
        await LocalChatterboxService(device="cpu").synthesize_to_file(
            SpeechSynthesisRequest(text="Long text.", voice="narrator"),
            destination,
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob("*part*"))


@pytest.mark.asyncio
async def test_reference_conditioning_is_prepared_once_per_voice(
    tmp_path: Path, mocked_chatterbox: None
) -> None:
    _ = mocked_chatterbox
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")
    service = LocalChatterboxService(device="cpu")
    for index in range(2):
        await service.synthesize_to_file(
            SpeechSynthesisRequest(
                text=f"Line {index}.",
                voice="narrator",
                reference_audio=reference,
            ),
            tmp_path / f"line-{index}.wav",
        )

    assert _Factory.loads == 1
    assert _Factory.model.prepare_calls == [(str(reference), 0.5)]


@pytest.mark.asyncio
async def test_reference_conditioning_lru_reuses_voice_after_switch(
    tmp_path: Path, mocked_chatterbox: None
) -> None:
    _ = mocked_chatterbox
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"first-reference")
    second.write_bytes(b"second-reference")
    service = LocalChatterboxService(device="cpu", conditioning_cache_size=2)

    for index, reference in enumerate((first, second, first), start=1):
        await service.synthesize_to_file(
            SpeechSynthesisRequest(
                text=f"Line {index}.",
                voice=f"voice-{index}",
                reference_audio=reference,
            ),
            tmp_path / f"line-{index}.wav",
        )

    assert _Factory.loads == 1
    assert _Factory.model.prepare_calls == [
        (str(first), 0.5),
        (str(second), 0.5),
    ]
    assert service.model_loaded
    assert service.conditioning_cache_entries == 2


@pytest.mark.asyncio
async def test_long_local_text_uses_semantic_chunks(
    tmp_path: Path, mocked_chatterbox: None
) -> None:
    _ = mocked_chatterbox
    text = "First sentence. " + "word " * 120 + "Final sentence."

    artifact = await LocalChatterboxService(device="cpu").synthesize_to_file(
        SpeechSynthesisRequest(text=text, voice="narrator"),
        tmp_path / "long.wav",
    )

    assert artifact.path.is_file()
    assert len(_Factory.model.calls) >= 2
    assert all(len(cast(str, call[0][0])) <= 500 for call in _Factory.model.calls)


def test_perth_resource_compat_is_narrow_and_removed_after_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = ModuleType("perth.perth_net")
    package.__file__ = str(tmp_path / "perth" / "perth_net" / "__init__.py")
    monkeypatch.setitem(sys.modules, "perth.perth_net", package)
    monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)
    original = importlib.import_module

    def load(name: str, package: str | None = None) -> object:
        if name != "chatterbox.tts":
            return original(name, package)
        compat = cast(_ResourceCompat, sys.modules["pkg_resources"])
        assert compat.resource_filename("perth.perth_net", "pretrained") == str(
            tmp_path / "perth" / "perth_net" / "pretrained"
        )
        return SimpleNamespace(ChatterboxTTS=_Factory)

    monkeypatch.setattr(importlib_util, "find_spec", lambda _name: None)
    monkeypatch.setattr(importlib, "import_module", load)

    imported = cast(_ImportedTTS, _import_chatterbox_module("chatterbox.tts"))

    assert imported.ChatterboxTTS is _Factory
    assert "pkg_resources" not in sys.modules
