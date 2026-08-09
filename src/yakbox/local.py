"""Lazy local Chatterbox adapter.

The adapter intentionally blocks in its async method when used directly. Audiobook
builds isolate it in a worker process; GPU calls are never sent to a thread.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import tempfile
import wave
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from yakbox._files import atomic_output_path, sha256_file
from yakbox.errors import ArtifactError, BackendUnavailableError, ValidationError
from yakbox.speech.capabilities import BackendCapabilities
from yakbox.speech.chunking import CHATTERBOX_CHUNK_CHARACTERS, plan_text_chunks
from yakbox.speech.models import (
    AudioFormat,
    ChatterboxSynthesisOptions,
    SpeechArtifact,
    SpeechSynthesisRequest,
    SpeechTransformationRequest,
)
from yakbox.speech.waves import WavJoinPart, concatenate_wavs

_CHATTERBOX_GENERATION_LIMIT_SECONDS = 38.0


class _AudioWriter(Protocol):
    def save(self, path: str, waveform: object, sample_rate: int) -> None: ...


class _ChatterboxModel(Protocol):
    sr: int

    def generate(self, *args: object, **kwargs: object) -> object: ...

    def prepare_conditionals(self, audio_path: str, *, exaggeration: float) -> None: ...


class _ChatterboxFactory(Protocol):
    @classmethod
    def from_pretrained(cls, *, device: str) -> _ChatterboxModel: ...


class _ChatterboxTTSModule(Protocol):
    ChatterboxTTS: _ChatterboxFactory


class _ChatterboxVCModule(Protocol):
    ChatterboxVC: _ChatterboxFactory


class _TorchSeed(Protocol):
    def manual_seed(self, seed: int) -> object: ...


class _DeviceAvailability(Protocol):
    def is_available(self) -> bool: ...


class _TorchBackends(Protocol):
    mps: _DeviceAvailability


class _TorchRuntime(_TorchSeed, Protocol):
    cuda: _DeviceAvailability
    backends: _TorchBackends


class _PerthPkgResourcesCompat(ModuleType):
    """Supply Perth's one legacy resource lookup during its import."""

    def resource_filename(self, package: str, resource: str) -> str:
        module = sys.modules.get(package)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise ImportError(f"Cannot locate package resources for {package}")
        return str(Path(module_file).resolve().parent / resource)


class LocalChatterboxService:
    capabilities = BackendCapabilities(
        name="chatterbox-local",
        synthesis=True,
        transformation=True,
        streaming=False,
        hosted=False,
        output_formats=("wav",),
        supports_reference_voice=True,
    )

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device
        self._tts_model: _ChatterboxModel | None = None
        self._vc_model: _ChatterboxModel | None = None
        self._tts_reference_key: tuple[str, float] | None = None

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        return self._synthesize_sync(request, destination, overwrite=overwrite)

    async def transform_to_file(
        self,
        request: SpeechTransformationRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        return self._transform_sync(request, destination, overwrite=overwrite)

    def _synthesize_sync(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool,
    ) -> SpeechArtifact:
        chunks = plan_text_chunks(request.text, CHATTERBOX_CHUNK_CHARACTERS)
        if len(chunks) == 1:
            options = request.chatterbox
            if options is None or options.seed is None:
                options = _direct_chunk_options(request, 1, chunks[0].text)
            return self._synthesize_one(
                replace(request, chatterbox=options),
                destination,
                overwrite=overwrite,
            )
        if destination.exists() and not overwrite:
            raise ArtifactError(f"Output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".yakbox-chatterbox-", dir=destination.parent
        ) as temporary_dir:
            root = Path(temporary_dir)
            parts: list[WavJoinPart] = []
            for index, chunk in enumerate(chunks, start=1):
                path = root / f"chunk-{index:04d}.wav"
                options = _direct_chunk_options(request, index, chunk.text)
                self._synthesize_one(
                    replace(request, text=chunk.text, chatterbox=options),
                    path,
                    overwrite=False,
                )
                parts.append(WavJoinPart(path, chunk.boundary.value))
            concatenate_wavs(tuple(parts), destination, overwrite=overwrite)
        return SpeechArtifact(
            path=destination.resolve(),
            backend="chatterbox-local",
            voice=request.voice,
            output_format=AudioFormat.WAV,
            bytes_written=destination.stat().st_size,
            sha256=sha256_file(destination),
            duration_seconds=_wav_duration(destination),
            sample_rate=int(self._tts_model.sr) if self._tts_model else None,
        )

    def _synthesize_one(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool,
    ) -> SpeechArtifact:
        if request.output_format is not AudioFormat.WAV:
            raise ValidationError("Local Chatterbox currently writes WAV")
        try:
            torchaudio = cast(_AudioWriter, importlib.import_module("torchaudio"))
            tts_module = cast(
                _ChatterboxTTSModule,
                _import_chatterbox_module("chatterbox.tts"),
            )
            factory = tts_module.ChatterboxTTS
        except ImportError as error:
            raise BackendUnavailableError(
                "Local Chatterbox is not installed; install with: "
                'uv tool install "yakbox[local]"'
            ) from error
        if self._tts_model is None:
            self.device = _resolve_device(self.device)
            self._tts_model = factory.from_pretrained(device=self.device)
            self._tts_reference_key = None
        kwargs, exaggeration = _generation_controls(request)
        self._prepare_tts_reference(factory, request, exaggeration)
        waveform = self._tts_model.generate(request.text, **kwargs)
        sample_rate = int(self._tts_model.sr)
        with atomic_output_path(destination, overwrite=overwrite) as temporary:
            torchaudio.save(str(temporary), waveform, sample_rate)
            duration = _validate_synthesis_duration(temporary)
        return SpeechArtifact(
            path=destination.resolve(),
            backend="chatterbox-local",
            voice=request.voice,
            output_format=AudioFormat.WAV,
            bytes_written=destination.stat().st_size,
            sha256=sha256_file(destination),
            duration_seconds=duration,
            sample_rate=sample_rate,
        )

    def _prepare_tts_reference(
        self,
        factory: _ChatterboxFactory,
        request: SpeechSynthesisRequest,
        exaggeration: float,
    ) -> None:
        if self._tts_model is None:
            raise BackendUnavailableError("Local Chatterbox model is not loaded")
        if request.reference_audio is not None:
            reference_key = (sha256_file(request.reference_audio), exaggeration)
            if reference_key != self._tts_reference_key:
                self._tts_model.prepare_conditionals(
                    str(request.reference_audio),
                    exaggeration=exaggeration,
                )
                self._tts_reference_key = reference_key
        elif self._tts_reference_key is not None:
            self._tts_model = factory.from_pretrained(device=self.device)
            self._tts_reference_key = None

    def _transform_sync(
        self,
        request: SpeechTransformationRequest,
        destination: Path,
        *,
        overwrite: bool,
    ) -> SpeechArtifact:
        try:
            torchaudio = cast(_AudioWriter, importlib.import_module("torchaudio"))
            vc_module = cast(
                _ChatterboxVCModule,
                _import_chatterbox_module("chatterbox.vc"),
            )
            factory = vc_module.ChatterboxVC
        except ImportError as error:
            raise BackendUnavailableError(
                "Local Chatterbox is not installed; install with: "
                'uv tool install "yakbox[local]"'
            ) from error
        if self._vc_model is None:
            self.device = _resolve_device(self.device)
            self._vc_model = factory.from_pretrained(device=self.device)
        kwargs: dict[str, object] = {}
        if request.reference_audio is not None:
            kwargs["target_voice_path"] = str(request.reference_audio)
        waveform = self._vc_model.generate(
            audio=str(request.input_path),
            **kwargs,
        )
        sample_rate = int(self._vc_model.sr)
        with atomic_output_path(destination, overwrite=overwrite) as temporary:
            torchaudio.save(str(temporary), waveform, sample_rate)
        return SpeechArtifact(
            path=destination.resolve(),
            backend="chatterbox-local",
            voice=request.voice,
            output_format=AudioFormat.WAV,
            bytes_written=destination.stat().st_size,
            sha256=sha256_file(destination),
            sample_rate=sample_rate,
        )


def _generation_controls(
    request: SpeechSynthesisRequest,
) -> tuple[dict[str, object], float]:
    options = request.chatterbox
    if options is None:
        return {}, 0.5
    kwargs: dict[str, object] = {}
    if options.cfg_weight is not None:
        kwargs["cfg_weight"] = options.cfg_weight
    exaggeration = options.exaggeration if options.exaggeration is not None else 0.5
    if options.exaggeration is not None:
        kwargs["exaggeration"] = exaggeration
    if options.seed is not None:
        torch = cast(_TorchSeed, importlib.import_module("torch"))
        torch.manual_seed(options.seed)
    return kwargs, exaggeration


def _import_chatterbox_module(name: str) -> object:
    """Import Chatterbox while bridging Perth's removed resource API."""

    if (
        sys.modules.get("pkg_resources") is not None
        or importlib.util.find_spec("pkg_resources") is not None
    ):
        return importlib.import_module(name)
    previous = sys.modules.get("pkg_resources")
    sys.modules["pkg_resources"] = _PerthPkgResourcesCompat("pkg_resources")
    try:
        return importlib.import_module(name)
    finally:
        if previous is None:
            del sys.modules["pkg_resources"]
        else:
            sys.modules["pkg_resources"] = previous


def _resolve_device(device: str) -> str:
    """Resolve Yakbox's optional ``auto`` value to an explicit torch device."""

    normalized = device.strip().casefold()
    if not normalized:
        raise ValidationError("Local Chatterbox device must not be empty")
    if normalized != "auto":
        return normalized
    try:
        torch = cast(_TorchRuntime, importlib.import_module("torch"))
    except ImportError as error:
        raise BackendUnavailableError(
            "Local Chatterbox is not installed; install with: "
            'uv tool install "yakbox[local]"'
        ) from error
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _direct_chunk_options(
    request: SpeechSynthesisRequest, chunk_index: int, text: str
) -> ChatterboxSynthesisOptions:
    options = request.chatterbox or ChatterboxSynthesisOptions()
    base_seed = options.seed if options.seed is not None else 0
    material = f"{base_seed}:{chunk_index}:{text}".encode()
    seed = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
    return replace(options, seed=seed)


def _validate_synthesis_duration(path: Path) -> float:
    duration = _wav_duration(path)
    if duration >= _CHATTERBOX_GENERATION_LIMIT_SECONDS:
        raise ArtifactError(
            "Local Chatterbox output approached its generation limit; "
            "reduce the synthesis chunk size"
        )
    return duration


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            sample_rate = audio.getframerate()
    except (OSError, EOFError, wave.Error) as error:
        raise ArtifactError("Local Chatterbox produced an unreadable WAV") from error
    if frames < 1 or sample_rate < 1:
        raise ArtifactError("Local Chatterbox produced an empty WAV")
    return frames / sample_rate
