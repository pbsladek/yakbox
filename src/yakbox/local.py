"""Lazy local Chatterbox adapter.

The adapter intentionally blocks in its async method when used directly. Audiobook
builds isolate it in a worker process; GPU calls are never sent to a thread.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol, cast

from yakbox._files import atomic_output_path, sha256_file
from yakbox.errors import BackendUnavailableError, ValidationError
from yakbox.speech.capabilities import BackendCapabilities
from yakbox.speech.models import (
    AudioFormat,
    SpeechArtifact,
    SpeechSynthesisRequest,
    SpeechTransformationRequest,
)


class _AudioWriter(Protocol):
    def save(self, path: str, waveform: object, sample_rate: int) -> None: ...


class _ChatterboxModel(Protocol):
    sr: int

    def generate(self, *args: object, **kwargs: object) -> object: ...


class _ChatterboxFactory(Protocol):
    @classmethod
    def from_pretrained(cls, *, device: str) -> _ChatterboxModel: ...


class _TorchSeed(Protocol):
    def manual_seed(self, seed: int) -> object: ...


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

    def __init__(self, *, device: str = "auto") -> None:
        self.device = device
        self._tts_model: _ChatterboxModel | None = None
        self._vc_model: _ChatterboxModel | None = None

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
        if request.output_format is not AudioFormat.WAV:
            raise ValidationError("Local Chatterbox currently writes WAV")
        try:
            torchaudio = cast(_AudioWriter, importlib.import_module("torchaudio"))
            tts_module = importlib.import_module("chatterbox.tts")
            factory = cast(_ChatterboxFactory, tts_module.ChatterboxTTS)
        except ImportError as error:
            raise BackendUnavailableError(
                "Local Chatterbox is not installed; install with: "
                'uv tool install "yakbox[local]"'
            ) from error
        if self._tts_model is None:
            self._tts_model = factory.from_pretrained(device=self.device)
        kwargs: dict[str, object] = {}
        if request.reference_audio is not None:
            kwargs["audio_prompt_path"] = str(request.reference_audio)
        if request.chatterbox is not None:
            if request.chatterbox.cfg_weight is not None:
                kwargs["cfg_weight"] = request.chatterbox.cfg_weight
            if request.chatterbox.exaggeration is not None:
                kwargs["exaggeration"] = request.chatterbox.exaggeration
            if request.chatterbox.seed is not None:
                torch = cast(_TorchSeed, importlib.import_module("torch"))
                torch.manual_seed(request.chatterbox.seed)
        waveform = self._tts_model.generate(request.text, **kwargs)
        sample_rate = int(self._tts_model.sr)
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

    def _transform_sync(
        self,
        request: SpeechTransformationRequest,
        destination: Path,
        *,
        overwrite: bool,
    ) -> SpeechArtifact:
        try:
            torchaudio = cast(_AudioWriter, importlib.import_module("torchaudio"))
            vc_module = importlib.import_module("chatterbox.vc")
            factory = cast(_ChatterboxFactory, vc_module.ChatterboxVC)
        except ImportError as error:
            raise BackendUnavailableError(
                "Local Chatterbox is not installed; install with: "
                'uv tool install "yakbox[local]"'
            ) from error
        if self._vc_model is None:
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
