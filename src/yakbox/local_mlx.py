"""Lazy MLX Chatterbox Turbo bridge; model execution stays in its owning process."""

from __future__ import annotations

import importlib
import os
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, cast

from yakbox.errors import BackendUnavailableError, BuildError


class _Generation(Protocol):
    audio: object
    sample_rate: int


class _Model(Protocol):
    _conds: object
    sample_rate: int

    def prepare_conditionals(self, path: str, *, exaggeration: float) -> None: ...
    def generate(self, text: str, **options: object) -> Iterator[_Generation]: ...


class _Loader(Protocol):
    def load_model(self, path: Path) -> _Model: ...


class _Writer(Protocol):
    def sf_write(self, file: str, data: object, samplerate: int) -> None: ...


class _Download(Protocol):
    def __call__(self, repo_id: str, filename: str, **options: object) -> str: ...


class _Hub(Protocol):
    hf_hub_download: _Download


@contextmanager
def _local_downloads(directory: Path) -> Generator[None]:
    """Resolve the port's auxiliary tokenizer lookup from the verified snapshot."""
    hub = cast(_Hub, importlib.import_module("huggingface_hub"))
    original = hub.hf_hub_download
    flags = {
        name: os.environ.get(name)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }

    def local_file(repo_id: str, filename: str, **_options: object) -> str:
        if repo_id != "mlx-community/S3TokenizerV2" or filename != "model.safetensors":
            raise BackendUnavailableError(
                "Model loading requires a complete local snapshot"
            )
        return str(directory / "s3tokenizer" / "model.safetensors")

    try:
        hub.hf_hub_download = local_file
        os.environ.update(dict.fromkeys(flags, "1"))
        yield
    finally:
        hub.hf_hub_download = original
        for name, value in flags.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class MlxTurboModel:
    """Present the explicit local MLX variant through the local synthesis contract."""

    def __init__(self, directory: Path) -> None:
        try:
            loader = cast(_Loader, importlib.import_module("mlx_audio.tts.utils"))
            with _local_downloads(directory):
                self.model = loader.load_model(directory)
        except ImportError as error:
            raise BackendUnavailableError(
                "MLX Turbo requires the analysis-qwen extra on Apple Silicon"
            ) from error
        self.sr = self.model.sample_rate

    @property
    def conds(self) -> object:
        return getattr(self.model, "_conds", None)

    @conds.setter
    def conds(self, value: object) -> None:
        self.model._conds = value

    def prepare_conditionals(self, audio_path: str, *, exaggeration: float) -> None:
        self.model.prepare_conditionals(audio_path, exaggeration=exaggeration)

    def generate(self, *args: object, **options: object) -> object:
        if len(args) != 1 or not isinstance(args[0], str):
            raise BuildError("MLX Turbo requires one text input")
        text = args[0]
        results = tuple(
            self.model.generate(text, split_pattern=None, stream=False, **options)
        )
        if len(results) != 1:
            raise BuildError("MLX Turbo returned an unexpected audio segment count")
        return results[0].audio


class MlxAudioWriter:
    """Write MLX waveform arrays through the upstream audio I/O implementation."""

    def save(self, path: str, waveform: object, sample_rate: int) -> None:
        module = cast(_Writer, importlib.import_module("mlx_audio.audio_io"))
        module.sf_write(path, waveform, sample_rate)
