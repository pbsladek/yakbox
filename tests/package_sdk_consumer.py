"""Built-wheel import and typing smoke test for the public SDK."""

from __future__ import annotations

import importlib
from pathlib import Path

from yakbox import ValidationError, YakboxError
from yakbox.audiobook import (
    AudiobookManifest,
    BuildProgress,
    BuildRequest,
    BuildResult,
    load_manifest,
    run_audiobook_build,
)
from yakbox.cloud import BatchReport, BatchRow, ResembleClient, SynthesisRequest
from yakbox.diagnostics import DoctorReport, run_doctor
from yakbox.speech import (
    AudioFormat,
    SpeechBackendOptions,
    SpeechSynthesisRequest,
    TextToSpeechService,
    open_configured_speech_backend,
)

PUBLIC_MODULES = (
    "yakbox",
    "yakbox.audio",
    "yakbox.audiobook",
    "yakbox.cloud",
    "yakbox.diagnostics",
    "yakbox.speech",
)


def progress(event: BuildProgress) -> None:
    _ = (event.event.value, event.stage.value)


async def consume_build(manifest: AudiobookManifest) -> BuildResult:
    return await run_audiobook_build(
        manifest,
        BuildRequest(dry_run=True),
        progress=progress,
    )


async def consume_speech() -> None:
    async with open_configured_speech_backend(
        "fake",
        SpeechBackendOptions(),
    ) as service:
        await service.synthesize_to_file(
            SpeechSynthesisRequest(
                text="typed consumer",
                voice="narrator",
                output_format=AudioFormat.WAV,
            ),
            Path("consumer.wav"),
        )


def type_surface(
    service: TextToSpeechService,
    batch: BatchReport,
    doctor: DoctorReport,
) -> tuple[int, int]:
    del service, doctor
    return batch.ok, batch.failed


def import_surface() -> None:
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            getattr(module, name)


def references() -> tuple[object, ...]:
    return (
        ValidationError,
        YakboxError,
        load_manifest,
        BatchRow,
        ResembleClient,
        SynthesisRequest,
        run_doctor,
    )


if __name__ == "__main__":
    import_surface()
