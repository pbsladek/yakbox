from __future__ import annotations

from pathlib import Path

from yakbox._files import sha256_file
from yakbox.cloud.client import ResembleClient
from yakbox.cloud.models import AudioFormat as CloudAudioFormat
from yakbox.cloud.models import Precision, SynthesisRequest
from yakbox.speech.capabilities import BackendCapabilities
from yakbox.speech.models import (
    AudioFormat,
    HostedUsageSnapshot,
    SpeechArtifact,
    SpeechSynthesisRequest,
)
from yakbox.speech.services import HostedUsageRecorder


class ResembleSpeechService:
    capabilities = BackendCapabilities(
        name="resemble",
        synthesis=True,
        transformation=False,
        streaming=True,
        hosted=True,
        output_formats=("wav", "mp3"),
        max_text_characters=3_000,
        supports_hd=True,
    )

    def __init__(self, client: ResembleClient) -> None:
        self.client = client

    async def usage_snapshot(self) -> HostedUsageSnapshot | None:
        return await self.client.usage_snapshot()

    def set_usage_recorder(self, recorder: HostedUsageRecorder | None) -> None:
        self.client.set_usage_recorder(recorder)

    async def restore_usage(self, snapshot: HostedUsageSnapshot) -> None:
        await self.client.restore_usage(snapshot)

    async def synthesize_to_file(
        self,
        request: SpeechSynthesisRequest,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> SpeechArtifact:
        provider_request = SynthesisRequest(
            text=request.text,
            voice_uuid=request.voice,
            project_uuid=request.project,
            title=request.title,
            output_format=CloudAudioFormat(request.output_format.value),
            sample_rate=request.sample_rate,
            use_hd=request.use_hd,
            precision=Precision(request.precision or Precision.PCM_32.value),
            apply_custom_pronunciations=request.apply_custom_pronunciations,
        )
        result = await self.client.synthesize_to_file(
            provider_request, destination, overwrite=overwrite
        )
        return SpeechArtifact(
            path=result.path,
            backend="resemble",
            voice=request.voice,
            output_format=AudioFormat(request.output_format.value),
            bytes_written=result.bytes_written,
            sha256=sha256_file(result.path),
            duration_seconds=result.duration_seconds,
            sample_rate=request.sample_rate,
            request_id=result.request_id,
            attempts=result.attempts,
        )
