from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from yakbox.errors import ValidationError


class AudioFormat(StrEnum):
    WAV = "wav"
    MP3 = "mp3"


class Precision(StrEnum):
    MULAW = "MULAW"
    PCM_16 = "PCM_16"
    PCM_24 = "PCM_24"
    PCM_32 = "PCM_32"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_backoff: float = 8.0
    max_retry_after: float = 60.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValidationError("max_attempts must be at least 1")
        if min(self.base_delay, self.max_backoff, self.max_retry_after) < 0:
            raise ValidationError("Retry delays cannot be negative")


@dataclass(frozen=True, slots=True)
class ClientOptions:
    management_base_url: str = "https://app.resemble.ai/api/v2"
    synthesis_base_url: str = "https://f.cluster.resemble.ai"
    connect_timeout: float = 10.0
    read_timeout: float = 120.0
    write_timeout: float = 30.0
    pool_timeout: float = 10.0
    max_connections: int = 20
    max_keepalive_connections: int = 20
    max_json_response_bytes: int = 64 * 1024 * 1024
    retry: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    text: str
    voice_uuid: str
    project_uuid: str | None = None
    title: str | None = None
    precision: Precision = Precision.PCM_32
    output_format: AudioFormat = AudioFormat.WAV
    sample_rate: int | None = None
    use_hd: bool = False
    apply_custom_pronunciations: bool = False

    def __post_init__(self) -> None:
        _validate_synthesis(self.text, self.voice_uuid, self.sample_rate, 3_000)


@dataclass(frozen=True, slots=True)
class StreamRequest:
    text: str
    voice_uuid: str
    project_uuid: str | None = None
    precision: Precision = Precision.PCM_32
    sample_rate: int | None = None
    use_hd: bool = False
    apply_custom_pronunciations: bool = False

    def __post_init__(self) -> None:
        _validate_synthesis(self.text, self.voice_uuid, self.sample_rate, 2_000)


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    audio: bytes
    duration_seconds: float | None
    synthesis_seconds: float | None
    output_format: AudioFormat
    sample_rate: int | None
    title: str | None
    issues: tuple[str, ...]
    request_id: str | None
    attempts: int


@dataclass(frozen=True, slots=True)
class FileSynthesisResult:
    path: Path
    bytes_written: int
    duration_seconds: float | None
    issues: tuple[str, ...]
    request_id: str | None
    attempts: int


@dataclass(frozen=True, slots=True)
class Voice:
    uuid: str
    name: str
    status: str | None = None
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class Project:
    uuid: str
    name: str
    description: str | None = None
    is_collaborative: bool = False
    is_archived: bool = False
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class Recording:
    uuid: str
    name: str
    text: str | None = None
    status: str | None = None


@dataclass(frozen=True, slots=True)
class Page[T]:
    items: tuple[T, ...]
    page: int | None
    page_count: int | None
    total_results: int | None


def _validate_synthesis(
    text: str, voice_uuid: str, sample_rate: int | None, maximum: int
) -> None:
    if not text.strip():
        raise ValidationError("Synthesis text must not be empty")
    if len(text) > maximum:
        raise ValidationError(
            f"Synthesis text exceeds the provider limit of {maximum} characters"
        )
    if not voice_uuid.strip():
        raise ValidationError("voice_uuid must not be empty")
    if sample_rate is not None and sample_rate <= 0:
        raise ValidationError("sample_rate must be positive")
