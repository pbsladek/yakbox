"""Typed Resemble.ai client and hosted batch services."""

from yakbox.cloud.client import ResembleClient
from yakbox.cloud.errors import (
    AmbiguousMutationError,
    BatchJournalError,
    ClientStateError,
    CloudError,
    HostedBudgetExceeded,
    ProviderError,
    ProviderProtocolError,
    ResumeMismatchError,
    RetryExhaustedError,
)
from yakbox.cloud.models import (
    AudioFormat,
    ClientOptions,
    FileSynthesisResult,
    Page,
    Precision,
    Project,
    Recording,
    RetryPolicy,
    StreamRequest,
    SynthesisRequest,
    SynthesisResult,
    Voice,
)
from yakbox.cloud.service import ResembleSpeechService
from yakbox.cloud.usage import HostedUsageGate

__all__ = [
    "AmbiguousMutationError",
    "AudioFormat",
    "BatchJournalError",
    "ClientOptions",
    "ClientStateError",
    "CloudError",
    "FileSynthesisResult",
    "HostedBudgetExceeded",
    "HostedUsageGate",
    "Page",
    "Precision",
    "Project",
    "ProviderError",
    "ProviderProtocolError",
    "Recording",
    "ResembleClient",
    "ResembleSpeechService",
    "ResumeMismatchError",
    "RetryExhaustedError",
    "RetryPolicy",
    "StreamRequest",
    "SynthesisRequest",
    "SynthesisResult",
    "Voice",
]
