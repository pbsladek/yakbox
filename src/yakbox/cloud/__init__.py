"""Typed Resemble.ai client and hosted batch services."""

from yakbox.cloud.batch import (
    BatchReport,
    BatchResult,
    BatchStatus,
    ProgressCallback,
    run_cloud_batch,
)
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
from yakbox.textutils import BatchRow

__all__ = [
    "AmbiguousMutationError",
    "AudioFormat",
    "BatchJournalError",
    "BatchReport",
    "BatchResult",
    "BatchRow",
    "BatchStatus",
    "ClientOptions",
    "ClientStateError",
    "CloudError",
    "FileSynthesisResult",
    "HostedBudgetExceeded",
    "HostedUsageGate",
    "Page",
    "Precision",
    "ProgressCallback",
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
    "run_cloud_batch",
]
