"""Typed speech-service boundary shared by direct and audiobook workflows."""

from yakbox.speech.capabilities import BackendCapabilities
from yakbox.speech.guardrails import (
    HostedWorkEstimate,
    estimate_hosted_work,
    hosted_confirmation_reasons,
    validate_hosted_preflight,
)
from yakbox.speech.models import (
    AudioFormat,
    ChatterboxSynthesisOptions,
    CurrencyCode,
    HostedUsageBudget,
    HostedUsageSnapshot,
    PricingSourceId,
    SpeechArtifact,
    SpeechSynthesisRequest,
    SpeechTransformationRequest,
)
from yakbox.speech.services import (
    BatchTextToSpeechService,
    FakeSpeechService,
    HostedUsageJournalingService,
    HostedUsageRecorder,
    HostedUsageReportingService,
    SpeechTransformationService,
    TextToSpeechService,
    open_speech_backend,
    open_transformation_backend,
)

__all__ = [
    "AudioFormat",
    "BackendCapabilities",
    "BatchTextToSpeechService",
    "ChatterboxSynthesisOptions",
    "CurrencyCode",
    "FakeSpeechService",
    "HostedUsageBudget",
    "HostedUsageJournalingService",
    "HostedUsageRecorder",
    "HostedUsageReportingService",
    "HostedUsageSnapshot",
    "HostedWorkEstimate",
    "PricingSourceId",
    "SpeechArtifact",
    "SpeechSynthesisRequest",
    "SpeechTransformationRequest",
    "SpeechTransformationService",
    "TextToSpeechService",
    "estimate_hosted_work",
    "hosted_confirmation_reasons",
    "open_speech_backend",
    "open_transformation_backend",
    "validate_hosted_preflight",
]
