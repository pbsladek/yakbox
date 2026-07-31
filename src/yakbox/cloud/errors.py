from __future__ import annotations

from dataclasses import dataclass

from yakbox.errors import YakboxError


class CloudError(YakboxError):
    """Base class for hosted-provider errors."""

    code = "cloud_error"


class ClientStateError(CloudError):
    """The HTTP client is not open."""

    code = "client_state_error"


@dataclass(frozen=True, slots=True)
class ProviderError(CloudError):
    """A bounded provider response error safe to report to callers."""

    code = "provider_error"
    status_code: int
    message: str
    request_id: str | None = None
    retry_after: float | None = None
    ambiguous: bool = False

    def __str__(self) -> str:
        request = f" (request {self.request_id})" if self.request_id else ""
        return f"Provider returned HTTP {self.status_code}{request}: {self.message}"


class ProviderProtocolError(CloudError):
    """The provider returned a malformed or oversized response."""

    code = "provider_protocol_error"


class AmbiguousMutationError(CloudError):
    """A management mutation may have reached the provider."""

    code = "ambiguous_mutation"

    def __init__(self, operation: str, *, cause: Exception) -> None:
        self.operation = operation
        self.cause = cause
        super().__init__(
            f"{operation} may have been accepted by the provider; verify provider "
            "state before retrying manually"
        )


class RetryExhaustedError(CloudError):
    """All retry attempts were consumed, retaining safe terminal evidence."""

    code = "retry_exhausted"

    def __init__(self, *, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        if isinstance(last_error, ProviderError):
            self.status_code: int | None = last_error.status_code
            self.request_id: str | None = last_error.request_id
            detail = str(last_error)
        else:
            self.status_code = None
            self.request_id = None
            detail = type(last_error).__name__
        super().__init__(
            f"Operation failed after {attempts} attempts; last error: {detail}"
        )


class HostedBudgetExceeded(CloudError):
    """A hosted usage limit would be exceeded."""

    code = "hosted_budget_exceeded"


class BatchJournalError(CloudError):
    """The durable cloud batch journal is unavailable or inconsistent."""

    code = "batch_journal_error"


class ResumeMismatchError(CloudError):
    """A batch resume input or configuration differs from its journal."""

    code = "resume_mismatch"
