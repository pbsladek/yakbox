from __future__ import annotations

import asyncio
import email.utils
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import httpx

from yakbox.cloud.errors import ProviderError, RetryExhaustedError
from yakbox.cloud.models import RetryPolicy
from yakbox.cloud.rate_limit import RateLimitGate

Sleep = Callable[[float], Awaitable[None]]

RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


async def retry_operation[T](
    operation: Callable[[int], Awaitable[T]],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[Exception], bool] | None = None,
    rate_limit_gate: RateLimitGate | None = None,
    sleep: Sleep = asyncio.sleep,
    random_source: Callable[[], float] = random.random,
) -> T:
    classify = is_retryable or default_retryable
    last_error: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        if rate_limit_gate is not None:
            await rate_limit_gate.wait()
        try:
            return await operation(attempt)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not classify(error):
                raise
            last_error = error
            if attempt == policy.max_attempts:
                break
            server_delay = (
                error.retry_after if isinstance(error, ProviderError) else None
            )
            if server_delay is not None:
                delay = min(policy.max_retry_after, server_delay)
                if rate_limit_gate is not None:
                    await rate_limit_gate.extend(delay)
                    continue
            else:
                delay = min(
                    policy.max_backoff,
                    policy.base_delay * 2 ** (attempt - 1) + random_source() * 0.25,
                )
            await sleep(delay)
    if last_error is None:
        raise RuntimeError("Retry loop ended without an error")
    raise RetryExhaustedError(
        attempts=policy.max_attempts,
        last_error=last_error,
    ) from last_error


def default_retryable(error: Exception) -> bool:
    if isinstance(error, httpx.TransportError):
        return True
    return isinstance(error, ProviderError) and error.status_code in RETRYABLE_STATUSES


def parse_retry_after(
    value: str | None, *, now: datetime | None = None
) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except TypeError, ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    return max(0.0, (parsed - current).total_seconds())
