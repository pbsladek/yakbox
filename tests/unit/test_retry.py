from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from yakbox.cloud.errors import ProviderError, RetryExhaustedError
from yakbox.cloud.models import RetryPolicy
from yakbox.cloud.rate_limit import RateLimitGate
from yakbox.cloud.retry import parse_retry_after, retry_operation


@pytest.mark.asyncio
async def test_retry_succeeds_and_uses_exponential_delay() -> None:
    attempts = 0
    delays: list[float] = []

    async def operation(_attempt: int) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ProviderError(503, "busy")
        return "ok"

    async def sleep(delay: float) -> None:
        delays.append(delay)

    result = await retry_operation(
        operation,
        policy=RetryPolicy(max_attempts=4, base_delay=0.5),
        sleep=sleep,
        random_source=lambda: 0,
    )

    assert result == "ok"
    assert attempts == 3
    assert delays == [0.5, 1.0]


@pytest.mark.asyncio
async def test_retry_gives_up() -> None:
    async def operation(_attempt: int) -> None:
        raise ProviderError(503, "busy")

    with pytest.raises(RetryExhaustedError) as raised:
        await retry_operation(
            operation,
            policy=RetryPolicy(max_attempts=2, base_delay=0),
            sleep=_no_sleep,
            random_source=lambda: 0,
        )
    assert raised.value.attempts == 2
    assert raised.value.status_code == 503


def test_retry_after_delta_and_date() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    date = (now + timedelta(seconds=7)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after("2") == 2
    assert parse_retry_after(date, now=now) == 7
    assert parse_retry_after("nonsense") is None
    assert parse_retry_after("-5") == 0


@pytest.mark.asyncio
async def test_retry_transport_backoff_cap_and_cancellation() -> None:
    attempts = 0
    delays: list[float] = []

    async def operation(_attempt: int) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("offline")
        return "ok"

    async def sleep(delay: float) -> None:
        delays.append(delay)

    assert (
        await retry_operation(
            operation,
            policy=RetryPolicy(
                max_attempts=3,
                base_delay=10,
                max_backoff=3,
            ),
            sleep=sleep,
            random_source=lambda: 1,
        )
        == "ok"
    )
    assert delays == [3, 3]

    cancelled_attempts = 0

    async def cancelled(_attempt: int) -> None:
        nonlocal cancelled_attempts
        cancelled_attempts += 1
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await retry_operation(cancelled, policy=RetryPolicy())
    assert cancelled_attempts == 1


@pytest.mark.asyncio
async def test_non_retryable_4xx_fails_immediately() -> None:
    attempts = 0

    async def operation(_attempt: int) -> None:
        nonlocal attempts
        attempts += 1
        raise ProviderError(400, "invalid")

    with pytest.raises(ProviderError):
        await retry_operation(operation, policy=RetryPolicy(max_attempts=4))
    assert attempts == 1


@pytest.mark.asyncio
async def test_rate_limit_deadline_cannot_be_shortened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = 100.0
    sleeps: list[float] = []

    def monotonic() -> float:
        return clock

    async def sleep(delay: float) -> None:
        nonlocal clock
        sleeps.append(delay)
        clock += delay

    monkeypatch.setattr("yakbox.cloud.rate_limit.time.monotonic", monotonic)
    monkeypatch.setattr("yakbox.cloud.rate_limit.asyncio.sleep", sleep)
    gate = RateLimitGate()
    await gate.extend(8)
    await gate.extend(2)
    await gate.wait()

    assert sleeps == [8]


async def _no_sleep(_delay: float) -> None:
    return None
