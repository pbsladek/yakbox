from __future__ import annotations

import asyncio
from decimal import Decimal

from yakbox.cloud.errors import HostedBudgetExceeded
from yakbox.speech.models import HostedUsageBudget, HostedUsageSnapshot
from yakbox.speech.services import HostedUsageRecorder


class HostedUsageGate:
    """Atomically reserves hosted request usage across concurrent workers."""

    def __init__(
        self,
        budget: HostedUsageBudget,
        *,
        price_per_character: Decimal | None = None,
    ) -> None:
        if budget.max_estimated_spend is not None and price_per_character is None:
            raise HostedBudgetExceeded(
                "A monetary cap requires an explicit, versioned pricing rate"
            )
        self.budget = budget
        self.price_per_character = price_per_character
        self._lock = asyncio.Lock()
        self._logical_items = 0
        self._attempts = 0
        self._characters = 0
        self._ambiguous = 0
        self._recorder: HostedUsageRecorder | None = None

    def set_recorder(self, recorder: HostedUsageRecorder | None) -> None:
        """Install the command-scoped durable reservation recorder."""

        self._recorder = recorder

    async def add_logical_item(self) -> None:
        """Record one logical requested item before provider attempts."""
        async with self._lock:
            self._logical_items += 1

    async def reserve_attempt(self, characters: int) -> None:
        """Reserve one provider attempt and its submitted characters durably."""
        async with self._lock:
            next_attempts = self._attempts + 1
            next_characters = self._characters + characters
            if (
                self.budget.max_provider_requests is not None
                and next_attempts > self.budget.max_provider_requests
            ):
                raise HostedBudgetExceeded(
                    "Hosted request limit reached "
                    f"({self.budget.max_provider_requests})"
                )
            if (
                self.budget.max_submitted_characters is not None
                and next_characters > self.budget.max_submitted_characters
            ):
                raise HostedBudgetExceeded(
                    "Hosted submitted-character limit reached "
                    f"({self.budget.max_submitted_characters})"
                )
            estimate = self._estimate(next_characters)
            if (
                self.budget.max_estimated_spend is not None
                and estimate is not None
                and estimate > self.budget.max_estimated_spend
            ):
                raise HostedBudgetExceeded(
                    f"Hosted estimated-spend limit reached "
                    f"({self.budget.max_estimated_spend} {self.budget.currency})"
                )
            snapshot = HostedUsageSnapshot(
                logical_items=self._logical_items,
                provider_attempts=next_attempts,
                submitted_characters=next_characters,
                estimated_spend=estimate,
                currency=self.budget.currency,
                ambiguous_attempts=self._ambiguous,
            )
            if self._recorder is not None:
                # The durable record must exist before the caller can send.
                # If recording fails, the provider attempt is not authorized.
                await self._recorder(snapshot, characters)
            self._attempts = next_attempts
            self._characters = next_characters

    async def mark_ambiguous(self) -> None:
        """Mark the latest attempt as potentially accepted or billed."""
        async with self._lock:
            self._ambiguous += 1

    async def restore_prior_usage(
        self,
        *,
        logical_items: int,
        provider_attempts: int,
        submitted_characters: int,
        ambiguous_attempts: int = 0,
    ) -> None:
        """Restore journaled counters without allowing a resume to reset limits."""
        if (
            min(
                logical_items,
                provider_attempts,
                submitted_characters,
                ambiguous_attempts,
            )
            < 0
        ):
            raise HostedBudgetExceeded("Hosted usage counters cannot be negative")
        async with self._lock:
            if self._attempts or self._characters or self._logical_items:
                raise HostedBudgetExceeded("Hosted usage has already been initialized")
            if (
                self.budget.max_provider_requests is not None
                and provider_attempts > self.budget.max_provider_requests
            ):
                raise HostedBudgetExceeded(
                    "Resumed hosted usage already exceeds the request limit"
                )
            if (
                self.budget.max_submitted_characters is not None
                and submitted_characters > self.budget.max_submitted_characters
            ):
                raise HostedBudgetExceeded(
                    "Resumed hosted usage already exceeds the character limit"
                )
            estimate = self._estimate(submitted_characters)
            if (
                self.budget.max_estimated_spend is not None
                and estimate is not None
                and estimate > self.budget.max_estimated_spend
            ):
                raise HostedBudgetExceeded(
                    "Resumed hosted usage already exceeds the spending limit"
                )
            self._logical_items = logical_items
            self._attempts = provider_attempts
            self._characters = submitted_characters
            self._ambiguous = ambiguous_attempts

    async def snapshot(self) -> HostedUsageSnapshot:
        """Return an immutable snapshot of current usage counters."""
        async with self._lock:
            return HostedUsageSnapshot(
                logical_items=self._logical_items,
                provider_attempts=self._attempts,
                submitted_characters=self._characters,
                estimated_spend=self._estimate(self._characters),
                currency=self.budget.currency,
                ambiguous_attempts=self._ambiguous,
            )

    def _estimate(self, characters: int) -> Decimal | None:
        if self.price_per_character is None:
            return None
        return self.price_per_character * Decimal(characters)
