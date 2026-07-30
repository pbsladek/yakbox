"""Provider-neutral hosted-work estimates and preflight guardrails."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from yakbox.errors import ValidationError
from yakbox.speech.models import HostedUsageBudget


@dataclass(frozen=True, slots=True)
class HostedWorkEstimate:
    """Conservative usage range for hosted synthesis work.

    The minimum assumes one successful provider attempt per logical item. The
    maximum assumes every configured attempt submits the complete item. It is
    deliberately an estimate rather than a bill.
    """

    logical_items: int
    logical_characters: int
    max_attempts: int
    minimum_provider_requests: int
    maximum_provider_requests: int
    minimum_submitted_characters: int
    maximum_submitted_characters: int
    minimum_estimated_spend: Decimal | None = None
    maximum_estimated_spend: Decimal | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_items": self.logical_items,
            "logical_characters": self.logical_characters,
            "max_attempts": self.max_attempts,
            "minimum_provider_requests": self.minimum_provider_requests,
            "maximum_provider_requests": self.maximum_provider_requests,
            "minimum_submitted_characters": self.minimum_submitted_characters,
            "maximum_submitted_characters": self.maximum_submitted_characters,
            "minimum_estimated_spend": (
                str(self.minimum_estimated_spend)
                if self.minimum_estimated_spend is not None
                else None
            ),
            "maximum_estimated_spend": (
                str(self.maximum_estimated_spend)
                if self.maximum_estimated_spend is not None
                else None
            ),
            "basis": "conservative_preflight_estimate",
        }


def estimate_hosted_work(
    texts: Iterable[str],
    *,
    max_attempts: int = 4,
    price_per_character: Decimal | None = None,
) -> HostedWorkEstimate:
    if max_attempts < 1:
        raise ValidationError("max_attempts must be at least 1")
    if price_per_character is not None and (
        not price_per_character.is_finite() or price_per_character < 0
    ):
        raise ValidationError("price_per_character must be finite and non-negative")
    materialized = tuple(texts)
    characters = sum(len(text) for text in materialized)
    maximum_characters = characters * max_attempts
    return HostedWorkEstimate(
        logical_items=len(materialized),
        logical_characters=characters,
        max_attempts=max_attempts,
        minimum_provider_requests=len(materialized),
        maximum_provider_requests=len(materialized) * max_attempts,
        minimum_submitted_characters=characters,
        maximum_submitted_characters=maximum_characters,
        minimum_estimated_spend=(
            price_per_character * Decimal(characters)
            if price_per_character is not None
            else None
        ),
        maximum_estimated_spend=(
            price_per_character * Decimal(maximum_characters)
            if price_per_character is not None
            else None
        ),
    )


def validate_hosted_preflight(
    budget: HostedUsageBudget,
    estimate: HostedWorkEstimate,
) -> None:
    """Reject a run that cannot complete even with no retries."""

    if (
        budget.max_provider_requests is not None
        and estimate.minimum_provider_requests > budget.max_provider_requests
    ):
        raise ValidationError(
            "Hosted request budget cannot cover the planned work: "
            f"{estimate.minimum_provider_requests} request(s) are required but the "
            f"limit is {budget.max_provider_requests}"
        )
    if (
        budget.max_submitted_characters is not None
        and estimate.minimum_submitted_characters > budget.max_submitted_characters
    ):
        raise ValidationError(
            "Hosted character budget cannot cover the planned work: "
            f"{estimate.minimum_submitted_characters} character(s) are required but "
            f"the limit is {budget.max_submitted_characters}"
        )
    if (
        budget.max_estimated_spend is not None
        and estimate.minimum_estimated_spend is not None
        and estimate.minimum_estimated_spend > budget.max_estimated_spend
    ):
        raise ValidationError(
            "Hosted spending budget cannot cover the planned work: "
            f"at least {estimate.minimum_estimated_spend} {budget.currency} is "
            f"estimated but the limit is {budget.max_estimated_spend}"
        )


def hosted_confirmation_reasons(
    budget: HostedUsageBudget,
    estimate: HostedWorkEstimate,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if (
        budget.confirm_above_characters is not None
        and estimate.logical_characters > budget.confirm_above_characters
    ):
        reasons.append(
            f"{estimate.logical_characters} logical character(s) exceed the "
            f"{budget.confirm_above_characters} confirmation threshold"
        )
    if (
        budget.confirm_above_requests is not None
        and estimate.logical_items > budget.confirm_above_requests
    ):
        reasons.append(
            f"{estimate.logical_items} logical request(s) exceed the "
            f"{budget.confirm_above_requests} confirmation threshold"
        )
    return tuple(reasons)
