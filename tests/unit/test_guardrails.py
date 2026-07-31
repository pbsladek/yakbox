from __future__ import annotations

from decimal import Decimal

import pytest

from yakbox.errors import ValidationError
from yakbox.speech import (
    CurrencyCode,
    HostedUsageBudget,
    PricingSourceId,
    estimate_hosted_work,
    hosted_confirmation_reasons,
    validate_hosted_preflight,
)


def test_hosted_work_estimate_reports_minimum_and_retry_maximum() -> None:
    estimate = estimate_hosted_work(
        ("abc", "defgh"),
        max_attempts=4,
        price_per_character=Decimal("0.01"),
    )

    assert estimate.logical_items == 2
    assert estimate.logical_characters == 8
    assert estimate.minimum_provider_requests == 2
    assert estimate.maximum_provider_requests == 8
    assert estimate.minimum_submitted_characters == 8
    assert estimate.maximum_submitted_characters == 32
    assert estimate.minimum_estimated_spend == Decimal("0.08")
    assert estimate.maximum_estimated_spend == Decimal("0.32")
    assert estimate.to_dict()["basis"] == "conservative_preflight_estimate"


@pytest.mark.parametrize(
    ("budget", "match"),
    [
        (
            HostedUsageBudget(max_provider_requests=1),
            "request budget cannot cover",
        ),
        (
            HostedUsageBudget(max_submitted_characters=7),
            "character budget cannot cover",
        ),
        (
            HostedUsageBudget(
                max_estimated_spend=Decimal("0.07"),
                currency=CurrencyCode("USD"),
                pricing_source=PricingSourceId("example-2026-07"),
            ),
            "spending budget cannot cover",
        ),
    ],
)
def test_hosted_preflight_rejects_impossible_minimum(
    budget: HostedUsageBudget,
    match: str,
) -> None:
    estimate = estimate_hosted_work(
        ("abc", "defgh"),
        price_per_character=Decimal("0.01"),
    )
    with pytest.raises(ValidationError, match=match):
        validate_hosted_preflight(budget, estimate)


def test_confirmation_thresholds_are_explicit_and_strictly_above() -> None:
    estimate = estimate_hosted_work(("abcd", "efgh"))
    exact = HostedUsageBudget(
        confirm_above_characters=8,
        confirm_above_requests=2,
    )
    exceeded = HostedUsageBudget(
        confirm_above_characters=7,
        confirm_above_requests=1,
    )

    assert hosted_confirmation_reasons(exact, estimate) == ()
    assert len(hosted_confirmation_reasons(exceeded, estimate)) == 2


@pytest.mark.parametrize("value", [Decimal("-1"), Decimal("NaN")])
def test_non_finite_or_negative_rates_are_rejected(value: Decimal) -> None:
    with pytest.raises(ValidationError):
        estimate_hosted_work(("text",), price_per_character=value)


def test_hosted_pricing_identifiers_normalize_at_the_boundary() -> None:
    assert CurrencyCode(" usd ") == "USD"
    assert PricingSourceId(" provider-price-list ") == "provider-price-list"


@pytest.mark.parametrize("value", ["", "US", "EURO", "U2D", "€UR"])
def test_invalid_currency_codes_are_rejected(value: str) -> None:
    with pytest.raises(ValidationError, match="three ASCII letters"):
        CurrencyCode(value)


def test_empty_pricing_source_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        PricingSourceId("  ")
