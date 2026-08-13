"""Unit tests for `credit_risk.services.risk_service`."""

import pytest

from credit_risk.services.risk_service import calculate_risk_score


def test_risk_score_probability_at_zero_returns_low() -> None:
    result = calculate_risk_score(0.0)

    assert result.value == 0
    assert result.level == "LOW"


def test_risk_score_probability_at_thirty_percent_returns_low() -> None:
    result = calculate_risk_score(0.30)

    assert result.value == 30
    assert result.level == "LOW"


def test_risk_score_probability_at_thirty_one_percent_returns_medium() -> None:
    result = calculate_risk_score(0.31)

    assert result.value == 31
    assert result.level == "MEDIUM"


def test_risk_score_probability_above_seventy_one_percent_returns_high() -> None:
    result = calculate_risk_score(0.73)

    assert result.value == 73
    assert result.level == "HIGH"


def test_risk_score_probability_at_one_returns_high() -> None:
    result = calculate_risk_score(1.0)

    assert result.value == 100
    assert result.level == "HIGH"


@pytest.mark.parametrize("invalid_probability", [-0.01, 1.01, -5.0, 2.0])
def test_risk_score_probability_out_of_range_raises_value_error(
    invalid_probability: float,
) -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        calculate_risk_score(invalid_probability)
