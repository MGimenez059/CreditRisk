"""Unit tests for `PredictionRequest` validation.

Covers SPECS.md §21 (API Validation) and the `test_invalid_income()`-style
examples in SPECS.md §27 (Testing Strategy).
"""

import pytest
from pydantic import ValidationError

from credit_risk.schemas.prediction import PredictionRequest

_VALID_PAYLOAD = {
    "age": 34,
    "income": 1_450_000,
    "employment_years": 6,
    "home_ownership": "RENT",
    "loan_amount": 500_000,
    "interest_rate": 12.5,
    "loan_intent": "PERSONAL",
    "credit_history_years": 7,
}


def test_prediction_request_accepts_a_valid_payload() -> None:
    request = PredictionRequest(**_VALID_PAYLOAD)

    assert request.age == 34
    assert request.home_ownership == "RENT"


def test_prediction_request_rejects_negative_income() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**_VALID_PAYLOAD, "income": -1})


def test_prediction_request_rejects_non_positive_loan_amount() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**_VALID_PAYLOAD, "loan_amount": 0})


def test_prediction_request_rejects_age_out_of_range() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**_VALID_PAYLOAD, "age": 121})


def test_prediction_request_rejects_invalid_home_ownership() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**_VALID_PAYLOAD, "home_ownership": "PALACE"})


def test_prediction_request_rejects_credit_utilization_above_one() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**_VALID_PAYLOAD, "credit_utilization": 1.5})


def test_prediction_request_rejects_missing_required_field() -> None:
    payload = {key: value for key, value in _VALID_PAYLOAD.items() if key != "age"}

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)
