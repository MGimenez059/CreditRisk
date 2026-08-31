"""Unit tests for `credit_risk.ml.features`."""

import pandas as pd
import pytest

from credit_risk.ml.features import DERIVED_FEATURE_COLUMNS, add_derived_features

_BASE_ROWS = {
    "loan_amnt": [10_000.0, 5_000.0],
    "person_income": [50_000.0, 20_000.0],
    "person_emp_length": [5.0, 0.0],
    "cb_person_cred_hist_length": [10, 3],
    "person_age": [30, 20],
}


def test_add_derived_features_adds_all_expected_columns() -> None:
    frame = add_derived_features(pd.DataFrame(_BASE_ROWS))

    for column in DERIVED_FEATURE_COLUMNS:
        assert column in frame.columns


def test_add_derived_features_computes_loan_to_income() -> None:
    frame = add_derived_features(pd.DataFrame(_BASE_ROWS))

    assert frame["loan_to_income"].iloc[0] == pytest.approx(10_000.0 / 50_000.0)


def test_add_derived_features_clips_employment_years_at_one_to_avoid_division_by_zero() -> None:
    frame = add_derived_features(pd.DataFrame(_BASE_ROWS))

    # Row 1 has person_emp_length=0.0; clip(lower=1) means the denominator
    # used is 1, not 0, so this must not be inf or NaN.
    assert frame["income_per_employment_year"].iloc[1] == pytest.approx(20_000.0 / 1.0)


def test_add_derived_features_computes_credit_age_ratio() -> None:
    frame = add_derived_features(pd.DataFrame(_BASE_ROWS))

    assert frame["credit_age_ratio"].iloc[0] == pytest.approx(10 / 30)


def test_add_derived_features_does_not_mutate_the_input_frame() -> None:
    original = pd.DataFrame(_BASE_ROWS)
    original_columns = list(original.columns)

    add_derived_features(original)

    assert list(original.columns) == original_columns
