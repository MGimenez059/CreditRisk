"""Unit tests for `credit_risk.ml.preprocessing`."""

import pandas as pd
import pytest

from credit_risk.ml.preprocessing import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    EXCLUDED_LEAKAGE_RISK_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessing_pipeline,
)

_SAMPLE_ROWS = {
    "person_age": [25, 40, None],
    "person_income": [45_000.0, 82_000.0, 60_000.0],
    "person_emp_length": [2.0, 10.0, 5.0],
    "loan_amnt": [10_000.0, 25_000.0, 15_000.0],
    "loan_int_rate": [11.5, 7.9, 13.2],
    "cb_person_cred_hist_length": [3, 12, 6],
    "loan_to_income": [0.22, 0.30, 0.25],
    "income_per_employment_year": [22_500.0, 8_200.0, 12_000.0],
    "credit_age_ratio": [0.12, 0.3, 0.2],
    "person_home_ownership": ["RENT", "MORTGAGE", "OWN"],
    "loan_intent": ["PERSONAL", "EDUCATION", "VENTURE"],
    "cb_person_default_on_file": ["N", "N", "Y"],
}


def test_loan_grade_is_excluded_from_every_feature_list() -> None:
    assert "loan_grade" not in NUMERIC_FEATURES
    assert "loan_grade" not in CATEGORICAL_FEATURES
    assert EXCLUDED_LEAKAGE_RISK_FEATURES == ["loan_grade"]


def test_all_feature_columns_combines_numeric_and_categorical() -> None:
    assert [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES] == ALL_FEATURE_COLUMNS


def test_preprocessing_pipeline_fits_and_transforms_sample_data() -> None:
    frame = pd.DataFrame(_SAMPLE_ROWS)
    pipeline = build_preprocessing_pipeline()

    transformed = pipeline.fit_transform(frame[ALL_FEATURE_COLUMNS])

    assert transformed.shape[0] == 3


def test_preprocessing_pipeline_imputes_missing_numeric_values() -> None:
    frame = pd.DataFrame(_SAMPLE_ROWS)
    pipeline = build_preprocessing_pipeline()

    # person_age has a None in row 2 — this must not raise.
    transformed = pipeline.fit_transform(frame[ALL_FEATURE_COLUMNS])

    assert not pd.isna(transformed).any()


def test_preprocessing_pipeline_handles_an_unseen_category_at_transform_time() -> None:
    frame = pd.DataFrame(_SAMPLE_ROWS)
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(frame[ALL_FEATURE_COLUMNS])

    unseen = frame[ALL_FEATURE_COLUMNS].iloc[[0]].copy()
    unseen["person_home_ownership"] = "SOMETHING_NEVER_SEEN"

    transformed = pipeline.transform(unseen)  # must not raise (handle_unknown="ignore")

    assert transformed.shape[0] == 1


@pytest.mark.parametrize("column", ["person_age", "person_home_ownership"])
def test_feature_lists_reference_columns_present_in_data_dictionary(column: str) -> None:
    # A cheap smoke check that the constants haven't drifted from the
    # sample schema used across every other Phase 2/3 test fixture.
    assert column in NUMERIC_FEATURES or column in CATEGORICAL_FEATURES
