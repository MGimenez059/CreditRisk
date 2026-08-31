"""Unit tests for `credit_risk.ml.train`'s baseline-training functions.

`train_model` (the Phase 4 XGBoost stub) is not tested here — it still
raises `NotImplementedError` by design; see `ml/train.py`'s docstring.
"""

import numpy as np
import pandas as pd
import pytest

from credit_risk.ml.features import add_derived_features
from credit_risk.ml.train import (
    build_baseline_pipeline,
    measure_class_balance,
    split_dataset,
    train_baseline,
)


def _synthetic_dataset(n: int = 200) -> pd.DataFrame:
    """Build a small, clearly-synthetic dataset with the real column names.

    Never a fixture that could be mistaken for a sample of the real
    Kaggle dataset — matches the pattern already used in
    `tests/unit/test_ingest_data.py`.
    """
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "person_age": rng.integers(20, 65, n),
            "person_income": rng.integers(20_000, 150_000, n),
            "person_home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n),
            "person_emp_length": rng.integers(0, 20, n).astype(float),
            "loan_intent": rng.choice(
                ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"],
                n,
            ),
            "loan_amnt": rng.integers(1_000, 35_000, n).astype(float),
            "loan_int_rate": rng.uniform(5, 23, n),
            "loan_status": rng.choice([0, 1], n, p=[0.78, 0.22]),
            "cb_person_default_on_file": rng.choice(["Y", "N"], n, p=[0.15, 0.85]),
            "cb_person_cred_hist_length": rng.integers(1, 25, n),
        }
    )
    return add_derived_features(frame)


# --- measure_class_balance -----------------------------------------------


def test_measure_class_balance_computes_rates_that_sum_to_one() -> None:
    target = pd.Series([0, 0, 0, 1])

    balance = measure_class_balance(target)

    assert balance.positive_rate == pytest.approx(0.25)
    assert balance.negative_rate == pytest.approx(0.75)
    assert balance.positive_rate + balance.negative_rate == pytest.approx(1.0)


def test_measure_class_balance_computes_class_ratio() -> None:
    target = pd.Series([0, 0, 0, 1])

    balance = measure_class_balance(target)

    assert balance.class_ratio == pytest.approx(3.0)


# --- split_dataset --------------------------------------------------------


def test_split_dataset_produces_roughly_70_15_15_proportions() -> None:
    frame = _synthetic_dataset(n=200)

    split = split_dataset(frame, target_column="loan_status")

    assert len(split.x_train) == pytest.approx(140, abs=2)
    assert len(split.x_val) == pytest.approx(30, abs=2)
    assert len(split.x_test) == pytest.approx(30, abs=2)


def test_split_dataset_keeps_features_and_target_aligned() -> None:
    frame = _synthetic_dataset(n=200)

    split = split_dataset(frame, target_column="loan_status")

    assert len(split.x_train) == len(split.y_train)
    assert len(split.x_val) == len(split.y_val)
    assert len(split.x_test) == len(split.y_test)


def test_split_dataset_is_deterministic_given_the_fixed_random_state() -> None:
    frame = _synthetic_dataset(n=200)

    first = split_dataset(frame, target_column="loan_status")
    second = split_dataset(frame, target_column="loan_status")

    assert list(first.y_train.index) == list(second.y_train.index)


def test_split_dataset_excludes_the_target_from_the_feature_frames() -> None:
    frame = _synthetic_dataset(n=200)

    split = split_dataset(frame, target_column="loan_status")

    assert "loan_status" not in split.x_train.columns


# --- build_baseline_pipeline / train_baseline ----------------------------


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest"])
def test_build_baseline_pipeline_builds_an_unfitted_pipeline(model_type: str) -> None:
    pipeline = build_baseline_pipeline(model_type)  # type: ignore[arg-type]

    assert not hasattr(pipeline, "classes_")  # unfitted sklearn estimators lack this attribute


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest"])
def test_train_baseline_produces_a_pipeline_that_predicts_probabilities(model_type: str) -> None:
    frame = _synthetic_dataset(n=200)
    split = split_dataset(frame, target_column="loan_status")

    fitted = train_baseline(model_type, split)  # type: ignore[arg-type]
    probabilities = fitted.predict_proba(split.x_val)

    assert probabilities.shape == (len(split.x_val), 2)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_split_dataset_train_val_test_indices_never_overlap() -> None:
    """Guards SPECS.md §9 rules 1 and 6: splits must stay isolated from each other."""
    frame = _synthetic_dataset(n=200)

    split = split_dataset(frame, target_column="loan_status")

    train_idx, val_idx, test_idx = (
        set(split.x_train.index),
        set(split.x_val.index),
        set(split.x_test.index),
    )
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
