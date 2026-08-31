"""Trains models for the credit default pipeline.

`train_baseline` (roadmap Phase 3, Logistic Regression / Random Forest) is
fully implemented. `train_model` (roadmap Phase 4, the production XGBoost
pipeline) remains a stub — see its own docstring.

Per CODESTYLE.md §14: `random_state=42` is set everywhere randomness is
involved, and the trained pipeline is a single serializable artifact
(preprocessing + feature engineering + estimator) — see
`ml.registry.save_model_artifact`.
"""

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from credit_risk.ml.preprocessing import ALL_FEATURE_COLUMNS, build_preprocessing_pipeline
from credit_risk.ml.protocols import FittedPipeline
from credit_risk.ml.registry import ModelArtifactMetadata

DEFAULT_RANDOM_STATE = 42

BaselineModelType = Literal["logistic_regression", "random_forest"]


@dataclass(frozen=True)
class ClassBalance:
    """Class balance measurements required by SPECS.md §11.

    Attributes:
        positive_rate: Fraction of rows where `loan_status == 1` (default).
        negative_rate: Fraction of rows where `loan_status == 0` (no default).
        class_ratio: `negative_rate / positive_rate` — e.g. `3.5` means 3.5
            non-defaults per default. `inf` if there are no positive examples.
    """

    positive_rate: float
    negative_rate: float
    class_ratio: float


@dataclass(frozen=True)
class DatasetSplit:
    """A stratified train/validation/test split, per SPECS.md §10.

    Attributes:
        x_train: Training features (70% of rows).
        x_val: Validation features (15% of rows).
        x_test: Test features (15% of rows). Untouched until final model
            selection is complete, per SPECS.md §10.
        y_train: Training target, aligned with `x_train`.
        y_val: Validation target, aligned with `x_val`.
        y_test: Test target, aligned with `x_test`.
    """

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def measure_class_balance(target: pd.Series) -> ClassBalance:
    """Measure the target's class balance, per SPECS.md §11.

    Args:
        target: The binary `loan_status` column (0 = no default, 1 = default).

    Returns:
        Positive rate, negative rate, and the ratio of negative to positive
        examples (e.g. `3.5` means 3.5 non-defaults per default).
    """
    positive_rate = float(target.mean())
    negative_rate = 1.0 - positive_rate
    class_ratio = negative_rate / positive_rate if positive_rate else float("inf")
    return ClassBalance(
        positive_rate=positive_rate, negative_rate=negative_rate, class_ratio=class_ratio
    )


def split_dataset(frame: pd.DataFrame, target_column: str) -> DatasetSplit:
    """Split into stratified train/validation/test sets, per SPECS.md §10.

    First splits off 70% train / 30% held-out, then splits that 30% held-out
    portion evenly into validation (15% of the original) and test (15% of
    the original) — both steps stratified by the target with
    `random_state=42`, matching SPECS.md §10 exactly.

    Args:
        frame: The full prepared dataset, including `target_column`.
        target_column: Name of the binary target column.

    Returns:
        The six-way train/validation/test feature/target split.
    """
    features = frame.drop(columns=[target_column])
    target = frame[target_column]

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features,
        target,
        test_size=0.30,
        stratify=target,
        random_state=DEFAULT_RANDOM_STATE,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=0.50,
        stratify=y_holdout,
        random_state=DEFAULT_RANDOM_STATE,
    )
    return DatasetSplit(
        x_train=x_train, x_val=x_val, x_test=x_test, y_train=y_train, y_val=y_val, y_test=y_test
    )


def _build_baseline_estimator(
    model_type: BaselineModelType,
) -> LogisticRegression | RandomForestClassifier:
    """Construct the estimator for a baseline model type.

    Args:
        model_type: Which baseline to build.

    Returns:
        An unfitted, class-weight-balanced estimator (SPECS.md §11 —
        the dataset is imbalanced ~78/22) with `random_state=42`.
    """
    if model_type == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=DEFAULT_RANDOM_STATE
        )
    return RandomForestClassifier(
        class_weight="balanced", random_state=DEFAULT_RANDOM_STATE, n_jobs=-1
    )


def build_baseline_pipeline(model_type: BaselineModelType) -> Pipeline:
    """Build the full unfitted pipeline (preprocessing + estimator) for a baseline.

    Args:
        model_type: Which baseline to build.

    Returns:
        An unfitted `sklearn.Pipeline`. Fit only on the training split
        (SPECS.md §9) via `train_baseline`.
    """
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("model", _build_baseline_estimator(model_type)),
        ]
    )


def train_baseline(model_type: BaselineModelType, split: DatasetSplit) -> FittedPipeline:
    """Fit a baseline pipeline on the training split only.

    Args:
        model_type: Which baseline to train.
        split: Output of `split_dataset`. Only `split.x_train` /
            `split.y_train` are used — validation and test stay untouched
            (SPECS.md §9 rules 1, 2, and 6).

    Returns:
        The fitted pipeline, ready for `ml.evaluate.evaluate_model`.
    """
    pipeline = build_baseline_pipeline(model_type)
    pipeline.fit(split.x_train[ALL_FEATURE_COLUMNS], split.y_train)
    # sklearn has no type stubs (see pyproject.toml's mypy overrides), so
    # its return type is Any here; Pipeline structurally satisfies FittedPipeline.
    return pipeline  # type: ignore[no-any-return]


def train_model(
    training_data: pd.DataFrame,
    dataset_version: str,
) -> tuple[FittedPipeline, ModelArtifactMetadata]:
    """Train the production XGBoost pipeline on a prepared training set.

    Args:
        training_data: Output of `ml.preprocessing` and `ml.features`,
            including the `loan_status` target column.
        dataset_version: Identifier of the dataset snapshot used, recorded
            in the resulting metadata for traceability.

    Returns:
        The fitted pipeline and its metadata, ready for
        `ml.registry.save_model_artifact`.

    Raises:
        NotImplementedError: Always, until roadmap Phase 4 is implemented.
    """
    # TODO(ROADMAP-P4): train XGBoost with Optuna tuning and
    # StratifiedKFold cross-validation, per docs/model_card.md once the
    # Phase 3 baselines (train_baseline, above) have set the comparison bar.
    raise NotImplementedError("Model training is implemented in roadmap Phase 4 (XGBoost).")
