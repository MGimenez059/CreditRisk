"""Trains models for the credit default pipeline.

`train_baseline` (roadmap Phase 3, Logistic Regression / Random Forest) is
fully implemented. XGBoost is an initial Phase 4 candidate, not a selected
production model.

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
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from credit_risk.ml.features import add_derived_features
from credit_risk.ml.preprocessing import RAW_FEATURE_COLUMNS, build_preprocessing_pipeline
from credit_risk.ml.protocols import FittedPipeline

DEFAULT_RANDOM_STATE = 42

BaselineModelType = Literal["logistic_regression", "random_forest"]
CandidateModelType = Literal["logistic_regression", "random_forest", "xgboost"]


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
        x_train: Training features (approximately 70% of rows).
        x_val: Validation features (approximately 15% of rows).
        x_test: Test features (approximately 15% of rows). Not scored until final model
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

    Split unique raw-input groups 70/15/15, stratifying by each group's
    majority target (positive on ties) with random_state=42. All rows are
    retained. Row proportions and class balance are approximate when group
    sizes differ. Report actual sizes in the experiment manifest.

    Args:
        frame: The full prepared dataset, including `target_column`.
        target_column: Name of the binary target column.

    Returns:
        The six-way train/validation/test feature/target split.
    """
    features = frame[RAW_FEATURE_COLUMNS]
    target = frame[target_column]
    # Identical model inputs stay together, including conflicting labels.
    # Group strata use the majority label; ties belong to the positive stratum.
    groups = pd.util.hash_pandas_object(features, index=False)
    group_labels = (target.groupby(groups).mean() >= 0.5).astype(int)
    train_groups, holdout_groups = train_test_split(
        group_labels.index,
        test_size=0.30,
        stratify=group_labels,
        random_state=DEFAULT_RANDOM_STATE,
    )
    val_groups, test_groups = train_test_split(
        holdout_groups,
        test_size=0.50,
        stratify=group_labels.loc[holdout_groups],
        random_state=DEFAULT_RANDOM_STATE,
    )
    train_mask = groups.isin(train_groups)
    val_mask = groups.isin(val_groups)
    test_mask = groups.isin(test_groups)
    return DatasetSplit(
        x_train=features.loc[train_mask],
        x_val=features.loc[val_mask],
        x_test=features.loc[test_mask],
        y_train=target.loc[train_mask],
        y_val=target.loc[val_mask],
        y_test=target.loc[test_mask],
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
    if model_type != "random_forest":
        raise ValueError(f"Unsupported baseline: {model_type}")
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
            ("features", FunctionTransformer(add_derived_features, validate=False)),
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
    pipeline.fit(split.x_train[RAW_FEATURE_COLUMNS], split.y_train)
    # sklearn has no type stubs (see pyproject.toml's mypy overrides), so
    # its return type is Any here; Pipeline structurally satisfies FittedPipeline.
    return pipeline  # type: ignore[no-any-return]


def build_candidate_pipeline(model_type: CandidateModelType) -> Pipeline:
    """Build a fixed initial candidate with the shared raw-input transformations."""
    if model_type != "xgboost":
        return build_baseline_pipeline(model_type)
    # Fixed before validation: modest CPU histogram model, without class weighting.
    estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=1.0,
        random_state=DEFAULT_RANDOM_STATE,
        n_jobs=1,
    )
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(add_derived_features, validate=False)),
            ("preprocessing", build_preprocessing_pipeline()),
            ("model", estimator),
        ]
    )
