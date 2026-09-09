"""Group-preserving cross-validation and bounded candidate search."""

from dataclasses import asdict, dataclass, field

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from credit_risk.ml.evaluate import evaluate_model
from credit_risk.ml.preprocessing import RAW_FEATURE_COLUMNS
from credit_risk.ml.train import (
    DEFAULT_RANDOM_STATE,
    CandidateModelType,
    build_candidate_pipeline,
    measure_class_balance,
)


@dataclass(frozen=True)
class Candidate:
    """Estimator configuration; weighting is resolved on each training fold."""

    name: str
    model_type: CandidateModelType
    weighted: bool
    parameters: dict[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class CVResult:
    """Per-fold evidence and aggregate metrics at threshold 0.5."""

    candidate: Candidate
    folds: list[dict[str, float]]
    mean: dict[str, float]
    std: dict[str, float]


def grouped_folds(
    features: pd.DataFrame, target: pd.Series, n_splits: int = 5
) -> list[tuple[list[int], list[int]]]:
    """Return relative row positions; identical raw inputs never cross folds."""
    groups = pd.util.hash_pandas_object(features[RAW_FEATURE_COLUMNS], index=False)
    labels = (target.groupby(groups).mean() >= 0.5).astype(int)
    if n_splits < 2 or labels.value_counts().min() < n_splits or labels.nunique() != 2:
        raise ValueError("Each group-label class needs at least n_splits groups.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    return [
        (
            np.flatnonzero(groups.isin(labels.index[train])).tolist(),
            np.flatnonzero(groups.isin(labels.index[validation])).tolist(),
        )
        for train, validation in splitter.split(labels.index, labels)
    ]


def configured_pipeline(candidate: Candidate, target: pd.Series) -> Pipeline:
    """Construct a raw-input pipeline with weights derived only from supplied labels."""
    pipeline = build_candidate_pipeline(candidate.model_type)
    pipeline.set_params(**{f"model__{key}": value for key, value in candidate.parameters.items()})
    if candidate.model_type == "xgboost":
        ratio = measure_class_balance(target).class_ratio if candidate.weighted else 1.0
        pipeline.set_params(model__scale_pos_weight=ratio)
    else:
        pipeline.set_params(model__class_weight="balanced" if candidate.weighted else None)
    return pipeline


def cross_validate_candidate(
    candidate: Candidate,
    features: pd.DataFrame,
    target: pd.Series,
    folds: list[tuple[list[int], list[int]]],
) -> CVResult:
    """Fit fresh transformations and estimator inside each supplied grouped fold."""
    reports = []
    for train, validation in folds:
        pipeline = configured_pipeline(candidate, target.take(train))
        pipeline.fit(features.iloc[train], target.take(train))
        reports.append(
            asdict(evaluate_model(pipeline, features.iloc[validation], target.take(validation)))
        )
    values = pd.DataFrame(reports)
    return CVResult(
        candidate,
        reports,
        {str(key): float(value) for key, value in values.mean().items()},
        {str(key): float(value) for key, value in values.std(ddof=1).items()},
    )


def select_candidate(results: list[CVResult]) -> CVResult:
    """Within 0.002 of best ROC-AUC, prefer log loss, then simpler family and name."""
    best_auc = max(result.mean["roc_auc"] for result in results)
    eligible = [result for result in results if result.mean["roc_auc"] >= best_auc - 0.002]
    complexity = {"logistic_regression": 0, "random_forest": 1, "xgboost": 2}
    return min(
        eligible,
        key=lambda result: (
            result.mean["log_loss"],
            complexity[result.candidate.model_type],
            result.candidate.name,
        ),
    )


def search_candidates(
    features: pd.DataFrame,
    target: pd.Series,
    folds: list[tuple[list[int], list[int]]],
    n_trials: int = 12,
) -> tuple[list[CVResult], list[CVResult]]:
    """Compare six fixed configurations and one seeded, sequential XGBoost search."""
    if n_trials < 1:
        raise ValueError("At least one tuning trial is required.")
    families: tuple[CandidateModelType, ...] = ("logistic_regression", "random_forest", "xgboost")
    results = []
    for family in families:
        for weighted in (False, True):
            candidate = Candidate(
                f"{family}-{'weighted' if weighted else 'unweighted'}", family, weighted
            )
            print(f"Cross-validating {candidate.name}", flush=True)
            results.append(cross_validate_candidate(candidate, features, target, folds))
    trials: list[CVResult] = []

    def objective(trial: optuna.Trial) -> float:
        parameters = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300]),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 5, 10]),
        }
        weighted = trial.suggest_categorical("weighted", [False, True])
        candidate = Candidate(f"xgboost-trial-{trial.number}", "xgboost", weighted, parameters)
        result = cross_validate_candidate(candidate, features, target, folds)
        trials.append(result)
        return result.mean["roc_auc"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_STATE, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    results.append(trials[study.best_trial.number])
    return results, trials
