"""Development-only calibration and threshold selection."""

from dataclasses import asdict

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score, precision_score, recall_score

from credit_risk.ml.evaluate import evaluate_probabilities
from credit_risk.ml.protocols import FittedPipeline


def reliability_bins(
    target: pd.Series, probabilities: npt.NDArray[np.float64]
) -> dict[str, object]:
    """Ten uniform probability bins; empty bins are omitted by scikit-learn."""
    observed, predicted = calibration_curve(target, probabilities, n_bins=10, strategy="uniform")
    return {"mean_predicted": predicted.tolist(), "fraction_positive": observed.tolist()}


def choose_threshold(
    target: pd.Series, probabilities: npt.NDArray[np.float64]
) -> tuple[float, list[dict[str, float]]]:
    """Maximize demo F1; ties favor precision, then higher threshold."""
    grid = []
    for step in range(5, 96):
        threshold = step / 100
        predictions = probabilities >= threshold
        grid.append(
            {
                "threshold": threshold,
                "f1": float(f1_score(target, predictions, zero_division=0)),
                "precision": float(precision_score(target, predictions, zero_division=0)),
                "recall": float(recall_score(target, predictions, zero_division=0)),
            }
        )
    best = max(grid, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    return best["threshold"], grid


def calibrate_and_select(
    base: FittedPipeline,
    calibration_features: pd.DataFrame,
    calibration_target: pd.Series,
    decision_features: pd.DataFrame,
    decision_target: pd.Series,
) -> tuple[FittedPipeline, str, float, dict[str, object]]:
    """Fit sigmoid on separate calibration rows, then choose on decision rows."""
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(calibration_features, calibration_target)
    options: dict[str, FittedPipeline] = {"none": base, "sigmoid": calibrated}
    evidence: dict[str, object] = {}
    losses = {}
    for name, model in options.items():
        probabilities = model.predict_proba(decision_features)[:, 1]
        metrics = asdict(evaluate_probabilities(decision_target, probabilities))
        evidence[name] = {
            "metrics_at_0_5": metrics,
            "reliability": reliability_bins(decision_target, probabilities),
        }
        losses[name] = metrics["log_loss"]
    choice = min(options, key=lambda name: (losses[name], name != "none"))
    selected = options[choice]
    probabilities = selected.predict_proba(decision_features)[:, 1]
    threshold, grid = choose_threshold(decision_target, probabilities)
    evidence["threshold_grid"] = grid
    evidence["selected_metrics"] = asdict(
        evaluate_probabilities(decision_target, probabilities, threshold)
    )
    return selected, choice, threshold, evidence
