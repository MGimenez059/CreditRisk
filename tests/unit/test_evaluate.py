"""Unit tests for `credit_risk.ml.evaluate`."""

import numpy as np
import pandas as pd

from credit_risk.ml.evaluate import evaluate_model


class _StubPipeline:
    """A pipeline stub with hand-picked, known probabilities for deterministic assertions."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        # Perfectly separates the two classes: low probability for the
        # negatives, high for the positives, so every metric below has a
        # known, checkable value rather than an opaque fitted number.
        return np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])


def test_evaluate_model_perfect_separation_gives_perfect_roc_auc() -> None:
    target = pd.Series([0, 0, 1, 1])

    report = evaluate_model(_StubPipeline(), pd.DataFrame(index=range(4)), target)

    assert report.roc_auc == 1.0


def test_evaluate_model_perfect_separation_gives_perfect_f1() -> None:
    target = pd.Series([0, 0, 1, 1])

    report = evaluate_model(_StubPipeline(), pd.DataFrame(index=range(4)), target)

    assert report.f1 == 1.0


def test_evaluate_model_returns_metrics_within_their_valid_ranges() -> None:
    target = pd.Series([0, 0, 1, 1])

    report = evaluate_model(_StubPipeline(), pd.DataFrame(index=range(4)), target)

    assert 0.0 <= report.roc_auc <= 1.0
    assert 0.0 <= report.pr_auc <= 1.0
    assert 0.0 <= report.f1 <= 1.0
    assert report.log_loss >= 0.0
    assert 0.0 <= report.brier_score <= 1.0
