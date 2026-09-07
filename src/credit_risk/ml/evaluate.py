"""Evaluates a trained pipeline against a held-out split.

Metrics computed here populate `docs/model_card.md` and (once Phase 4
trains the production model) the `metrics` recorded in `ModelArtifactMetadata`.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credit_risk.ml.protocols import FittedPipeline

# Baseline models (roadmap Phase 3) report F1 at this fixed threshold for a
# simple, comparable sanity check across Logistic Regression and Random
# Forest. Per SPECS.md §17, the *production* model's decision threshold
# (Phase 4, XGBoost) is selected against a business objective, not this
# default — this constant is not that threshold.
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5


@dataclass(frozen=True)
class EvaluationReport:
    """Evaluation metrics for a single trained pipeline, per README.md.

    Attributes:
        roc_auc: Area under the ROC curve — discrimination, threshold-independent.
        pr_auc: Average precision (non-interpolated PR summary) — more informative
            than ROC-AUC alone on an imbalanced target (SPECS.md §11).
        f1: F1 score at `DEFAULT_CLASSIFICATION_THRESHOLD`.
        log_loss: Cross-entropy loss on the predicted probabilities.
        brier_score: Mean squared error between predicted probability and
            the true label — a calibration signal, not just discrimination.
    """

    roc_auc: float
    pr_auc: float
    f1: float
    log_loss: float
    brier_score: float
    precision: float
    recall: float


def evaluate_model(
    pipeline: FittedPipeline,
    holdout_features: pd.DataFrame,
    holdout_target: pd.Series,
) -> EvaluationReport:
    """Score a fitted pipeline against a held-out validation or test set.

    Args:
        pipeline: A fitted scikit-learn-compatible pipeline.
        holdout_features: Feature frame not used during training.
        holdout_target: True `loan_status` labels aligned with `holdout_features`.

    Returns:
        The full set of metrics tracked in README.md's Model section.
        Accuracy is deliberately not included — SPECS.md §16 excludes it
        as a primary metric for an imbalanced target.
    """
    probabilities = pipeline.predict_proba(holdout_features)[:, 1]
    predictions = (probabilities >= DEFAULT_CLASSIFICATION_THRESHOLD).astype(int)

    return EvaluationReport(
        roc_auc=float(roc_auc_score(holdout_target, probabilities)),
        pr_auc=float(average_precision_score(holdout_target, probabilities)),
        f1=float(f1_score(holdout_target, predictions)),
        log_loss=float(log_loss(holdout_target, probabilities)),
        brier_score=float(brier_score_loss(holdout_target, probabilities)),
        precision=float(precision_score(holdout_target, predictions, zero_division=0)),
        recall=float(recall_score(holdout_target, predictions, zero_division=0)),
    )
