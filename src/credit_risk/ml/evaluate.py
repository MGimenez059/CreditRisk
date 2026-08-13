"""Evaluates a trained pipeline, implemented in roadmap Phase 4.

Metrics computed here populate `docs/model_card.md` and the `metrics` field
of `ModelArtifactMetadata` — see CODESTYLE.md §14.
"""

from dataclasses import dataclass

import pandas as pd

from credit_risk.ml.protocols import FittedPipeline


@dataclass(frozen=True)
class EvaluationReport:
    """Evaluation metrics for a single trained pipeline, per README.md."""

    roc_auc: float
    pr_auc: float
    f1: float
    log_loss: float
    brier_score: float


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

    Raises:
        NotImplementedError: Always, until roadmap Phase 4 is implemented.
    """
    # TODO(ROADMAP-P4): compute ROC-AUC, PR-AUC, F1, Log Loss, and Brier
    # Score using scikit-learn's metrics module against StratifiedKFold splits.
    raise NotImplementedError("Model evaluation is implemented in roadmap Phase 4 (XGBoost).")
