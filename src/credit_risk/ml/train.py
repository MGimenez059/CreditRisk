"""Trains the credit default model, implemented in roadmap Phase 4.

Per CODESTYLE.md §14: `random_state=42` (or an explicitly configured seed)
must be set everywhere randomness is involved once this is implemented, and
the trained pipeline must be a single serializable artifact (preprocessing +
feature engineering + estimator) — see `ml.registry.save_model_artifact`.
"""

import pandas as pd

from credit_risk.ml.protocols import FittedPipeline
from credit_risk.ml.registry import ModelArtifactMetadata

DEFAULT_RANDOM_STATE = 42


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
    # TODO(ROADMAP-P4): train Logistic Regression + Random Forest baselines,
    # then XGBoost with Optuna tuning and StratifiedKFold cross-validation,
    # per docs/model_card.md once it is populated with real metrics.
    raise NotImplementedError("Model training is implemented in roadmap Phase 4 (XGBoost).")
