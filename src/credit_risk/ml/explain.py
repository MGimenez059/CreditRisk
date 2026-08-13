"""Computes SHAP-based explanations, implemented in roadmap Phase 5.

Called exclusively by `credit_risk.services.explanation_service`, per the
layering rules in CODESTYLE.md §3 — nothing outside `ml/` should import
`shap` directly.
"""

import pandas as pd

from credit_risk.ml.protocols import FittedPipeline


def explain_prediction(pipeline: FittedPipeline, features: pd.DataFrame) -> list[tuple[str, float]]:
    """Return per-feature SHAP contributions for a single prediction.

    Args:
        pipeline: The fitted pipeline used to produce the prediction being
            explained.
        features: The single-row raw feature frame passed to `ml.predict.predict`.

    Returns:
        `(feature_name, shap_value)` pairs, ordered by descending absolute value.

    Raises:
        ExplanationError: If the SHAP explainer cannot be built for this
            pipeline or fails to produce values for the given input.
        NotImplementedError: Always, until roadmap Phase 5 is implemented.
    """
    # TODO(ROADMAP-P5): build a shap.TreeExplainer around the fitted XGBoost
    # step of the pipeline once roadmap Phase 4 produces a trained model.
    raise NotImplementedError("SHAP explanations are implemented in roadmap Phase 5.")
