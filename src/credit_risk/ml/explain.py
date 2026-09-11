"""Tree SHAP for the selected uncalibrated binary XGBoost raw-input pipeline."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from shap.utils._exceptions import ExplainerError
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

from credit_risk.exceptions import ExplanationError
from credit_risk.ml.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from credit_risk.ml.protocols import FittedPipeline


@dataclass(frozen=True)
class PredictionExplanation:
    """All grouped contributions reconstruct output_value in natural log-odds."""

    base_value: float
    output_value: float
    probability: float
    contributions: list[tuple[str, float]]


def _feature_groups(names: list[str]) -> dict[str, list[int]]:
    """Sum encoded categories per source field; retain derived numeric features."""
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        if name.startswith("numeric__") and name.removeprefix("numeric__") in NUMERIC_FEATURES:
            feature = name.removeprefix("numeric__")
        else:
            matches = [
                field for field in CATEGORICAL_FEATURES if name.startswith(f"categorical__{field}_")
            ]
            if len(matches) != 1:
                raise ExplanationError(f"Unsupported transformed feature: {name}")
            feature = matches[0]
        groups.setdefault(feature, []).append(index)
    return groups


def explain_batch(pipeline: FittedPipeline, features: pd.DataFrame) -> list[PredictionExplanation]:
    """Explain raw input rows without fitting or mutating the saved pipeline.

    Uses tree_path_dependent background (training path counts), not a new sample.
    Grouped sums preserve additivity but are not independently recomputed group SHAP.
    Calibrated wrappers and other model families are deliberately rejected.
    """
    if not isinstance(pipeline, Pipeline) or not isinstance(
        pipeline.named_steps.get("model"), XGBClassifier
    ):
        raise ExplanationError("Explanations require an uncalibrated XGBoost pipeline.")
    if features.empty:
        raise ExplanationError("At least one input row is required.")
    model = pipeline.named_steps["model"]
    if model.get_params()["objective"] != "binary:logistic":
        raise ExplanationError("Only binary logistic XGBoost output is supported.")
    try:
        derived = pipeline.named_steps["features"].transform(features)
        preprocessing = pipeline.named_steps["preprocessing"]
        transformed = preprocessing.transform(derived)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        groups = _feature_groups(list(preprocessing.get_feature_names_out()))
        explainer = shap.TreeExplainer(
            model, model_output="raw", feature_perturbation="tree_path_dependent"
        )
        values = np.asarray(explainer.shap_values(transformed, check_additivity=True), dtype=float)
        base = float(np.asarray(explainer.expected_value).item())
        margins = np.asarray(model.predict(transformed, output_margin=True), dtype=float)
        probabilities = np.asarray(pipeline.predict_proba(features)[:, 1], dtype=float)
    except (ValueError, TypeError, KeyError, AttributeError, ExplainerError, XGBoostError) as err:
        raise ExplanationError("Unable to explain this model/input combination.") from err
    reconstructed = base + values.sum(axis=1)
    if not np.isfinite(values).all() or not np.isfinite(base):
        raise ExplanationError("SHAP returned non-finite contributions.")
    if not np.allclose(reconstructed, margins, atol=1e-5, rtol=1e-5):
        raise ExplanationError("SHAP contributions do not reconstruct the raw margin.")
    if not np.allclose(np.exp(-np.logaddexp(0, -reconstructed)), probabilities, atol=1e-6):
        raise ExplanationError("SHAP output does not match pipeline probabilities.")
    return [
        PredictionExplanation(
            base,
            float(margins[row]),
            float(probabilities[row]),
            sorted(
                [(name, float(values[row, indices].sum())) for name, indices in groups.items()],
                key=lambda item: -abs(item[1]),
            ),
        )
        for row in range(len(features))
    ]


def explain_prediction(pipeline: FittedPipeline, features: pd.DataFrame) -> PredictionExplanation:
    """Explain exactly one raw-input row, retaining all contributions and the base."""
    if len(features) != 1:
        raise ExplanationError("A local explanation requires exactly one input row.")
    return explain_batch(pipeline, features)[0]
