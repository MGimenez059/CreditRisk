"""Runs inference against a loaded model pipeline.

This module has no FastAPI or database dependency, no knowledge of API
schemas, and is testable standalone, per CODESTYLE.md §3.
"""

import pandas as pd

from credit_risk.exceptions import InvalidFeatureSchemaError
from credit_risk.ml.protocols import FittedPipeline


def predict(pipeline: FittedPipeline, features: pd.DataFrame) -> float:
    """Return the predicted default probability for a single-row feature frame.

    Args:
        pipeline: A fitted scikit-learn-compatible pipeline exposing
            `predict_proba`, as produced by `ml.train` and loaded via
            `ml.registry.load_model_artifact`.
        features: A single-row DataFrame with the raw (pre-pipeline) input
            columns the pipeline's preprocessing step expects.

    Returns:
        The predicted probability of the positive (default) class, in [0, 1].

    Raises:
        InvalidFeatureSchemaError: If `features` does not match the schema
            the pipeline was fitted on.
    """
    try:
        probabilities = pipeline.predict_proba(features)
    except (ValueError, KeyError) as err:
        raise InvalidFeatureSchemaError(
            "Input features do not match the schema the model was trained on."
        ) from err
    return float(probabilities[0, 1])
