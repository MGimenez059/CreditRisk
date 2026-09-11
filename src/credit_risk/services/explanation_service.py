"""Map model explanations into domain objects shared with the response boundary."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from credit_risk.ml.explain import explain_prediction
from credit_risk.ml.protocols import FittedPipeline


@dataclass(frozen=True)
class FeatureContribution:
    """Signed impact in log-odds, not probability percentage points."""

    feature: str
    impact: float
    direction: Literal["positive", "negative", "neutral"]


@dataclass(frozen=True)
class Explanation:
    """Complete additive explanation of the uncalibrated model output."""

    base_value: float
    output_value: float
    contributions: list[FeatureContribution]
    output_space: Literal["log_odds"] = "log_odds"
    method: Literal["tree_path_dependent"] = "tree_path_dependent"
    explains: Literal["uncalibrated_model"] = "uncalibrated_model"


class ExplanationService:
    """Use the same loaded pipeline as prediction, avoiding model identity drift."""

    def explain(self, pipeline: FittedPipeline, features: pd.DataFrame) -> Explanation:
        """Return all grouped impacts, ranked by absolute magnitude."""
        result = explain_prediction(pipeline, features)
        return Explanation(
            result.base_value,
            result.output_value,
            [
                FeatureContribution(
                    name,
                    impact,
                    "positive" if impact > 0 else "negative" if impact < 0 else "neutral",
                )
                for name, impact in result.contributions
            ],
        )
