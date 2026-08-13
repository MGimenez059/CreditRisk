"""Orchestrates per-prediction explanations.

Delegates the actual SHAP computation to `credit_risk.ml.explain`, which is
implemented in roadmap Phase 5. This service owns the mapping from raw SHAP
output to the API's `FeatureContribution` shape.
"""

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's contribution to a single prediction."""

    feature: str
    impact: float
    direction: Literal["positive", "negative"]


class ExplanationService:
    """Produces ranked feature contributions for a single prediction."""

    def explain(self, features: pd.DataFrame) -> list[FeatureContribution]:
        """Return the top feature contributions for a single-row feature frame.

        Args:
            features: A single-row DataFrame in the exact column order the
                active model was trained on.

        Returns:
            Feature contributions ordered by descending absolute impact.

        Raises:
            ExplanationError: If the SHAP explainer cannot be built or fails
                to produce values for the given input.
        """
        # TODO(ROADMAP-P5): delegate to credit_risk.ml.explain once the SHAP
        # explainer is implemented.
        raise NotImplementedError(
            "Explanation generation is implemented in roadmap Phase 5 (Explicabilidad)."
        )
