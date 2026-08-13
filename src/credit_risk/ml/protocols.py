"""Structural typing for fitted ML pipelines.

`Any` is disallowed by CODESTYLE.md §5 unless justified; a `Protocol` gives
`ml.predict`, `ml.evaluate`, `ml.explain`, and `ml.registry` a concrete,
checkable type for "whatever scikit-learn-compatible object `ml.train`
produces" without coupling them to a specific estimator class.
"""

from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd


class FittedPipeline(Protocol):
    """Structural type for a fitted scikit-learn-compatible inference pipeline."""

    # X/y naming matches scikit-learn's convention, allowed by CODESTYLE.md §4.
    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:  # noqa: N803
        """Return class probabilities for each row of `X`."""
        ...
