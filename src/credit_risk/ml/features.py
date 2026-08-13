"""Derived feature engineering, implemented in roadmap Phase 3.

Kept separate from `ml.preprocessing`: preprocessing handles scaling and
encoding of existing columns, this module derives new columns from them
(e.g. `debt_to_income`), per the layering note in CODESTYLE.md §3.
"""

import pandas as pd


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns to a raw feature frame.

    Args:
        frame: Raw feature frame matching `docs/data_dictionary.md`.

    Returns:
        A copy of `frame` with derived columns appended (e.g.
        `debt_to_income`, `credit_history_to_age_ratio`).

    Raises:
        NotImplementedError: Always, until roadmap Phase 3 is implemented.
    """
    # TODO(ROADMAP-P3): derive domain features once the training dataset
    # (Kaggle laotse/credit-risk-dataset) is ingested and profiled.
    raise NotImplementedError("Feature engineering is implemented in roadmap Phase 3.")
