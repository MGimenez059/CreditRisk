"""Derived feature engineering, per SPECS.md §8.

Kept separate from `ml.preprocessing`: preprocessing handles scaling and
encoding of existing columns, this module derives new columns from them,
per the layering note in CODESTYLE.md §3.

Unavailable derived features are deliberately omitted:
`debt_to_income` needs a `total_debt` column this dataset does not have,
and `late_payment_rate` needs `late_payments`, one of the fields
documented as absent from the Phase 0 source in `docs/data_dictionary.md`.
Both would require a richer dataset to compute honestly rather than a
fabricated proxy.
"""

import pandas as pd

from credit_risk.exceptions import InvalidFeatureSchemaError

# `loan_to_income` is deliberately re-derived from `loan_amnt` /
# `person_income` rather than reused from the source's own
# `loan_percent_income` column (which may be rounded), so it is
# reproducible from raw inputs alone, per SPECS.md §8's "reusable Python
# code" requirement. `ml.preprocessing.NUMERIC_FEATURES` uses this derived
# column and excludes `loan_percent_income` to avoid training on two
# overlapping versions of the same signal.
DERIVED_FEATURE_COLUMNS = ["loan_to_income", "income_per_employment_year", "credit_age_ratio"]


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns to a raw feature frame.

    Args:
        frame: Raw feature frame matching `docs/data_dictionary.md`, with
            at least `loan_amnt`, `person_income`, `person_emp_length`,
            `cb_person_cred_hist_length`, and `person_age`.

    Returns:
        A copy of `frame` with `loan_to_income`, `income_per_employment_year`,
        and `credit_age_ratio` appended (see `DERIVED_FEATURE_COLUMNS`).
    """
    result = frame.copy()
    if (result["person_income"] <= 0).any():
        raise InvalidFeatureSchemaError("Income must be positive to compute loan_to_income.")
    result["loan_to_income"] = result["loan_amnt"] / result["person_income"]
    result["income_per_employment_year"] = result["person_income"] / result[
        "person_emp_length"
    ].clip(lower=1)
    result["credit_age_ratio"] = result["cb_person_cred_hist_length"] / result["person_age"].clip(
        lower=1
    )
    return result
