"""Reusable, scikit-learn-compatible preprocessing transformers.

Per SPECS.md §9 (Data Leakage Prevention), this module only *builds* the
transformer — it never calls `.fit()` itself. `ml.train` is responsible for
fitting it exclusively on the training split, never on validation, test,
or the full dataset.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_cred_hist_length",
    "loan_to_income",
    "income_per_employment_year",
    "credit_age_ratio",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]

# `loan_grade` is deliberately excluded from both feature lists above.
# SPECS.md §8 lists it as a candidate "Loan feature", but SPECS.md §9 rule
# 5 ("avoid features that would only be available after the credit
# decision") takes precedence: the lender assigns `loan_grade` as part of
# the same underwriting decision this model is meant to inform. Its
# near-perfect correlation with the target in the real dataset (grade G
# defaults at ~98%, per the EDA in notebooks/01_data_exploration.ipynb) is
# consistent with that leakage risk rather than genuine independent
# signal. See docs/model_card.md's Limitations section.
EXCLUDED_LEAKAGE_RISK_FEATURES = ["loan_grade"]

# Also excluded: `loan_percent_income`, superseded by the derived
# `loan_to_income` in NUMERIC_FEATURES above (identical computation,
# re-derived from raw columns) — see `ml.features`'s module docstring.
ALL_FEATURE_COLUMNS = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build the column-wise preprocessing pipeline for the raw feature schema.

    Returns:
        An unfitted `ColumnTransformer`: median imputation + standard
        scaling for `NUMERIC_FEATURES`, most-frequent imputation + one-hot
        encoding for `CATEGORICAL_FEATURES`. The caller fits it as part of
        a full pipeline on the training split only (SPECS.md §9).
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
