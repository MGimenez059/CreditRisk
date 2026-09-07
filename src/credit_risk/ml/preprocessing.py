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

# Excluded conservatively because its availability and derivation are unverified.
# Strong association with default is not proof of leakage; see the model card.
EXCLUDED_LEAKAGE_RISK_FEATURES = ["loan_grade"]

# Source loan_percent_income is replaced by the reproducible loan_to_income ratio.
RAW_FEATURE_COLUMNS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_cred_hist_length",
    *CATEGORICAL_FEATURES,
]

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
