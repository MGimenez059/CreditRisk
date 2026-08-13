"""Reusable, scikit-learn-compatible preprocessing transformers.

Implemented in roadmap Phase 3 (Baseline de ML). Transformers defined here
must be composable into a single `sklearn.pipeline.Pipeline` alongside the
feature engineering steps in `ml.features` and the estimator, so the full
inference pipeline can be serialized as one artifact per CODESTYLE.md §14.
"""

from sklearn.base import TransformerMixin


def build_preprocessing_pipeline() -> TransformerMixin:
    """Build the column-wise preprocessing pipeline for the raw feature schema.

    Returns:
        A fitted-on-call scikit-learn transformer (e.g. a `ColumnTransformer`)
        handling numeric scaling, categorical encoding, and null handling for
        every field in `docs/data_dictionary.md`.

    Raises:
        NotImplementedError: Always, until roadmap Phase 3 is implemented.
    """
    # TODO(ROADMAP-P3): implement numeric scaling + categorical encoding per
    # docs/data_dictionary.md, informed by the EDA in notebooks/01_data_exploration.ipynb.
    raise NotImplementedError(
        "Preprocessing pipeline is implemented in roadmap Phase 3 (Baseline de ML)."
    )
