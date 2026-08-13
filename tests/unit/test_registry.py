"""Unit tests for `credit_risk.ml.registry`.

Uses pytest's `tmp_path` fixture rather than any configured application
path, so these tests remain fast, deterministic, and free of any dependency
on real application state.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from credit_risk.exceptions import ModelNotFoundError
from credit_risk.ml.registry import (
    ModelArtifactMetadata,
    load_model_artifact,
    save_model_artifact,
)


class _StubPipeline:
    """A minimal stand-in for a fitted scikit-learn pipeline."""

    def predict_proba(self, features: object) -> list[list[float]]:
        return [[0.7, 0.3]]


def _sample_metadata() -> ModelArtifactMetadata:
    return ModelArtifactMetadata(
        name="credit-risk-xgboost",
        version="0.0.1-test",
        algorithm="XGBoost",
        dataset_version="laotse/credit-risk-dataset@test",
        feature_version="v0",
        metrics={"roc_auc": 0.5},
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_load_model_artifact_missing_path_raises_model_not_found_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.joblib"

    with pytest.raises(ModelNotFoundError):
        load_model_artifact(missing_path)


def test_save_then_load_model_artifact_round_trips_metadata(tmp_path: Path) -> None:
    artifact_path = tmp_path / "model.joblib"
    metadata = _sample_metadata()

    save_model_artifact(_StubPipeline(), metadata, artifact_path)
    loaded = load_model_artifact(artifact_path)

    assert loaded.metadata == metadata
    assert loaded.pipeline.predict_proba(None) == [[0.7, 0.3]]
