"""Reads and writes versioned model artifacts.

An "artifact" is the full inference pipeline (preprocessing + feature
engineering + model) serialized as a single `joblib` file, paired with a
JSON metadata sidecar of the same name — per CODESTYLE.md §14, this makes
training-serving skew structurally impossible and keeps every artifact
traceable to the data and code that produced it.

This module has no FastAPI or database dependency, per CODESTYLE.md §3.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib

from credit_risk.exceptions import ModelLoadError, ModelNotFoundError
from credit_risk.ml.protocols import FittedPipeline


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """Metadata persisted alongside a trained model artifact.

    Attributes:
        name: Registered model name (e.g. "credit-risk-xgboost").
        version: Semantic version of this artifact.
        algorithm: Human-readable algorithm identifier (e.g. "XGBoost").
        dataset_version: Identifier of the training dataset snapshot used.
        feature_version: Identifier of the feature engineering version used.
        metrics: Evaluation metrics recorded at training time.
        trained_at: UTC timestamp when training completed.
    """

    name: str
    version: str
    algorithm: str
    dataset_version: str
    feature_version: str
    metrics: dict[str, float]
    trained_at: datetime


@dataclass(frozen=True)
class ModelArtifact:
    """An in-memory, ready-to-use model artifact."""

    pipeline: FittedPipeline
    metadata: ModelArtifactMetadata


def _metadata_path(artifact_path: Path) -> Path:
    """Return the sidecar metadata path for a given artifact path."""
    return artifact_path.with_suffix(".json")


def save_model_artifact(
    pipeline: FittedPipeline,
    metadata: ModelArtifactMetadata,
    artifact_path: Path,
) -> None:
    """Serialize a trained pipeline and its metadata to disk.

    Args:
        pipeline: A fitted scikit-learn-compatible pipeline, including all
            preprocessing and feature engineering steps.
        metadata: Metadata describing this training run.
        artifact_path: Destination path for the `.joblib` file. The metadata
            sidecar is written next to it with a `.json` extension.
    """
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_path)

    payload = asdict(metadata)
    payload["trained_at"] = metadata.trained_at.isoformat()
    _metadata_path(artifact_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model_artifact(artifact_path: Path) -> ModelArtifact:
    """Load a trained pipeline and its metadata from disk.

    Args:
        artifact_path: Path to the `.joblib` file, as configured by
            `Settings.model_path`.

    Returns:
        The deserialized pipeline paired with its metadata.

    Raises:
        ModelNotFoundError: If no artifact or metadata sidecar exists at
            the given path.
        ModelLoadError: If either file exists but fails to deserialize.
    """
    metadata_path = _metadata_path(artifact_path)

    if not artifact_path.exists() or not metadata_path.exists():
        raise ModelNotFoundError(
            f"No model artifact found at '{artifact_path}'. "
            "Run `python scripts/train_model.py` to produce one."
        )

    try:
        pipeline = joblib.load(artifact_path)
        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_metadata["trained_at"] = datetime.fromisoformat(raw_metadata["trained_at"])
        metadata = ModelArtifactMetadata(**raw_metadata)
    except (OSError, ValueError, TypeError, KeyError) as err:
        raise ModelLoadError(f"Failed to load model artifact at '{artifact_path}'") from err

    return ModelArtifact(pipeline=pipeline, metadata=metadata)
