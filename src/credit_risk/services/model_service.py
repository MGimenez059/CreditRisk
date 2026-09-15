"""Load a serving artifact and reconcile its immutable identity with PostgreSQL."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session
from xgboost import XGBClassifier

from credit_risk.db.models.model import ModelMetadata
from credit_risk.exceptions import ModelLoadError, ModelNotFoundError
from credit_risk.ml.preprocessing import RAW_FEATURE_COLUMNS
from credit_risk.ml.registry import ModelArtifact, load_model_artifact
from credit_risk.ml.selection_state import file_sha256
from credit_risk.repositories.interfaces import ModelRepositoryProtocol

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ServingArtifact:
    """Loaded pipeline with its checked on-disk identity."""

    artifact: ModelArtifact
    sha256: str
    metadata_payload: dict[str, object]
    path: str


def load_serving_artifact(path: Path) -> ServingArtifact:
    """Load trusted local files and reject unsupported or changing artifacts."""
    try:
        before = (file_sha256(path), file_sha256(path.with_suffix(".json")))
        artifact = load_model_artifact(path)
        after = (file_sha256(path), file_sha256(path.with_suffix(".json")))
    except FileNotFoundError as err:
        raise ModelNotFoundError("Configured model artifact or sidecar is missing.") from err
    except OSError as err:
        raise ModelLoadError("Configured model files could not be read.") from err
    if before != after:
        raise ModelLoadError("Artifact changed while being loaded; deploy immutable files.")
    pipeline = artifact.pipeline
    if (
        not isinstance(pipeline, Pipeline)
        or not isinstance(pipeline.named_steps.get("model"), XGBClassifier)
        or artifact.metadata.calibration != "none"
        or artifact.metadata.algorithm not in {"XGBClassifier", "XGBoost"}
        or list(getattr(pipeline, "feature_names_in_", [])) != RAW_FEATURE_COLUMNS
        or list(getattr(pipeline, "classes_", [])) != [0, 1]
    ):
        raise ModelLoadError("Serving requires the raw-input uncalibrated binary XGBoost pipeline.")
    frozen_path = path.parent / "frozen.json"
    if frozen_path.exists():
        try:
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            if (frozen["artifact_sha256"], frozen["metadata_sha256"]) != before:
                raise ModelLoadError("Configured artifact differs from its frozen selection.")
        except (OSError, ValueError, KeyError, TypeError) as err:
            raise ModelLoadError("Frozen selection manifest is invalid.") from err
    payload = asdict(artifact.metadata)
    payload["trained_at"] = artifact.metadata.trained_at.isoformat()
    logger.info(
        "model_loaded", model_name=artifact.metadata.name, model_version=artifact.metadata.version
    )
    return ServingArtifact(artifact, before[0], payload, str(path.resolve()))


class ModelService:
    """Keep the configured artifact and registered identity consistent."""

    def __init__(self, session: Session, repository: ModelRepositoryProtocol, path: Path) -> None:
        """Share the caller's session and model repository."""
        self.session = session
        self.repository = repository
        self.path = path

    def active(self) -> ModelArtifact:
        """Register/activate the configured artifact and return its actual metadata."""
        loaded = load_serving_artifact(self.path)
        with self.session.begin():
            self.resolve(loaded)
        return loaded.artifact

    def resolve(self, loaded: ServingArtifact) -> ModelMetadata:
        """Resolve and activate inside the caller's transaction; reject version reuse."""
        self.repository.lock_registry()
        metadata = loaded.artifact.metadata
        row = self.repository.get_by_name_version(metadata.name, metadata.version)
        expected = {
            "algorithm": metadata.algorithm,
            "training_dataset": metadata.dataset_version,
            "feature_version": metadata.feature_version,
            "trained_at": metadata.trained_at,
            "roc_auc": metadata.metrics.get("roc_auc"),
            "pr_auc": metadata.metrics.get("pr_auc"),
            "f1": metadata.metrics.get("f1"),
            "brier_score": metadata.metrics.get("brier_score"),
            "artifact_sha256": loaded.sha256,
            "metadata_payload": loaded.metadata_payload,
        }
        if row is None:
            row = self.repository.add(
                ModelMetadata(
                    name=metadata.name,
                    version=metadata.version,
                    artifact_path=loaded.path,
                    is_active=False,
                    **expected,
                )
            )
        elif any(getattr(row, key) != value for key, value in expected.items()):
            raise ModelLoadError("Registered model identity differs from the configured artifact.")
        row.artifact_path = loaded.path
        self.repository.activate(row)
        return row
