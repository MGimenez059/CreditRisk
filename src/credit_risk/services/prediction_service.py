"""Orchestrates the end-to-end single-prediction flow.

Route handlers depend only on this service, never on `credit_risk.ml`,
`credit_risk.db`, or a repository directly, per the layering rules in
CODESTYLE.md §3.
"""

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import structlog

from credit_risk.config.settings import Settings, get_settings
from credit_risk.db.models.model import ModelMetadata
from credit_risk.db.models.prediction import Prediction
from credit_risk.ml import predict as ml_predict
from credit_risk.ml import registry as ml_registry
from credit_risk.ml.registry import ModelArtifactMetadata
from credit_risk.repositories.interfaces import (
    ModelRepositoryProtocol,
    PredictionRepositoryProtocol,
)
from credit_risk.schemas.prediction import PredictionRequest
from credit_risk.services.explanation_service import ExplanationService, FeatureContribution
from credit_risk.services.risk_service import calculate_risk_score

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Domain-level result of a single prediction, independent of the API schema."""

    default_probability: float
    risk_score: int
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    model_name: str
    model_version: str
    explanation: list[FeatureContribution]


class PredictionService:
    """Predicts default probability for a single loan application."""

    def __init__(
        self,
        prediction_repository: PredictionRepositoryProtocol,
        model_repository: ModelRepositoryProtocol,
        explanation_service: ExplanationService | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Wire the service with its collaborators.

        Args:
            prediction_repository: Storage for the resulting prediction.
            model_repository: The model registry (`models` table), used to
                resolve the `model_id` FK that SPECS.md §6 requires every
                prediction to carry.
            explanation_service: Computes per-prediction SHAP contributions.
                Defaults to a new `ExplanationService` instance.
            settings: Application settings. Defaults to the process-wide
                cached settings.
        """
        self._prediction_repository = prediction_repository
        self._model_repository = model_repository
        self._explanation_service = explanation_service or ExplanationService()
        self._settings = settings or get_settings()

    def predict(self, request: PredictionRequest) -> PredictionResult:
        """Run the full inference pipeline for one loan application.

        Args:
            request: A validated loan application.

        Returns:
            The domain-level prediction result, already persisted.

        Raises:
            ModelNotFoundError: If no trained model artifact exists at the
                configured `MODEL_PATH`.
            ModelLoadError: If the artifact exists but fails to deserialize.
        """
        request_id = str(uuid.uuid4())
        started_at = time.perf_counter()

        artifact = ml_registry.load_model_artifact(Path(self._settings.model_path))
        model_row = self._resolve_model_metadata(artifact.metadata)
        features = _to_feature_frame(request)

        default_probability = ml_predict.predict(artifact.pipeline, features)
        risk = calculate_risk_score(default_probability)
        explanation = self._explanation_service.explain(features)

        latency_ms = (time.perf_counter() - started_at) * 1000

        self._prediction_repository.add(
            Prediction(
                customer_id=None,
                model_id=model_row.id,
                request_id=request_id,
                default_probability=default_probability,
                risk_score=risk.value,
                risk_level=risk.level,
                prediction_version=model_row.version,
                latency_ms=latency_ms,
                explanation=[contribution.__dict__ for contribution in explanation],
            )
        )

        logger.info(
            "prediction_served",
            request_id=request_id,
            model_version=model_row.version,
            latency_ms=latency_ms,
            risk_level=risk.level,
        )

        return PredictionResult(
            default_probability=default_probability,
            risk_score=risk.value,
            risk_level=risk.level,
            model_name=model_row.name,
            model_version=model_row.version,
            explanation=explanation,
        )

    def _resolve_model_metadata(self, artifact_metadata: ModelArtifactMetadata) -> ModelMetadata:
        """Return the `models` table row matching a loaded artifact, registering it if needed.

        The `ml.registry` sidecar JSON is the source of truth for "what the
        currently configured artifact is"; this method makes sure a
        matching row exists in the `models` table so `Prediction.model_id`
        (SPECS.md §6) always has something to point at, without requiring a
        separate manual "register this model" step before the first
        prediction can be served. Auto-registered rows are marked active;
        promoting a different model later is a `ModelRepositoryProtocol`
        concern, not this method's.
        """
        existing = self._model_repository.get_by_name_version(
            artifact_metadata.name, artifact_metadata.version
        )
        if existing is not None:
            return existing

        return self._model_repository.add(
            ModelMetadata(
                name=artifact_metadata.name,
                version=artifact_metadata.version,
                algorithm=artifact_metadata.algorithm,
                training_dataset=artifact_metadata.dataset_version,
                feature_version=artifact_metadata.feature_version,
                roc_auc=artifact_metadata.metrics.get("roc_auc"),
                pr_auc=artifact_metadata.metrics.get("pr_auc"),
                f1=artifact_metadata.metrics.get("f1"),
                brier_score=artifact_metadata.metrics.get("brier_score"),
                artifact_path=str(self._settings.model_path),
                trained_at=artifact_metadata.trained_at,
                is_active=True,
            )
        )


def _to_feature_frame(request: PredictionRequest) -> pd.DataFrame:
    """Convert a validated request into the single-row frame the pipeline expects."""
    return pd.DataFrame([request.model_dump()])
