"""Compute predictions and commit each single request or complete batch atomically."""

import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import structlog
from sqlalchemy.orm import Session

from credit_risk.config.settings import Settings
from credit_risk.db.models.prediction import Prediction
from credit_risk.exceptions import BatchSizeExceededError
from credit_risk.ml import predict as ml_predict
from credit_risk.repositories.interfaces import (
    ModelRepositoryProtocol,
    PredictionRepositoryProtocol,
)
from credit_risk.schemas.prediction import PredictionRequest
from credit_risk.services.explanation_service import Explanation, ExplanationService
from credit_risk.services.model_service import ModelService, load_serving_artifact
from credit_risk.services.risk_service import calculate_risk_score

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Domain result independent of the HTTP schema."""

    default_probability: float
    risk_score: int
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    model_name: str
    model_version: str
    explanation: Explanation


class PredictionService:
    """Use one artifact per request and one transaction for all persistence."""

    def __init__(
        self,
        prediction_repository: PredictionRepositoryProtocol,
        model_repository: ModelRepositoryProtocol,
        session: Session,
        settings: Settings,
        explanation_service: ExplanationService | None = None,
    ) -> None:
        """Wire collaborators sharing the same request-scoped database session."""
        self._prediction_repository = prediction_repository
        self._session = session
        self._settings = settings
        self._models = ModelService(session, model_repository, Path(settings.model_path))
        self._explanation_service = explanation_service or ExplanationService()

    def predict(self, request: PredictionRequest) -> PredictionResult:
        """Return a result only after its database transaction commits."""
        return self.predict_batch([request])[0]

    def predict_batch(self, requests: list[PredictionRequest]) -> list[PredictionResult]:
        """Preserve order; roll back the entire batch on any persistence failure."""
        if not 1 <= len(requests) <= self._settings.max_batch_size:
            raise BatchSizeExceededError(
                f"Batch size must be between 1 and {self._settings.max_batch_size}."
            )
        loaded = load_serving_artifact(Path(self._settings.model_path))
        results = []
        latencies = []
        for request in requests:
            started = time.perf_counter()
            frame = _to_feature_frame(request)
            probability = ml_predict.predict(loaded.artifact.pipeline, frame)
            risk = calculate_risk_score(probability)
            explanation = self._explanation_service.explain(loaded.artifact.pipeline, frame)
            results.append(
                PredictionResult(
                    probability,
                    risk.value,
                    risk.level,
                    loaded.artifact.metadata.name,
                    loaded.artifact.metadata.version,
                    explanation,
                )
            )
            latencies.append((time.perf_counter() - started) * 1000)
        # Compute before acquiring the registry lock; commit before responding.
        with self._session.begin():
            model = self._models.resolve(loaded)
            for result, latency in zip(results, latencies, strict=True):
                self._prediction_repository.add(
                    Prediction(
                        customer_id=None,
                        model_id=model.id,
                        request_id=str(uuid.uuid4()),
                        default_probability=result.default_probability,
                        risk_score=result.risk_score,
                        risk_level=result.risk_level,
                        prediction_version=result.model_version,
                        latency_ms=latency,
                        explanation=asdict(result.explanation),
                    )
                )
        logger.info(
            "predictions_committed",
            model_name=loaded.artifact.metadata.name,
            model_version=loaded.artifact.metadata.version,
            count=len(results),
            inference_latency_ms=sum(latencies),
        )
        return results


def _to_feature_frame(request: PredictionRequest) -> pd.DataFrame:
    """Convert a validated request into the single-row frame the pipeline expects."""
    return pd.DataFrame(
        [
            {
                "person_age": request.age,
                "person_income": request.income,
                "person_emp_length": request.employment_years,
                "person_home_ownership": request.home_ownership,
                "loan_amnt": request.loan_amount,
                "loan_int_rate": request.interest_rate,
                "loan_intent": request.loan_intent,
                "cb_person_cred_hist_length": request.credit_history_years,
                "cb_person_default_on_file": "Y" if request.previous_defaults else "N",
            }
        ]
    )
