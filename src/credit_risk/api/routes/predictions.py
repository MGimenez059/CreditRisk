"""Prediction endpoints: single and batch."""

from typing import Annotated

from fastapi import APIRouter, Depends

from credit_risk.api.dependencies import get_prediction_service
from credit_risk.config.settings import Settings, get_settings
from credit_risk.exceptions import BatchSizeExceededError
from credit_risk.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureContribution,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
)
from credit_risk.services.prediction_service import PredictionResult, PredictionService

router = APIRouter(prefix="/predictions", tags=["predictions"])


def _to_response(result: PredictionResult) -> PredictionResponse:
    """Map a domain-level prediction result onto its API response schema."""
    return PredictionResponse(
        default_probability=result.default_probability,
        risk_score=result.risk_score,
        risk_level=result.risk_level,
        model=ModelInfo(name=result.model_name, version=result.model_version),
        explanation=[
            FeatureContribution(
                feature=contribution.feature,
                impact=contribution.impact,
                direction=contribution.direction,
            )
            for contribution in result.explanation
        ],
    )


@router.post("", response_model=PredictionResponse)
async def create_prediction(
    payload: PredictionRequest,
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> PredictionResponse:
    """Predict the default probability for a single loan application.

    Raises:
        ModelNotFoundError: If no trained model artifact exists yet.
            Translated to `HTTP 404` by the centralized exception handler.
    """
    result = prediction_service.predict(payload)
    return _to_response(result)


@router.post("/batch", response_model=BatchPredictionResponse)
async def create_batch_prediction(
    payload: BatchPredictionRequest,
    prediction_service: Annotated[PredictionService, Depends(get_prediction_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> BatchPredictionResponse:
    """Predict default probability for a batch of loan applications.

    Raises:
        BatchSizeExceededError: If `payload.items` exceeds `MAX_BATCH_SIZE`.
            Translated to `HTTP 422` by the centralized exception handler.
        ModelNotFoundError: If no trained model artifact exists yet.
    """
    if len(payload.items) > settings.max_batch_size:
        raise BatchSizeExceededError(
            f"Batch size {len(payload.items)} exceeds the maximum of {settings.max_batch_size}."
        )

    results = [prediction_service.predict(item) for item in payload.items]
    return BatchPredictionResponse(results=[_to_response(result) for result in results])
