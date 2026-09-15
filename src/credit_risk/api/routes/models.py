"""Model registry endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends

from credit_risk.api.dependencies import get_model_service
from credit_risk.schemas.model import ActiveModelResponse
from credit_risk.services.model_service import ModelService

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/active", response_model=ActiveModelResponse)
def get_active_model(
    service: Annotated[ModelService, Depends(get_model_service)],
) -> ActiveModelResponse:
    """Return metadata for the model artifact currently serving predictions.

    Raises:
        ModelNotFoundError: If no trained model artifact exists at the
            configured `MODEL_PATH`. Translated to `HTTP 404` by the
            centralized exception handler.
    """
    artifact = service.active()
    return ActiveModelResponse(
        threshold=artifact.metadata.threshold,
        calibration=artifact.metadata.calibration,
        metrics_partition=artifact.metadata.metrics_partition,
        name=artifact.metadata.name,
        version=artifact.metadata.version,
        algorithm=artifact.metadata.algorithm,
        roc_auc=artifact.metadata.metrics.get("roc_auc"),
    )
