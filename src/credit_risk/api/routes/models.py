"""Model registry endpoints."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends

from credit_risk.config.settings import Settings, get_settings
from credit_risk.ml import registry as ml_registry
from credit_risk.schemas.model import ActiveModelResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/active", response_model=ActiveModelResponse)
async def get_active_model(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ActiveModelResponse:
    """Return metadata for the model artifact currently serving predictions.

    Raises:
        ModelNotFoundError: If no trained model artifact exists at the
            configured `MODEL_PATH`. Translated to `HTTP 404` by the
            centralized exception handler.
    """
    artifact = ml_registry.load_model_artifact(Path(settings.model_path))
    return ActiveModelResponse(
        name=artifact.metadata.name,
        version=artifact.metadata.version,
        algorithm=artifact.metadata.algorithm,
        roc_auc=artifact.metadata.metrics.get("roc_auc", 0.0),
    )
