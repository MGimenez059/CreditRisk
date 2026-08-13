"""Pydantic schemas describing the active model, returned by `/models/active`."""

from pydantic import BaseModel


class ActiveModelResponse(BaseModel):
    """Metadata about the model currently serving predictions."""

    name: str
    version: str
    algorithm: str
    roc_auc: float
