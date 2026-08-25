"""Pydantic schemas for the prediction endpoints.

These are the only shapes that cross the API boundary — route handlers never
accept or return ORM models or raw dicts, per CODESTYLE.md §12. `Literal` is
used for every closed-set string field instead of bare `str`, per
CODESTYLE.md §6.
"""

from typing import Literal

from pydantic import BaseModel, Field

HomeOwnership = Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
LoanIntent = Literal[
    "PERSONAL",
    "EDUCATION",
    "MEDICAL",
    "VENTURE",
    "HOMEIMPROVEMENT",
    "DEBTCONSOLIDATION",
]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class PredictionRequest(BaseModel):
    """A single loan application submitted for a default-probability prediction."""

    age: int = Field(..., gt=0, lt=120, description="Applicant age in years.")
    income: float = Field(..., ge=0, description="Annual income, in the dataset's currency unit.")
    employment_years: float = Field(..., ge=0, description="Years in current employment.")
    home_ownership: HomeOwnership
    loan_amount: float = Field(..., gt=0)
    interest_rate: float = Field(..., ge=0, description="Annual interest rate, as a percentage.")
    term_months: int | None = Field(default=None, gt=0)
    loan_intent: LoanIntent
    credit_history_years: int = Field(..., ge=0)
    late_payments: int | None = Field(default=None, ge=0)
    previous_defaults: int | None = Field(default=None, ge=0)
    credit_utilization: float | None = Field(default=None, ge=0, le=1)
    active_credit_lines: int | None = Field(default=None, ge=0)


class ModelInfo(BaseModel):
    """Identifies the exact model artifact that produced a prediction."""

    name: str
    version: str


class FeatureContribution(BaseModel):
    """A single feature's SHAP contribution to one prediction."""

    feature: str
    impact: float
    direction: Literal["positive", "negative"]


class PredictionResponse(BaseModel):
    """Result of a single default-probability prediction."""

    default_probability: float = Field(..., ge=0, le=1)
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    model: ModelInfo
    explanation: list[FeatureContribution]


class BatchPredictionResponse(BaseModel):
    """Results for a batch prediction request, one entry per input item.

    The request body itself is a bare `list[PredictionRequest]` — see
    `api/routes/predictions.create_batch_prediction` — per SPECS.md's Batch
    prediction section ("Input: JSON array initially"). The response stays
    wrapped in a named field rather than mirroring that as a bare array,
    since an unconstrained response shape is free to grow (e.g. pagination
    metadata) without a breaking change later.
    """

    results: list[PredictionResponse]
