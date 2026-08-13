"""Domain exception hierarchy for CreditRisk.

Every exception raised by application code (outside of framework and
third-party internals) must derive from :class:`CreditRiskError`. The API
layer maps these to HTTP responses via a centralized exception handler — see
``credit_risk.api.exception_handlers`` — so services and repositories should
never construct HTTP responses themselves.
"""


class CreditRiskError(Exception):
    """Base class for all domain-specific exceptions in this project."""


class EntityNotFoundError(CreditRiskError):
    """Raised when a requested entity does not exist in storage."""


class ModelNotFoundError(CreditRiskError):
    """Raised when no trained model artifact is available at the configured path."""


class ModelLoadError(CreditRiskError):
    """Raised when a model artifact exists but fails to deserialize."""


class InvalidFeatureSchemaError(CreditRiskError):
    """Raised when input features do not match the schema the model was trained on."""


class DataValidationError(CreditRiskError):
    """Raised when raw or processed data fails validation during the data pipeline."""


class ExplanationError(CreditRiskError):
    """Raised when a SHAP explanation cannot be computed for a prediction."""


class BatchSizeExceededError(CreditRiskError):
    """Raised when a batch prediction request exceeds the configured maximum size."""
