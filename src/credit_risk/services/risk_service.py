"""Converts a model-predicted default probability into a presentation risk score."""

from dataclasses import dataclass
from typing import Literal

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]

_LOW_MAX = 30
_MEDIUM_MAX = 70


@dataclass(frozen=True)
class RiskScore:
    """A risk score derived from a default probability.

    Attributes:
        value: Numeric score in [0, 100].
        level: Bucket the score falls into (LOW, MEDIUM, HIGH).
    """

    value: int
    level: RiskLevel


def calculate_risk_score(default_probability: float) -> RiskScore:
    """Convert a model-predicted default probability into a risk score.

    Args:
        default_probability: Predicted probability of default, in [0, 1].

    Returns:
        A RiskScore containing the numeric score (0-100) and the
        corresponding risk level (LOW, MEDIUM, HIGH).

    Raises:
        ValueError: If default_probability is outside [0, 1].
    """
    if not 0.0 <= default_probability <= 1.0:
        raise ValueError(f"default_probability must be in [0, 1], got {default_probability}")

    score = round(default_probability * 100)
    level = _bucket_for(score)
    return RiskScore(value=score, level=level)


def _bucket_for(score: int) -> RiskLevel:
    """Map a numeric score to its risk bucket using the thresholds in README.md."""
    if score <= _LOW_MAX:
        return "LOW"
    if score <= _MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"
