"""Closed-set string enums backed at the database level.

These enums back PostgreSQL `ENUM` columns so invalid values are rejected by
the database, not only by application code — see CODESTYLE.md §13. Pydantic
schemas define their own `Literal` aliases with the same values rather than
importing from this module, to keep the `schemas/` layer independent of
`db/` per the layering rules in CODESTYLE.md §3.
"""

import enum


class HomeOwnership(enum.StrEnum):
    """Type of home ownership reported by the applicant."""

    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"


class LoanIntent(enum.StrEnum):
    """Stated purpose of the loan."""

    PERSONAL = "PERSONAL"
    EDUCATION = "EDUCATION"
    MEDICAL = "MEDICAL"
    VENTURE = "VENTURE"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"


class LoanGrade(enum.StrEnum):
    """Lender-assigned credit grade, A (best) through G (worst)."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class RiskLevel(enum.StrEnum):
    """Presentation-layer risk bucket derived from `default_probability`.

    See `services/risk_service.py` for the thresholds that produce this
    value; they are not repeated here to avoid two sources of truth.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
