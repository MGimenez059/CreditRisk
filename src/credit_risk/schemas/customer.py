"""Pydantic schemas for the customer resource.

Not yet used by any route — see `api/routes/customers.py`. Defined now so
the Phase 1 schema layer matches SPECS.md §4's file structure.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

HomeOwnership = Literal["RENT", "OWN", "MORTGAGE", "OTHER"]


class CustomerCreate(BaseModel):
    """Fields required to register a new customer."""

    age: int = Field(..., gt=0, lt=120)
    income: float = Field(..., ge=0)
    employment_years: float = Field(..., ge=0)
    home_ownership: HomeOwnership


class CustomerRead(CustomerCreate):
    """A persisted customer, as returned by the API."""

    id: UUID
