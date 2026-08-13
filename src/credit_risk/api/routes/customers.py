"""Customer endpoints.

Note:
    This router is implemented but deliberately **not** mounted on the
    FastAPI app yet — see `main.py`. SPECS.md §4 lists `customers.py` as
    part of the Phase 1 repository structure, and ROADMAP.md bundles the
    live `customers` endpoint into Phase 6 (Backend) alongside the rest of
    the persistence layer. The handlers below are real and tested-ready,
    not stubs, so wiring them in is a one-line change in `main.py` once
    Phase 6 starts — not a rewrite.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from credit_risk.db.models.customer import Customer as CustomerModel
from credit_risk.db.session import get_db
from credit_risk.exceptions import EntityNotFoundError
from credit_risk.repositories.customer import SQLAlchemyCustomerRepository
from credit_risk.schemas.customer import CustomerCreate, CustomerRead

router = APIRouter(prefix="/customers", tags=["customers"])


def _get_customer_repository(
    session: Annotated[Session, Depends(get_db)],
) -> SQLAlchemyCustomerRepository:
    """Provide a request-scoped customer repository."""
    return SQLAlchemyCustomerRepository(session)


@router.post("", response_model=CustomerRead)
async def create_customer(
    payload: CustomerCreate,
    repository: Annotated[SQLAlchemyCustomerRepository, Depends(_get_customer_repository)],
) -> CustomerRead:
    """Register a new customer."""
    customer = repository.add(CustomerModel(**payload.model_dump()))
    return CustomerRead(id=customer.id, **payload.model_dump())


@router.get("/{customer_id}", response_model=CustomerRead)
async def get_customer(
    customer_id: UUID,
    repository: Annotated[SQLAlchemyCustomerRepository, Depends(_get_customer_repository)],
) -> CustomerRead:
    """Fetch a customer by id.

    Raises:
        EntityNotFoundError: If no customer with `customer_id` exists.
            Translated to `HTTP 404` by the centralized exception handler.
    """
    customer = repository.get_by_id(customer_id)
    if customer is None:
        raise EntityNotFoundError(f"No customer found with id '{customer_id}'.")
    return CustomerRead(
        id=customer.id,
        age=customer.age,
        income=float(customer.income),
        employment_years=float(customer.employment_years),
        home_ownership=customer.home_ownership.value,
    )
