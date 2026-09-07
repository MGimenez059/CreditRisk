"""Customer endpoints.

Note:
    This router is implemented but deliberately **not** mounted on the
    FastAPI app yet — see `main.py`. SPECS.md §4 lists `customers.py` as
    part of the initial scaffold. Customer CRUD is now outside the MVP.
    Before enabling it, add a customer service and integration tests;
    mounting the existing router alone is not a completion criterion.
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
def create_customer(
    payload: CustomerCreate,
    repository: Annotated[SQLAlchemyCustomerRepository, Depends(_get_customer_repository)],
) -> CustomerRead:
    """Register a new customer."""
    customer = repository.add(CustomerModel(**payload.model_dump()))
    return CustomerRead(id=customer.id, **payload.model_dump())


@router.get("/{customer_id}", response_model=CustomerRead)
def get_customer(
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
