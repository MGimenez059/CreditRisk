"""Unit tests that construct each ORM model with the keyword arguments used
elsewhere in the codebase (`services/`, `api/routes/`).

These do not touch a database — plain object construction only — but they
would have caught the `Loan.loan_amount` → `amount` /
`CreditHistory.has_previous_default` → `previous_defaults` field renames
made to align with SPECS.md §6 before those mismatches reached a repository
or a running server.
"""

import uuid
from datetime import UTC, datetime

from credit_risk.db.models.credit_history import CreditHistory
from credit_risk.db.models.customer import Customer
from credit_risk.db.models.enums import HomeOwnership, LoanGrade, LoanIntent, RiskLevel
from credit_risk.db.models.loan import Loan
from credit_risk.db.models.model import ModelMetadata
from credit_risk.db.models.prediction import Prediction


def test_customer_constructs_with_canonical_field_names() -> None:
    customer = Customer(
        age=34,
        income=1_450_000,
        employment_years=6.0,
        home_ownership=HomeOwnership.RENT,
    )

    assert customer.age == 34
    assert customer.home_ownership is HomeOwnership.RENT


def test_credit_history_previous_defaults_is_an_integer_count() -> None:
    history = CreditHistory(
        customer_id=uuid.uuid4(),
        credit_history_years=7,
        previous_defaults=0,
        late_payments=1,
        credit_utilization=0.42,
        active_credit_lines=4,
    )

    assert history.previous_defaults == 0
    assert isinstance(history.previous_defaults, int)


def test_loan_constructs_with_canonical_field_names() -> None:
    loan = Loan(
        customer_id=uuid.uuid4(),
        amount=500_000,
        interest_rate=12.5,
        term_months=36,
        purpose=LoanIntent.PERSONAL,
        grade=LoanGrade.A,
        loan_status=0,
        loan_percent_income=0.34,
    )

    assert loan.amount == 500_000
    assert loan.purpose is LoanIntent.PERSONAL
    assert loan.loan_status == 0


def test_prediction_constructs_with_model_id_and_customer_id() -> None:
    prediction = Prediction(
        customer_id=None,
        model_id=uuid.uuid4(),
        request_id=str(uuid.uuid4()),
        default_probability=0.183,
        risk_score=18,
        risk_level=RiskLevel.LOW,
        prediction_version="1.0.0",
        latency_ms=12.3,
        explanation=[],
    )

    assert prediction.customer_id is None
    assert prediction.risk_level is RiskLevel.LOW


def test_model_metadata_constructs_with_spec_column_names() -> None:
    metadata = ModelMetadata(
        name="credit-risk-xgboost",
        version="1.0.0",
        algorithm="XGBoost",
        training_dataset="laotse/credit-risk-dataset@v1",
        feature_version="v0",
        roc_auc=0.82,
        pr_auc=0.65,
        f1=0.58,
        brier_score=0.09,
        artifact_path="models/credit_risk_xgboost_v1.joblib",
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        is_active=True,
    )

    assert metadata.roc_auc == 0.82
    assert metadata.artifact_path == "models/credit_risk_xgboost_v1.joblib"
