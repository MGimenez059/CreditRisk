"""Create persistence schema

Revision ID: 0001
Revises:
Create Date: 2026-09-14 16:40:23.348740
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create registry/predictions and retained FK-related scaffolds."""
    op.create_table(
        "customers",
        sa.Column("age", sa.Integer(), nullable=False),
        sa.Column("income", sa.Numeric(precision=14, scale=2), nullable=False),
        sa.Column(
            "home_ownership",
            sa.Enum("RENT", "OWN", "MORTGAGE", "OTHER", name="home_ownership_enum"),
            nullable=False,
        ),
        sa.Column("employment_years", sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("age > 0 AND age < 120", name="ck_customers_age_range"),
        sa.CheckConstraint("income >= 0", name="ck_customers_income_non_negative"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "models",
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("algorithm", sa.String(length=64), nullable=False),
        sa.Column("training_dataset", sa.String(length=128), nullable=False),
        sa.Column("feature_version", sa.String(length=64), nullable=False),
        sa.Column("roc_auc", sa.Float(), nullable=True),
        sa.Column("pr_auc", sa.Float(), nullable=True),
        sa.Column("f1", sa.Float(), nullable=True),
        sa.Column("brier_score", sa.Float(), nullable=True),
        sa.Column("artifact_path", sa.String(length=512), nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("metadata_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "version", name="uq_models_name_version"),
    )
    op.create_index(
        "uq_models_active",
        "models",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )
    op.create_table(
        "credit_histories",
        sa.Column("customer_id", sa.UUID(), nullable=False),
        sa.Column("credit_history_years", sa.Integer(), nullable=False),
        sa.Column("previous_defaults", sa.Integer(), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_credit_histories_customer_id"), "credit_histories", ["customer_id"], unique=True
    )
    op.create_table(
        "loans",
        sa.Column("customer_id", sa.UUID(), nullable=False),
        sa.Column("amount", sa.Numeric(precision=14, scale=2), nullable=False),
        sa.Column("interest_rate", sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column(
            "purpose",
            sa.Enum(
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION",
                name="loan_intent_enum",
            ),
            nullable=False,
        ),
        sa.Column(
            "grade",
            sa.Enum("A", "B", "C", "D", "E", "F", "G", name="loan_grade_enum"),
            nullable=True,
        ),
        sa.Column("loan_status", sa.Integer(), nullable=True),
        sa.Column("loan_percent_income", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("amount > 0", name="ck_loans_amount_positive"),
        sa.CheckConstraint("interest_rate >= 0", name="ck_loans_interest_rate_non_negative"),
        sa.CheckConstraint("loan_status IN (0, 1)", name="ck_loans_status_binary"),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_loans_customer_id"), "loans", ["customer_id"], unique=False)
    op.create_table(
        "predictions",
        sa.Column("customer_id", sa.UUID(), nullable=True),
        sa.Column("model_id", sa.UUID(), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("default_probability", sa.Float(), nullable=False),
        sa.Column("risk_score", sa.Integer(), nullable=False),
        sa.Column(
            "risk_level", sa.Enum("LOW", "MEDIUM", "HIGH", name="risk_level_enum"), nullable=False
        ),
        sa.Column("prediction_version", sa.String(length=32), nullable=False),
        sa.Column("latency_ms", sa.Numeric(precision=10, scale=3), nullable=False),
        sa.Column("explanation", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "default_probability >= 0 AND default_probability <= 1",
            name="ck_predictions_probability",
        ),
        sa.CheckConstraint("latency_ms >= 0", name="ck_predictions_latency"),
        sa.CheckConstraint("risk_score >= 0 AND risk_score <= 100", name="ck_predictions_score"),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_predictions_customer_id"), "predictions", ["customer_id"], unique=False
    )
    op.create_index(op.f("ix_predictions_model_id"), "predictions", ["model_id"], unique=False)
    op.create_index(op.f("ix_predictions_request_id"), "predictions", ["request_id"], unique=False)


def downgrade() -> None:
    """Drop tables and their PostgreSQL enum types."""
    op.drop_index(op.f("ix_predictions_request_id"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_model_id"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_customer_id"), table_name="predictions")
    op.drop_table("predictions")
    op.drop_index(op.f("ix_loans_customer_id"), table_name="loans")
    op.drop_table("loans")
    op.drop_index(op.f("ix_credit_histories_customer_id"), table_name="credit_histories")
    op.drop_table("credit_histories")
    op.drop_index("uq_models_active", table_name="models", postgresql_where=sa.text("is_active"))
    op.drop_table("models")
    op.drop_table("customers")
    for name in ("risk_level_enum", "loan_grade_enum", "loan_intent_enum", "home_ownership_enum"):
        sa.Enum(name=name).drop(op.get_bind(), checkfirst=True)
