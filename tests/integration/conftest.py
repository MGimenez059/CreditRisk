"""Explicit test database opt-in; each test owns an isolated PostgreSQL schema."""

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from credit_risk.api.dependencies import get_settings
from credit_risk.config.settings import Settings
from credit_risk.db.session import get_db
from credit_risk.main import app
from credit_risk.ml.registry import ModelArtifactMetadata, save_model_artifact
from credit_risk.ml.train import build_candidate_pipeline, split_dataset
from tests.unit.test_train import _synthetic_dataset


@pytest.fixture(scope="session")
def artifact_path(tmp_path_factory):
    split = split_dataset(_synthetic_dataset(200), "loan_status")
    pipeline = build_candidate_pipeline("xgboost")
    pipeline.fit(split.x_train, split.y_train)
    metadata = ModelArtifactMetadata(
        name="synthetic-xgboost",
        version="test-v1",
        algorithm="XGBClassifier",
        dataset_version="sha256:" + "a" * 64,
        feature_version="v1",
        metrics={"roc_auc": 0.8},
        trained_at=datetime.now(UTC),
    )
    path = tmp_path_factory.mktemp("serving") / "model.joblib"
    save_model_artifact(pipeline, metadata, path)
    return path


@pytest.fixture
def database():
    raw_url = os.environ.get("TEST_DATABASE_URL")
    if raw_url is None:
        pytest.skip("Set TEST_DATABASE_URL to a dedicated PostgreSQL database ending in _test")
    url = make_url(raw_url)
    if url.get_backend_name() != "postgresql" or not (url.database or "").endswith("_test"):
        pytest.fail("TEST_DATABASE_URL must target a dedicated PostgreSQL database ending in _test")
    schema = "test_" + uuid.uuid4().hex
    admin = create_engine(url)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    try:
        with engine.begin() as connection:
            config = Config(str(Path("alembic.ini")))
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture
def api(database, artifact_path):
    settings = Settings(
        _env_file=None, app_env="test", model_path=str(artifact_path), max_batch_size=3
    )

    def sessions() -> Iterator[Session]:
        with Session(database, expire_on_commit=False) as session:
            yield session

    app.dependency_overrides[get_db] = sessions
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, settings
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def request_payload():
    return {
        "age": 34,
        "income": 60000,
        "employment_years": None,
        "home_ownership": "RENT",
        "loan_amount": 12000,
        "interest_rate": None,
        "loan_intent": "PERSONAL",
        "credit_history_years": 7,
        "previous_defaults": 1,
    }
