"""Real PostgreSQL transactions, model identity, HTTP contracts and failure paths."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import event, func, inspect, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from credit_risk.db.models.model import ModelMetadata
from credit_risk.db.models.prediction import Prediction
from credit_risk.exceptions import ExplanationError
from credit_risk.ml.registry import load_model_artifact, save_model_artifact
from credit_risk.repositories.prediction import SQLAlchemyPredictionRepository
from credit_risk.services.explanation_service import Explanation, ExplanationService
from credit_risk.services.model_service import ServingArtifact

pytestmark = pytest.mark.integration


def counts(database):
    with Session(database) as session:
        return tuple(
            session.scalar(select(func.count()).select_from(model))
            for model in (ModelMetadata, Prediction)
        )


def test_single_request_commits_exact_explanation(api, database, request_payload):
    client, _ = api
    response = client.post("/api/v1/predictions", json=request_payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert counts(database) == (1, 1)
    with Session(database) as session:
        prediction = session.scalars(select(Prediction)).one()
        model = session.scalars(select(ModelMetadata)).one()
        assert prediction.model_id == model.id
        assert prediction.explanation == body["explanation"]
        assert prediction.default_probability == body["default_probability"]
        assert prediction.prediction_version == body["model"]["version"]
        assert model.training_dataset == "sha256:" + "a" * 64
        assert len(model.artifact_sha256) == 64
        assert model.is_active
    explanation = body["explanation"]
    total = explanation["base_value"] + sum(x["impact"] for x in explanation["contributions"])
    assert 1 / (1 + np.exp(-total)) == pytest.approx(body["default_probability"], abs=1e-6)
    assert client.get("/api/v1/models/active").json()["version"] == body["model"]["version"]


def test_readiness_checks_real_database_and_artifact_without_writing(api, database):
    client, _ = api

    response = client.get("/ready")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "status": "ready",
        "checks": {"database": "ready", "model": "ready"},
        "model": {"name": "synthetic-xgboost", "version": "test-v1"},
    }
    assert counts(database) == (0, 0)


def test_readiness_rejects_missing_serving_column(api, database):
    client, _ = api
    # The fixture owns this isolated schema; leave all shared databases untouched.
    with database.begin() as connection:
        connection.execute(text("ALTER TABLE predictions DROP COLUMN explanation"))
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"detail": "Service dependencies are unavailable."}
    assert client.get("/health").status_code == 200


def test_batch_commits_in_order_with_one_model_load(api, database, request_payload, monkeypatch):
    import credit_risk.services.prediction_service as module

    original = module.load_serving_artifact
    loads = []

    def load(path) -> ServingArtifact:
        loads.append(path)
        return original(path)

    monkeypatch.setattr(module, "load_serving_artifact", load)
    client, _ = api
    payloads = [request_payload, dict(request_payload, income=20000)]
    body = client.post("/api/v1/predictions/batch", json=payloads).json()
    assert len(body["results"]) == 2
    assert len(loads) == 1
    assert counts(database) == (1, 2)
    for item, result in zip(payloads, body["results"], strict=True):
        assert client.post("/api/v1/predictions", json=item).json() == result


@pytest.mark.parametrize("payload", [[], [{}], [None]])
def test_invalid_batches_do_not_write(api, database, payload):
    client, _ = api
    assert client.post("/api/v1/predictions/batch", json=payload).status_code == 422
    assert counts(database) == (0, 0)


def test_oversized_batch_does_not_write(api, database, request_payload):
    client, _ = api
    assert client.post("/api/v1/predictions/batch", json=[request_payload] * 4).status_code == 422
    assert counts(database) == (0, 0)


@pytest.mark.parametrize("mode,expected", [("missing", 404), ("corrupt", 500), ("metadata", 500)])
def test_artifact_failures_do_not_write(api, database, request_payload, tmp_path, mode, expected):
    client, settings = api
    original = settings.model_path
    path = tmp_path / "model.joblib"
    settings.model_path = str(path)
    if mode != "missing":
        from pathlib import Path

        path.write_bytes(b"corrupt" if mode == "corrupt" else Path(original).read_bytes())
        path.with_suffix(".json").write_text(
            Path(original).with_suffix(".json").read_text() if mode == "corrupt" else "{}"
        )
    assert client.post("/api/v1/predictions", json=request_payload).status_code == expected
    assert counts(database) == (0, 0)


def test_shap_failure_leaves_no_partial_batch(api, database, request_payload, monkeypatch):
    original = ExplanationService.explain
    calls = []

    def explain(self, pipeline, frame) -> Explanation:
        calls.append(1)
        if len(calls) == 2:
            raise ExplanationError("Synthetic failure")
        return original(self, pipeline, frame)

    monkeypatch.setattr(ExplanationService, "explain", explain)
    client, _ = api
    assert client.post("/api/v1/predictions/batch", json=[request_payload] * 2).status_code == 500
    assert counts(database) == (0, 0)


def test_database_failure_rolls_back_model_and_first_prediction(
    api, database, request_payload, monkeypatch
):
    original = SQLAlchemyPredictionRepository.add
    calls = []

    def add(self, prediction) -> Prediction:
        calls.append(1)
        if len(calls) == 2:
            prediction.default_probability = 2.0
        return original(self, prediction)

    monkeypatch.setattr(SQLAlchemyPredictionRepository, "add", add)
    client, _ = api
    response = client.post("/api/v1/predictions/batch", json=[request_payload] * 2)
    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction storage is unavailable."}
    assert counts(database) == (0, 0)


def test_commit_failure_is_not_reported_as_success(api, database, request_payload):
    def fail(session) -> None:
        raise SQLAlchemyError("Synthetic commit failure")

    client, _ = api
    event.listen(Session, "before_commit", fail)
    try:
        assert client.post("/api/v1/predictions", json=request_payload).status_code == 503
    finally:
        event.remove(Session, "before_commit", fail)
    assert counts(database) == (0, 0)


def test_active_switch_and_metadata_conflict(
    api, database, request_payload, artifact_path, tmp_path
):
    client, settings = api
    assert client.post("/api/v1/predictions", json=request_payload).status_code == 200
    artifact = load_model_artifact(artifact_path)
    path = tmp_path / "other.joblib"
    save_model_artifact(artifact.pipeline, replace(artifact.metadata, version="test-v2"), path)
    settings.model_path = str(path)
    assert client.get("/api/v1/models/active").json()["version"] == "test-v2"
    assert client.post("/api/v1/predictions", json=request_payload).status_code == 200
    with Session(database) as session:
        assert (
            session.scalars(select(ModelMetadata).where(ModelMetadata.is_active)).one().version
            == "test-v2"
        )
    payload = json.loads(path.with_suffix(".json").read_text())
    payload["threshold"] = 0.9
    path.with_suffix(".json").write_text(json.dumps(payload))
    assert client.post("/api/v1/predictions", json=request_payload).status_code == 500
    assert counts(database) == (2, 2)


def test_concurrent_first_requests_register_once(api, database, request_payload):
    client, _ = api
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(lambda _: client.post("/api/v1/predictions", json=request_payload), range(2))
        )
    assert [response.status_code for response in responses] == [200, 200]
    assert counts(database) == (1, 2)


def test_migration_downgrade_upgrade_and_metadata_parity(database):
    with database.begin() as connection:
        config = Config("alembic.ini")
        config.attributes["connection"] = connection
        command.check(config)
        command.downgrade(config, "base")
        assert inspect(connection).get_table_names() == ["alembic_version"]
        assert inspect(connection).get_enums() == []
        command.upgrade(config, "head")
        command.check(config)


@pytest.mark.parametrize(
    "change", ["binary", "algorithm", "calibration", "unfitted", "frozen", "bad_manifest"]
)
def test_changed_or_unsupported_artifact_is_rejected(
    api, database, request_payload, artifact_path, tmp_path, change
):
    from credit_risk.ml.train import build_candidate_pipeline

    client, settings = api
    assert client.post("/api/v1/predictions", json=request_payload).status_code == 200
    artifact = load_model_artifact(artifact_path)
    path = tmp_path / "changed.joblib"
    pipeline = build_candidate_pipeline("xgboost") if change == "unfitted" else artifact.pipeline
    metadata = artifact.metadata
    if change == "algorithm":
        metadata = replace(metadata, algorithm="RandomForest")
    if change == "calibration":
        metadata = replace(metadata, calibration="sigmoid")
    save_model_artifact(pipeline, metadata, path)
    if change == "binary":
        with path.open("ab") as stream:
            stream.write(b"changed")
    if change == "frozen":
        (path.parent / "frozen.json").write_text(
            json.dumps({"artifact_sha256": "wrong", "metadata_sha256": "wrong"})
        )
    if change == "bad_manifest":
        (path.parent / "frozen.json").write_text("{}")
    settings.model_path = str(path)
    assert client.post("/api/v1/predictions", json=request_payload).status_code == 500
    assert counts(database) == (1, 1)


def test_failure_during_promotion_preserves_previous_active(
    api, database, request_payload, artifact_path, tmp_path, monkeypatch
):
    client, settings = api
    assert client.post("/api/v1/predictions", json=request_payload).status_code == 200
    artifact = load_model_artifact(artifact_path)
    path = tmp_path / "v2.joblib"
    save_model_artifact(artifact.pipeline, replace(artifact.metadata, version="v2"), path)
    settings.model_path = str(path)

    def fail(self, prediction) -> Prediction:
        raise SQLAlchemyError("Synthetic insert failure")

    monkeypatch.setattr(SQLAlchemyPredictionRepository, "add", fail)
    assert client.post("/api/v1/predictions", json=request_payload).status_code == 503
    assert counts(database) == (1, 1)
    with Session(database) as session:
        assert (
            session.scalars(select(ModelMetadata).where(ModelMetadata.is_active)).one().version
            == "test-v1"
        )


def test_inference_failure_does_not_write(api, database, request_payload, monkeypatch):
    from credit_risk.exceptions import InvalidFeatureSchemaError

    def fail(pipeline, features) -> float:
        raise InvalidFeatureSchemaError("Synthetic invalid feature mapping")

    monkeypatch.setattr("credit_risk.services.prediction_service.ml_predict.predict", fail)
    client, _ = api
    assert client.post("/api/v1/predictions", json=request_payload).status_code == 422
    assert counts(database) == (0, 0)
