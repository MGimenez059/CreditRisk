"""Regression tests for the Phase 3 artifact and evaluation boundaries."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import train_baselines
from credit_risk.ml.evaluate import EvaluationReport, evaluate_model
from credit_risk.ml.preprocessing import RAW_FEATURE_COLUMNS
from credit_risk.ml.registry import load_model_artifact
from credit_risk.ml.train import build_baseline_pipeline, split_dataset
from credit_risk.schemas.prediction import PredictionRequest
from credit_risk.services.prediction_service import _to_feature_frame
from tests.unit.test_train import _synthetic_dataset


def test_split_keeps_identical_inputs_and_conflicting_labels_together() -> None:
    frame = _synthetic_dataset()
    copies = pd.concat([frame.iloc[:20], frame.iloc[:20]], ignore_index=True)
    copies.loc[20:, "loan_status"] = 1 - copies.loc[20:, "loan_status"]
    copies["loan_grade"] = "G"
    frame["loan_grade"] = "A"
    frame = pd.concat([frame, copies], ignore_index=True)

    split = split_dataset(frame, "loan_status")
    partitions = [split.x_train, split.x_val, split.x_test]
    groups = [set(pd.util.hash_pandas_object(part, index=False)) for part in partitions]

    assert sum(map(len, partitions)) == len(frame)
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])


def test_baseline_run_round_trips_api_input_without_scoring_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _synthetic_dataset()
    input_path = tmp_path / "validated.parquet"
    frame.to_parquet(input_path, index=False)
    expected_split = split_dataset(frame, "loan_status")
    evaluated_indices = []

    def evaluate_validation_only(pipeline, features, target) -> EvaluationReport:
        evaluated_indices.append(features.index.tolist())
        assert features.index.tolist() == expected_split.x_val.index.tolist()
        return evaluate_model(pipeline, features, target)

    monkeypatch.setattr(train_baselines, "evaluate_model", evaluate_validation_only)
    output_dir = tmp_path / "run"
    manifest_path = train_baselines.run_baselines(input_path, output_dir)
    manifest = json.loads(manifest_path.read_text())
    assert len(evaluated_indices) == 2
    assert manifest["test_evaluated"] is False
    assert manifest["dataset_version"].startswith("sha256:")
    assert set(manifest["split_positions"]["test"]) == set(expected_split.x_test.index)
    assert (output_dir / "validation.md").exists()

    request = PredictionRequest(
        age=34,
        income=60000,
        employment_years=None,
        home_ownership="RENT",
        loan_amount=12000,
        interest_rate=None,
        loan_intent="PERSONAL",
        credit_history_years=7,
        previous_defaults=1,
    )
    raw_input = _to_feature_frame(request)
    assert set(raw_input.columns) == set(RAW_FEATURE_COLUMNS)
    assert raw_input.loc[0, "cb_person_default_on_file"] == "Y"
    for model_type in train_baselines.BASELINE_TYPES:
        artifact = load_model_artifact(output_dir / f"{model_type}.joblib")
        probability = artifact.pipeline.predict_proba(raw_input)
        reference = build_baseline_pipeline(model_type)
        reference.fit(expected_split.x_train, expected_split.y_train)
        np.testing.assert_allclose(probability, reference.predict_proba(raw_input))
        assert np.isfinite(probability).all()
        assert ((probability >= 0) & (probability <= 1)).all()
        assert artifact.metadata.python_version != "unknown"
        assert "scikit-learn" in artifact.metadata.dependency_versions
    with pytest.raises(FileExistsError):
        train_baselines.run_baselines(input_path, output_dir)


def test_pipeline_ignores_supplied_derived_values_and_fits_training_medians_only() -> None:
    split = split_dataset(_synthetic_dataset(), "loan_status")
    pipeline = build_baseline_pipeline("logistic_regression")
    pipeline.fit(split.x_train, split.y_train)
    imputer = (
        pipeline.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["impute"]
    )
    assert imputer.statistics_[1] == split.x_train["person_income"].median()
    raw = split.x_val.copy()
    expected = pipeline.predict_proba(raw)
    raw["loan_to_income"] = -999999
    np.testing.assert_allclose(expected, pipeline.predict_proba(raw))


@pytest.mark.parametrize("income", [0, -1, float("inf"), float("nan")])
def test_api_rejects_undefined_income(income: float) -> None:
    with pytest.raises(ValueError):
        PredictionRequest(
            age=34,
            income=income,
            employment_years=6,
            home_ownership="RENT",
            loan_amount=12000,
            interest_rate=12.5,
            loan_intent="PERSONAL",
            credit_history_years=7,
            previous_defaults=0,
        )


@pytest.mark.parametrize("indicator", [None, -1, 2])
def test_api_requires_a_binary_prior_default(indicator: int | None) -> None:
    with pytest.raises(ValueError):
        PredictionRequest(
            age=34,
            income=60000,
            employment_years=6,
            home_ownership="RENT",
            loan_amount=12000,
            interest_rate=12.5,
            loan_intent="PERSONAL",
            credit_history_years=7,
            previous_defaults=indicator,
        )
