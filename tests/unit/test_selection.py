"""Selection boundaries and final-test safeguards on explicitly synthetic data."""

import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from credit_risk.ml import final_evaluation, selection_cv
from credit_risk.ml.calibration import calibrate_and_select, choose_threshold
from credit_risk.ml.evaluate import EvaluationReport
from credit_risk.ml.final_evaluation import evaluate_frozen, verify_frozen_selection
from credit_risk.ml.registry import load_model_artifact
from credit_risk.ml.selection import run_selection
from credit_risk.ml.selection_cv import Candidate, configured_pipeline, grouped_folds
from credit_risk.ml.train import split_dataset
from credit_risk.schemas.prediction import PredictionRequest
from credit_risk.services.prediction_service import _to_feature_frame
from tests.unit.test_train import _synthetic_dataset


def test_grouped_cv_preserves_conflicting_duplicates_and_covers_every_row():
    frame = _synthetic_dataset(400)
    copies = frame.iloc[:20].copy()
    copies["loan_status"] = 1 - copies["loan_status"]
    frame = pd.concat([frame, copies], ignore_index=True)
    features = frame.drop(columns="loan_status")
    folds = grouped_folds(features, frame.loan_status)
    hashes = pd.util.hash_pandas_object(features, index=False)
    seen = []
    for train, validation in folds:
        assert set(hashes.iloc[train]).isdisjoint(hashes.iloc[validation])
        seen.extend(validation)
    assert sorted(seen) == list(range(len(frame)))
    assert folds == grouped_folds(features, frame.loan_status)


def test_weight_ratio_uses_only_supplied_training_labels():
    candidate = Candidate("weighted", "xgboost", True)
    pipeline = configured_pipeline(candidate, pd.Series([0, 0, 0, 1]))
    assert pipeline.named_steps["model"].scale_pos_weight == 3
    assert (
        configured_pipeline(replace(candidate, weighted=False), pd.Series([0, 1]))
        .named_steps["model"]
        .scale_pos_weight
        == 1
    )


def test_cv_fits_preprocessing_on_each_training_fold(monkeypatch):
    frame = _synthetic_dataset(200)
    features, target = frame.drop(columns="loan_status"), frame.loan_status
    folds = grouped_folds(features, target, 2)
    original = selection_cv.evaluate_model
    calls = []

    def check_fit(pipeline, validation, labels) -> EvaluationReport:
        train, expected_validation = folds[len(calls)]
        assert validation.index.tolist() == features.iloc[expected_validation].index.tolist()
        imputer = (
            pipeline.named_steps["preprocessing"]
            .named_transformers_["numeric"]
            .named_steps["impute"]
        )
        assert imputer.statistics_[1] == features.iloc[train].person_income.median()
        calls.append(1)
        return original(pipeline, validation, labels)

    monkeypatch.setattr(selection_cv, "evaluate_model", check_fit)
    selection_cv.cross_validate_candidate(
        Candidate("lr", "logistic_regression", False), features, target, folds
    )
    assert len(calls) == 2


def test_calibration_does_not_refit_base_estimator(monkeypatch):
    split = split_dataset(_synthetic_dataset(400), "loan_status")
    calibration, decision = grouped_folds(split.x_val, split.y_val, 2)[0]
    base = configured_pipeline(Candidate("lr", "logistic_regression", False), split.y_train)
    base.fit(split.x_train, split.y_train)
    before = base.predict_proba(split.x_val)

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        pytest.fail("Calibration refitted the base estimator")

    monkeypatch.setattr(base, "fit", forbidden_fit)
    calibrate_and_select(
        base,
        split.x_val.iloc[calibration],
        split.y_val.iloc[calibration],
        split.x_val.iloc[decision],
        split.y_val.iloc[decision],
    )
    np.testing.assert_allclose(before, base.predict_proba(split.x_val))


def test_threshold_uses_fixed_f1_objective_and_tie_break():
    threshold, grid = choose_threshold(pd.Series([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    assert threshold == 0.6
    assert len(grid) == 91


@pytest.fixture(scope="module")
def synthetic_selection(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-selection")
    input_path = root / "synthetic.parquet"
    _synthetic_dataset(400).to_parquet(input_path, index=False)
    run = root / "run"
    run_selection(input_path, run, n_trials=1, n_folds=2)
    return input_path, run


def test_selection_freezes_disjoint_groups_and_round_trips_api_input(synthetic_selection):
    input_path, run = synthetic_selection
    frozen, frame = verify_frozen_selection(input_path, run)
    groups = []
    for positions in frozen.positions.values():
        part = frame.iloc[positions][frozen.feature_columns]
        groups.append(set(pd.util.hash_pandas_object(part, index=False)))
    for i, left in enumerate(groups):
        for right in groups[i + 1 :]:
            assert left.isdisjoint(right)
    assert sum(map(len, frozen.positions.values())) == len(frame)
    artifact = load_model_artifact(run / "model.joblib")
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
    probabilities = artifact.pipeline.predict_proba(_to_feature_frame(request))
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert artifact.metadata.threshold == frozen.threshold
    assert not (run / "test.json").exists()
    with pytest.raises(FileExistsError):
        run_selection(input_path, run)


@pytest.mark.parametrize(
    "filename", ["model.joblib", "selection.json", "model.json", "protocol.md"]
)
def test_evaluation_rejects_changed_frozen_files(synthetic_selection, tmp_path, filename):
    input_path, source = synthetic_selection
    run = tmp_path / "run"
    shutil.copytree(source, run)
    with (run / filename).open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="Frozen file changed"):
        evaluate_frozen(input_path, run)
    assert not (run / "test-evaluation.started").exists()


def test_evaluation_rejects_changed_data_and_partitions(synthetic_selection, tmp_path):
    input_path, source = synthetic_selection
    run = tmp_path / "run"
    shutil.copytree(source, run)
    other_input = tmp_path / "other.parquet"
    _synthetic_dataset(401).to_parquet(other_input)
    with pytest.raises(ValueError, match="snapshot"):
        evaluate_frozen(other_input, run)
    frozen_path = run / "frozen.json"
    frozen = json.loads(frozen_path.read_text())
    frozen["positions"]["test"][0] = frozen["positions"]["train"][0]
    frozen_path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="partitions"):
        evaluate_frozen(input_path, run)


def test_final_evaluation_predicts_once_and_refuses_repetition(
    synthetic_selection, tmp_path, monkeypatch
):
    input_path, source = synthetic_selection
    run = tmp_path / "run"
    shutil.copytree(source, run)
    artifact = load_model_artifact(run / "model.joblib")
    calls = []

    class CountingPipeline:
        def predict_proba(self, features) -> np.ndarray:
            calls.append(features.index.tolist())
            return artifact.pipeline.predict_proba(features)

    monkeypatch.setattr(
        final_evaluation,
        "load_model_artifact",
        lambda _path: replace(artifact, pipeline=CountingPipeline()),
    )
    result = json.loads(evaluate_frozen(input_path, run).read_text())
    assert len(calls) == 1
    frozen, _ = verify_frozen_selection(input_path, run)
    assert calls[0] == frozen.positions["test"]
    assert result["threshold"] == frozen.threshold
    with pytest.raises(FileExistsError):
        evaluate_frozen(input_path, run)
    assert len(calls) == 1


def test_failed_prediction_keeps_one_shot_marker(synthetic_selection, tmp_path, monkeypatch):
    input_path, source = synthetic_selection
    run = tmp_path / "run"
    shutil.copytree(source, run)
    artifact = load_model_artifact(run / "model.joblib")

    class BrokenPipeline:
        def predict_proba(self, features) -> np.ndarray:
            raise RuntimeError("Synthetic scoring failure")

    monkeypatch.setattr(
        final_evaluation,
        "load_model_artifact",
        lambda _path: replace(artifact, pipeline=BrokenPipeline()),
    )
    with pytest.raises(RuntimeError, match="Synthetic"):
        evaluate_frozen(input_path, run)
    assert (run / "test-evaluation.started").exists()
    with pytest.raises(FileExistsError):
        evaluate_frozen(input_path, run)


def test_candidate_selection_respects_auc_tolerance_and_log_loss():
    from credit_risk.ml.selection_cv import CVResult, select_candidate

    def result(name: str, auc: float, loss: float) -> CVResult:
        return CVResult(
            Candidate(name, "xgboost", False), [], {"roc_auc": auc, "log_loss": loss}, {}
        )

    best_auc = result("best-auc", 0.94, 0.25)
    near = result("near", 0.939, 0.22)
    distant = result("distant", 0.93, 0.20)
    assert select_candidate([best_auc, near, distant]) == near
