"""SHAP additivity, raw-input mapping and explanation response regression tests."""

import numpy as np
import pytest
from shap.utils._exceptions import ExplainerError
from xgboost.core import XGBoostError

from credit_risk.exceptions import ExplanationError
from credit_risk.ml.explain import explain_batch, explain_prediction
from credit_risk.ml.train import build_candidate_pipeline, split_dataset
from credit_risk.schemas.prediction import ExplanationResponse, PredictionRequest
from credit_risk.services.explanation_service import ExplanationService
from credit_risk.services.prediction_service import _to_feature_frame
from tests.unit.test_train import _synthetic_dataset


@pytest.fixture(scope="module")
def fitted_pipeline():
    split = split_dataset(_synthetic_dataset(200), "loan_status")
    pipeline = build_candidate_pipeline("xgboost")
    pipeline.fit(split.x_train, split.y_train)
    return pipeline, split


def test_batch_additivity_and_category_aggregation(fitted_pipeline):
    pipeline, split = fitted_pipeline
    results = explain_batch(pipeline, split.x_val.iloc[:5])
    assert len(results) == 5
    for result in results:
        assert result.base_value + sum(value for _, value in result.contributions) == pytest.approx(
            result.output_value, abs=1e-5
        )
        assert 1 / (1 + np.exp(-result.output_value)) == pytest.approx(result.probability, abs=1e-6)
        names = [name for name, _ in result.contributions]
        assert len(names) == 12
        assert "person_home_ownership" in names
        assert "loan_to_income" in names
        assert not any(name.startswith("categorical__") for name in names)


def test_api_missing_values_and_response_contract(fitted_pipeline):
    from dataclasses import asdict

    pipeline, _ = fitted_pipeline
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
    frame = _to_feature_frame(request)
    expected = pipeline.predict_proba(frame)[0, 1]
    explanation = ExplanationService().explain(pipeline, frame)
    response = ExplanationResponse.model_validate(asdict(explanation))
    assert response.output_space == "log_odds"
    assert response.explains == "uncalibrated_model"
    assert 1 / (1 + np.exp(-response.output_value)) == pytest.approx(expected, abs=1e-6)
    assert len(response.contributions) == 12
    assert all(
        item.direction
        == ("positive" if item.impact > 0 else "negative" if item.impact < 0 else "neutral")
        for item in response.contributions
    )


def test_local_rejects_multiple_rows_and_unsupported_pipeline(fitted_pipeline):
    pipeline, split = fitted_pipeline
    with pytest.raises(ExplanationError, match="exactly one"):
        explain_prediction(pipeline, split.x_val.iloc[:2])
    with pytest.raises(ExplanationError, match="uncalibrated"):
        explain_prediction(build_candidate_pipeline("random_forest"), split.x_val.iloc[:1])


def test_explanation_does_not_fit_or_mutate_predictions(fitted_pipeline, monkeypatch):
    pipeline, split = fitted_pipeline
    before = pipeline.predict_proba(split.x_val)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Explanation must not fit the pipeline")

    monkeypatch.setattr(pipeline, "fit", forbidden)
    explain_batch(pipeline, split.x_val)
    np.testing.assert_array_equal(before, pipeline.predict_proba(split.x_val))


@pytest.mark.parametrize("error_type", [ExplainerError, XGBoostError])
def test_library_failures_preserve_domain_error_and_cause(fitted_pipeline, monkeypatch, error_type):
    pipeline, split = fitted_pipeline
    original = error_type("Explanation failed")

    def fail(*args: object, **kwargs: object) -> None:
        raise original

    monkeypatch.setattr("credit_risk.ml.explain.shap.TreeExplainer", fail)
    with pytest.raises(ExplanationError) as caught:
        explain_prediction(pipeline, split.x_val.iloc[:1])
    assert caught.value.__cause__ is original


def test_route_keeps_explanation_base_and_units(fitted_pipeline):
    from credit_risk.api.routes.predictions import _to_response
    from credit_risk.services.prediction_service import PredictionResult

    pipeline, split = fitted_pipeline
    explanation = ExplanationService().explain(pipeline, split.x_val.iloc[:1])
    probability = float(pipeline.predict_proba(split.x_val.iloc[:1])[0, 1])
    result = PredictionResult(
        probability, round(probability * 100), "LOW", "synthetic", "test", explanation
    )
    response = _to_response(result)
    assert response.explanation.base_value == explanation.base_value
    assert response.explanation.output_space == "log_odds"
    assert len(response.explanation.contributions) == 12
