#!/usr/bin/env python3
"""Explain a fixed development sample and a synthetic request, never test rows."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np

from credit_risk.ml.explain import explain_batch, explain_prediction
from credit_risk.ml.final_evaluation import verify_frozen_selection
from credit_risk.ml.registry import load_model_artifact
from credit_risk.ml.selection_state import file_sha256
from credit_risk.schemas.prediction import PredictionRequest
from credit_risk.services.prediction_service import _to_feature_frame

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_explanations(input_path: Path, run_dir: Path, output_dir: Path) -> Path:
    """Write SHAP evidence and plots in a new directory, keeping model hashes intact."""
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    frozen, frame = verify_frozen_selection(input_path, run_dir)
    artifact = load_model_artifact(run_dir / "model.joblib")
    if artifact.metadata.calibration != "none":
        raise ValueError("This report only supports the uncalibrated selected model.")
    development = frame.iloc[frozen.positions["decision"]]
    sample = development.sample(n=min(500, len(development)), random_state=42)
    explanations = explain_batch(artifact.pipeline, sample[frozen.feature_columns])
    names = sorted(name for name, _ in explanations[0].contributions)
    impacts = np.array([[dict(row.contributions)[name] for name in names] for row in explanations])
    importance = np.abs(impacts).mean(axis=0)
    order = np.argsort(importance)
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
    local = explain_prediction(artifact.pipeline, _to_feature_frame(request))
    report = {
        "artifact_sha256": frozen.artifact_sha256,
        "dataset_sha256": frozen.dataset_sha256,
        "partition": "decision",
        "sample_seed": 42,
        "sample_positions": sample.index.tolist(),
        "output_space": "log_odds",
        "method": "tree_path_dependent",
        "background": "training path counts stored in the frozen trees",
        "aggregation": (
            "sum encoded impacts per source field; derived numeric features remain separate"
        ),
        "importance": {names[i]: float(importance[i]) for i in order[::-1]},
        "max_additivity_error": max(
            abs(row.base_value + sum(value for _, value in row.contributions) - row.output_value)
            for row in explanations
        ),
        "synthetic_request": request.model_dump(),
        "synthetic_explanation": asdict(local),
        "test_scored": False,
    }
    output_dir.mkdir(parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    axes[0].barh(np.array(names)[order], importance[order], color="#2677ac")
    axes[0].set(
        xlabel="Mean absolute grouped SHAP (log-odds)",
        title=f"Global importance: {len(sample)} development rows",
    )
    rng = np.random.default_rng(42)
    for rank, index in enumerate(order):
        axes[1].scatter(
            impacts[:, index],
            rank + rng.uniform(-0.25, 0.25, len(sample)),
            s=6,
            alpha=0.35,
            color="#2677ac",
        )
    axes[1].set_yticks(range(len(names)), np.array(names)[order])
    axes[1].axvline(0, color="gray", linewidth=1)
    axes[1].set(
        xlabel="Grouped SHAP impact (log-odds)", title="SHAP summary: signed impact distribution"
    )
    fig.savefig(output_dir / "global_summary.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ordered = local.contributions[::-1]
    ax.barh(
        [name for name, _ in ordered],
        [value for _, value in ordered],
        color=["#d55e00" if value > 0 else "#2677ac" for _, value in ordered],
    )
    ax.axvline(0, color="gray", linewidth=1)
    ax.set(
        xlabel="SHAP contribution (log-odds)",
        title=(
            f"Synthetic request: p(default) = {local.probability:.3f}\n"
            f"Base {local.base_value:.3f} + contributions = margin {local.output_value:.3f}"
        ),
    )
    fig.savefig(output_dir / "local_explanation.png", dpi=160)
    plt.close(fig)
    if file_sha256(run_dir / "model.joblib") != frozen.artifact_sha256:
        raise ValueError("Model changed during explanation.")
    report_path = output_dir / "explanations.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report_path


def main() -> None:
    """Run from repository root with the optional viz dependencies installed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/interim/credit_risk_validated.parquet")
    )
    parser.add_argument("--run", type=Path, default=Path("models/selected-v1"))
    parser.add_argument("--output", type=Path, default=Path("models/explanations-v1"))
    args = parser.parse_args()
    print(run_explanations(args.input, args.run, args.output))


if __name__ == "__main__":
    main()
