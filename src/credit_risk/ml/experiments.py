"""Record fixed-candidate training and validation without scoring the test set."""

import hashlib
import json
import math
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

import pandas as pd

from credit_risk.ml.evaluate import evaluate_model
from credit_risk.ml.preprocessing import RAW_FEATURE_COLUMNS
from credit_risk.ml.registry import ModelArtifactMetadata, save_model_artifact
from credit_risk.ml.train import (
    DEFAULT_RANDOM_STATE,
    CandidateModelType,
    build_candidate_pipeline,
    measure_class_balance,
    split_dataset,
)

BASELINE_TYPES: tuple[CandidateModelType, ...] = ("logistic_regression", "random_forest")
CANDIDATE_TYPES: tuple[CandidateModelType, ...] = (*BASELINE_TYPES, "xgboost")


def run_experiment(
    input_path: Path,
    output_dir: Path,
    model_types: tuple[CandidateModelType, ...] = BASELINE_TYPES,
) -> Path:
    """Train on a validated Parquet snapshot and write artifacts plus a run manifest.

    Args:
        input_path: Ingestion output; raw-input columns plus binary loan_status.
        output_dir: New directory for this run; existing directories are rejected.
        model_types: Fixed candidates to compare on the same validation rows.

    Returns:
        Path to the JSON manifest, including validation metrics and split positions.
    """
    if not model_types or len(set(model_types)) != len(model_types):
        raise ValueError("Choose a non-empty set of distinct candidates.")
    if any(name not in CANDIDATE_TYPES for name in model_types):
        raise ValueError("Unsupported candidate.")
    if output_dir.exists():
        raise FileExistsError(f"Run directory already exists: {output_dir}. Choose a new path.")
    frame = pd.read_parquet(input_path).reset_index(drop=True)
    if frame.empty or frame["loan_status"].isna().any():
        raise ValueError("Training requires non-empty data with a non-null binary target.")
    if set(frame["loan_status"].unique()) != {0, 1}:
        raise ValueError("Training requires both binary target classes 0 and 1.")
    split = split_dataset(frame, "loan_status")
    dataset_version = "sha256:" + hashlib.sha256(input_path.read_bytes()).hexdigest()
    versions = {package.metadata["Name"]: package.version for package in distributions()}
    output_dir.mkdir(parents=True)
    results: dict[str, dict[str, float]] = {}
    parameters: dict[str, object] = {}
    for model_type in model_types:
        pipeline = build_candidate_pipeline(model_type)
        pipeline.fit(split.x_train, split.y_train)
        metrics = asdict(evaluate_model(pipeline, split.x_val, split.y_val))
        results[model_type] = metrics
        parameters[model_type] = {
            key: "NaN" if isinstance(value, float) and math.isnan(value) else value
            for key, value in pipeline.named_steps["model"].get_params().items()
        }
        metadata = ModelArtifactMetadata(
            name=f"credit-risk-{model_type}",
            version="0.1.0",
            algorithm=type(pipeline.named_steps["model"]).__name__,
            dataset_version=dataset_version,
            feature_version="raw-input-v1",
            metrics=metrics,
            trained_at=datetime.now(UTC),
            python_version=platform.python_version(),
            dependency_versions=versions,
        )
        save_model_artifact(pipeline, metadata, output_dir / f"{model_type}.joblib")
    positions = {
        "train": split.x_train.index.tolist(),
        "validation": split.x_val.index.tolist(),
        "test": split.x_test.index.tolist(),
    }
    manifest = {
        "dataset_version": dataset_version,
        "input_path": input_path.as_posix(),
        "python_version": platform.python_version(),
        "dependency_versions": versions,
        "random_state": DEFAULT_RANDOM_STATE,
        "features": RAW_FEATURE_COLUMNS,
        "split_policy": "raw-input-groups-majority-label-70-15-15-v1",
        "split_positions": positions,
        "split_rows": {name: len(indices) for name, indices in positions.items()},
        "development_positive_rates": {
            "train": float(split.y_train.mean()),
            "validation": float(split.y_val.mean()),
        },
        "development_class_balance": {
            "train": asdict(measure_class_balance(split.y_train)),
            "validation": asdict(measure_class_balance(split.y_val)),
        },
        "test_evaluated": False,
        "parameters": parameters,
        "validation_metrics": results,
        "pr_auc_definition": "sklearn.metrics.average_precision_score",
        "threshold": 0.5,
    }
    manifest_path = output_dir / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Candidate validation report",
        "",
        f"Dataset: `{dataset_version}`",
        "",
        "Grouped split; test predictions and metrics were not computed.",
        "",
        "`pr_auc` is average precision; F1, precision and recall use threshold 0.5.",
        "",
        "| Metric | " + " | ".join(model_types) + " |",
        "|---|" + "---:|" * len(model_types),
    ]
    for metric in results[model_types[0]]:
        values = " | ".join(f"{results[name][metric]:.6f}" for name in model_types)
        lines.append(f"| {metric} | {values} |")
    (output_dir / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path
