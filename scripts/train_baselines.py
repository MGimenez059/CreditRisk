#!/usr/bin/env python3
"""Train and record both baselines without evaluating the test split.

Usage: python scripts/train_baselines.py --help
"""

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

import pandas as pd

from credit_risk.exceptions import CreditRiskError
from credit_risk.ml.evaluate import evaluate_model
from credit_risk.ml.preprocessing import RAW_FEATURE_COLUMNS
from credit_risk.ml.registry import ModelArtifactMetadata, save_model_artifact
from credit_risk.ml.train import (
    DEFAULT_RANDOM_STATE,
    BaselineModelType,
    build_baseline_pipeline,
    measure_class_balance,
    split_dataset,
)

BASELINE_TYPES: tuple[BaselineModelType, ...] = ("logistic_regression", "random_forest")


def run_baselines(input_path: Path, output_dir: Path) -> Path:
    """Train on a validated Parquet snapshot and write artifacts plus a run manifest.

    Args:
        input_path: Ingestion output; raw-input columns plus binary loan_status.
        output_dir: New directory for this run; existing directories are rejected.

    Returns:
        Path to the JSON manifest, including validation metrics and split positions.
    """
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
    for model_type in BASELINE_TYPES:
        pipeline = build_baseline_pipeline(model_type)
        pipeline.fit(split.x_train, split.y_train)
        metrics = asdict(evaluate_model(pipeline, split.x_val, split.y_val))
        results[model_type] = metrics
        parameters[model_type] = pipeline.named_steps["model"].get_params()
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
        "# Baseline validation report",
        "",
        f"Dataset: `{dataset_version}`",
        "",
        "Grouped split; test predictions and metrics were not computed.",
        "",
        "`pr_auc` is average precision; F1, precision and recall use threshold 0.5.",
        "",
        "| Metric | Logistic Regression | Random Forest |",
        "|---|---:|---:|",
    ]
    for metric, value in results["logistic_regression"].items():
        lines.append(f"| {metric} | {value:.6f} | {results['random_forest'][metric]:.6f} |")
    (output_dir / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    """Parse paths and run Phase 3 baselines, reporting actionable failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/interim/credit_risk_validated.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("models/baselines-v1"))
    args = parser.parse_args()
    try:
        manifest_path = run_baselines(args.input, args.output)
    except (OSError, ValueError, KeyError, CreditRiskError) as err:
        print(f"Baseline training failed: {err}", file=sys.stderr)
        return 1
    print(f"Validation results and split manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
