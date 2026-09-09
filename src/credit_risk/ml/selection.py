"""Select and freeze a complete model using development data only."""

import json
import math
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

import pandas as pd

from credit_risk.ml.calibration import calibrate_and_select
from credit_risk.ml.evaluate import evaluate_model
from credit_risk.ml.preprocessing import RAW_FEATURE_COLUMNS
from credit_risk.ml.registry import ModelArtifactMetadata, save_model_artifact
from credit_risk.ml.selection_cv import (
    configured_pipeline,
    grouped_folds,
    search_candidates,
    select_candidate,
)
from credit_risk.ml.selection_state import FrozenSelection, file_sha256
from credit_risk.ml.train import split_dataset


def run_selection(input_path: Path, output_dir: Path, n_trials: int = 12, n_folds: int = 5) -> Path:
    """Run CV/tuning and freeze the artifact; never predict on test rows."""
    if output_dir.exists():
        raise FileExistsError(f"Run directory already exists: {output_dir}")
    frame = pd.read_parquet(input_path).reset_index(drop=True)
    if (
        frame.empty
        or frame["loan_status"].isna().any()
        or set(frame["loan_status"].unique()) != {0, 1}
    ):
        raise ValueError("Selection requires non-empty data with both binary target classes.")
    split = split_dataset(frame, "loan_status")
    folds = grouped_folds(split.x_train, split.y_train, n_folds)
    calibration_rows, decision_rows = grouped_folds(split.x_val, split.y_val, 2)[0]
    output_dir.mkdir(parents=True)
    protocol = Path("docs/selection_protocol.md").read_text(encoding="utf-8")
    (output_dir / "protocol.md").write_text(protocol, encoding="utf-8")
    results, trials = search_candidates(split.x_train, split.y_train, folds, n_trials)
    winner = select_candidate(results)
    base = configured_pipeline(winner.candidate, split.y_train)
    base.fit(split.x_train, split.y_train)
    selected, calibration, threshold, calibration_evidence = calibrate_and_select(
        base,
        split.x_val.iloc[calibration_rows],
        split.y_val.take(calibration_rows),
        split.x_val.iloc[decision_rows],
        split.y_val.take(decision_rows),
    )
    positions = {
        "train": split.x_train.index.tolist(),
        "calibration": split.x_val.iloc[calibration_rows].index.tolist(),
        "decision": split.x_val.iloc[decision_rows].index.tolist(),
        "test": split.x_test.index.tolist(),
    }
    parameters = {
        key: "NaN" if isinstance(value, float) and math.isnan(value) else value
        for key, value in base.named_steps["model"].get_params().items()
    }
    evidence = {
        "protocol": "phase4-v1",
        "n_trials": n_trials,
        "n_folds": n_folds,
        "cv_partition": "original_train",
        "cv_threshold": 0.5,
        "candidates": [asdict(result) for result in results],
        "trials": [asdict(result) for result in trials],
        "cv_positions": [
            {
                "train": split.x_train.iloc[train].index.tolist(),
                "validation": split.x_train.iloc[val].index.tolist(),
            }
            for train, val in folds
        ],
        "selected_candidate": asdict(winner.candidate),
        "resolved_parameters": parameters,
        "calibration": calibration_evidence,
        "test_evaluated": False,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8"
    )
    decision_metrics = asdict(
        evaluate_model(
            selected, split.x_val.iloc[decision_rows], split.y_val.take(decision_rows), threshold
        )
    )
    metadata = ModelArtifactMetadata(
        name=f"credit-risk-{winner.candidate.model_type}",
        version=output_dir.name,
        algorithm=type(base.named_steps["model"]).__name__,
        dataset_version="sha256:" + file_sha256(input_path),
        feature_version="raw-input-v1",
        metrics=decision_metrics,
        trained_at=datetime.now(UTC),
        python_version=platform.python_version(),
        dependency_versions={
            package.metadata["Name"]: package.version for package in distributions()
        },
        threshold=threshold,
        calibration=calibration,
        metrics_partition="decision",
    )
    save_model_artifact(selected, metadata, output_dir / "model.joblib")
    frozen = FrozenSelection(
        frozen_at=datetime.now(UTC).isoformat(),
        dataset_sha256=file_sha256(input_path),
        artifact_sha256=file_sha256(output_dir / "model.joblib"),
        metadata_sha256=file_sha256(output_dir / "model.json"),
        evidence_sha256=file_sha256(output_dir / "selection.json"),
        protocol_sha256=file_sha256(output_dir / "protocol.md"),
        feature_columns=RAW_FEATURE_COLUMNS,
        positions=positions,
        candidate=winner.candidate.name,
        calibration=calibration,
        threshold=threshold,
    )
    freeze_path = output_dir / "frozen.json"
    freeze_path.write_text(frozen.model_dump_json(indent=2), encoding="utf-8")
    return freeze_path
