"""One-shot evaluation of an unchanged frozen selection on its original test split."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from credit_risk.ml.calibration import reliability_bins
from credit_risk.ml.evaluate import evaluate_probabilities
from credit_risk.ml.preprocessing import RAW_FEATURE_COLUMNS
from credit_risk.ml.registry import load_model_artifact
from credit_risk.ml.selection_cv import grouped_folds
from credit_risk.ml.selection_state import FrozenSelection, file_sha256
from credit_risk.ml.train import split_dataset


def verify_frozen_selection(
    input_path: Path, run_dir: Path
) -> tuple[FrozenSelection, pd.DataFrame]:
    """Reject changed artifacts, data or partitions before any test prediction."""
    frozen = FrozenSelection.model_validate_json(
        (run_dir / "frozen.json").read_text(encoding="utf-8")
    )
    if frozen.protocol != "phase4-v1" or frozen.feature_columns != RAW_FEATURE_COLUMNS:
        raise ValueError("Unsupported protocol or feature schema.")
    expected_hashes = {
        "model.joblib": frozen.artifact_sha256,
        "model.json": frozen.metadata_sha256,
        "selection.json": frozen.evidence_sha256,
        "protocol.md": frozen.protocol_sha256,
    }
    if file_sha256(input_path) != frozen.dataset_sha256:
        raise ValueError("Input snapshot differs from the frozen selection.")
    for filename, expected in expected_hashes.items():
        if file_sha256(run_dir / filename) != expected:
            raise ValueError(f"Frozen file changed: {filename}")
    frame = pd.read_parquet(input_path).reset_index(drop=True)
    split = split_dataset(frame, "loan_status")
    calibration, decision = grouped_folds(split.x_val, split.y_val, 2)[0]
    expected_positions = {
        "train": split.x_train.index.tolist(),
        "calibration": split.x_val.iloc[calibration].index.tolist(),
        "decision": split.x_val.iloc[decision].index.tolist(),
        "test": split.x_test.index.tolist(),
    }
    if frozen.positions != expected_positions:
        raise ValueError("Frozen partitions differ from the original grouped split.")
    return frozen, frame


def evaluate_frozen(input_path: Path, run_dir: Path) -> Path:
    """Score once; an exclusive marker also blocks retries after interrupted scoring."""
    marker_path = run_dir / "test-evaluation.started"
    report_path = run_dir / "test.json"
    if marker_path.exists() or report_path.exists():
        raise FileExistsError("Test evaluation already started for this run; use the saved report.")
    frozen, frame = verify_frozen_selection(input_path, run_dir)
    artifact = load_model_artifact(run_dir / "model.joblib")
    if (
        artifact.metadata.threshold != frozen.threshold
        or artifact.metadata.calibration != frozen.calibration
        or artifact.metadata.dataset_version != "sha256:" + frozen.dataset_sha256
    ):
        raise ValueError("Frozen configuration does not match artifact metadata.")
    with marker_path.open("x", encoding="utf-8") as marker:
        marker.write(
            json.dumps(
                {
                    "started_at": datetime.now(UTC).isoformat(),
                    "frozen_sha256": file_sha256(run_dir / "frozen.json"),
                }
            )
        )
    test = frame.iloc[frozen.positions["test"]]
    probabilities = np.asarray(
        artifact.pipeline.predict_proba(test[RAW_FEATURE_COLUMNS])[:, 1], dtype=np.float64
    )
    target = test["loan_status"]
    metrics = asdict(evaluate_probabilities(target, probabilities, frozen.threshold))
    report = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "partition": "test",
        "rows": len(test),
        "candidate": frozen.candidate,
        "threshold": frozen.threshold,
        "calibration": frozen.calibration,
        "dataset_sha256": frozen.dataset_sha256,
        "frozen_sha256": file_sha256(run_dir / "frozen.json"),
        "metrics": metrics,
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(
            target, probabilities >= frozen.threshold, labels=[0, 1]
        ).tolist(),
        "reliability": reliability_bins(target, probabilities),
    }
    with report_path.open("x", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, allow_nan=False)
    return report_path
