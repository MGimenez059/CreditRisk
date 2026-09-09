#!/usr/bin/env python3
"""Select, calibrate and freeze a model without scoring the test set."""

import argparse
import sys
from pathlib import Path

from xgboost.core import XGBoostError

from credit_risk.exceptions import CreditRiskError
from credit_risk.ml.experiments import CANDIDATE_TYPES, run_experiment
from credit_risk.ml.selection import run_selection


def main() -> int:
    """Write candidate artifacts and validation evidence to a fresh run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/interim/credit_risk_validated.parquet")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--initial-only", action="store_true", help="Repeat the fixed-candidate comparison only."
    )
    args = parser.parse_args()
    try:
        if args.initial_only:
            output = args.output or Path("models/candidates-v1")
            manifest_path = run_experiment(args.input, output, CANDIDATE_TYPES)
        else:
            output = args.output or Path("models/selected-v1")
            manifest_path = run_selection(args.input, output)
    except (OSError, ValueError, KeyError, CreditRiskError, XGBoostError) as err:
        print(f"Candidate training failed: {err}", file=sys.stderr)
        return 1
    print(f"Training evidence: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
