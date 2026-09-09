#!/usr/bin/env python3
"""Compare fixed XGBoost and baseline candidates without scoring the test set."""

import argparse
import sys
from pathlib import Path

from xgboost.core import XGBoostError

from credit_risk.exceptions import CreditRiskError
from credit_risk.ml.experiments import CANDIDATE_TYPES, run_experiment


def main() -> int:
    """Write candidate artifacts and validation evidence to a fresh run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/interim/credit_risk_validated.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("models/candidates-v1"))
    args = parser.parse_args()
    try:
        manifest_path = run_experiment(args.input, args.output, CANDIDATE_TYPES)
    except (OSError, ValueError, KeyError, CreditRiskError, XGBoostError) as err:
        print(f"Candidate training failed: {err}", file=sys.stderr)
        return 1
    print(f"Validation comparison and split manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
