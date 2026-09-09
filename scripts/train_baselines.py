#!/usr/bin/env python3
"""Train and record both baselines without evaluating the test split."""

import argparse
import sys
from pathlib import Path

from credit_risk.exceptions import CreditRiskError
from credit_risk.ml.experiments import run_experiment


def main() -> int:
    """Parse paths and run Phase 3 baselines, reporting actionable failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/interim/credit_risk_validated.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("models/baselines-v1"))
    args = parser.parse_args()
    try:
        manifest_path = run_experiment(args.input, args.output)
    except (OSError, ValueError, KeyError, CreditRiskError) as err:
        print(f"Baseline training failed: {err}", file=sys.stderr)
        return 1
    print(f"Validation results and split manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
