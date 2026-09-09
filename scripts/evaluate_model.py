#!/usr/bin/env python3
"""Evaluate the unchanged Phase 4 selection on test once; preserve the report."""

import argparse
import sys
from pathlib import Path

from credit_risk.exceptions import CreditRiskError
from credit_risk.ml.final_evaluation import evaluate_frozen


def main() -> int:
    """Verify the frozen run and reject repeated test evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/interim/credit_risk_validated.parquet")
    )
    parser.add_argument("--run", type=Path, default=Path("models/selected-v1"))
    args = parser.parse_args()
    try:
        report = evaluate_frozen(args.input, args.run)
    except (OSError, ValueError, KeyError, CreditRiskError) as err:
        print(f"Final evaluation failed: {err}", file=sys.stderr)
        return 1
    print(f"Final test report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
