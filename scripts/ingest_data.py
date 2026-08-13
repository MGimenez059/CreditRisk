#!/usr/bin/env python3
"""CLI entrypoint for data ingestion.

Downloads (or validates a manually placed copy of) the training dataset,
runs schema validation, and writes a data quality report. Implemented in
roadmap Phase 2 (Datos). See `docs/data_dictionary.md` for the dataset's
provenance and license.

Usage:
    python scripts/ingest_data.py
"""

import sys
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")


def main() -> int:
    """Run data ingestion end-to-end.

    Returns:
        Process exit code: 0 on success, 1 if the pipeline cannot proceed.
    """
    # TODO(ROADMAP-P2): download/validate Kaggle "laotse/credit-risk-dataset"
    # into data/raw, run schema + range validation, and write a data quality
    # report, per docs/data_dictionary.md.
    print(
        "Data ingestion is implemented in roadmap Phase 2 (Datos). "
        f"Expected input location once implemented: {RAW_DATA_DIR}/credit_risk_dataset.csv",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())