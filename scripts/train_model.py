#!/usr/bin/env python3
"""CLI entrypoint for model training.

Runnable end-to-end on a clean checkout with no manual steps beyond
`pip install` and running `scripts/ingest_data.py` first, per the
reproducibility requirement in CODESTYLE.md §14. Implemented in roadmap
Phase 4 (XGBoost).

Usage:
    python scripts/train_model.py
"""

import sys

from credit_risk.ml.train import DEFAULT_RANDOM_STATE


def main() -> int:
    """Train the production pipeline and persist it via `ml.registry`.

    Returns:
        Process exit code: 0 on success, 1 if training cannot proceed.
    """
    # TODO(ROADMAP-P4): load processed data, call credit_risk.ml.train.train_model
    # with random_state=DEFAULT_RANDOM_STATE, then ml.registry.save_model_artifact.
    print(
        "Model training is implemented in roadmap Phase 4 (XGBoost). "
        f"Configured random_state: {DEFAULT_RANDOM_STATE}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
