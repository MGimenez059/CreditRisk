#!/usr/bin/env python3
"""CLI entrypoint for standalone model evaluation.

Reports the metrics in README.md's Model section for the currently active
artifact and writes them to `docs/model_card.md`. Implemented in roadmap
Phase 4 (XGBoost).

Usage:
    python scripts/evaluate_model.py
"""

import sys
from pathlib import Path

from credit_risk.config.settings import get_settings


def main() -> int:
    """Evaluate the active model artifact against the held-out test set.

    Returns:
        Process exit code: 0 on success, 1 if evaluation cannot proceed.
    """
    settings = get_settings()
    # TODO(ROADMAP-P4): load the artifact at settings.model_path via
    # credit_risk.ml.registry.load_model_artifact, call
    # credit_risk.ml.evaluate.evaluate_model, and update docs/model_card.md.
    print(
        "Model evaluation is implemented in roadmap Phase 4 (XGBoost). "
        f"Configured model path: {Path(settings.model_path)}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
