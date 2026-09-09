#!/usr/bin/env python3
"""CLI entrypoint for standalone model evaluation.

Final test evaluation remains pending until model, features, calibration and
threshold are frozen under the Phase 4 protocol. Initial candidate validation
is available through scripts/train_model.py.

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
        "Final test evaluation is not implemented; freeze Phase 4 selection first. "
        f"Configured model path: {Path(settings.model_path)}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
