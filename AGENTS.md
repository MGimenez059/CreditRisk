# CreditRisk — Project Context

## Start here

Before changing this repository, read the current versions of:

1. [README.md](README.md): purpose, setup and implemented behavior.
2. [CODESTYLE.md](CODESTYLE.md): engineering and commit conventions.
3. [SPECS.md](SPECS.md): technical requirements and MVP scope.
4. [ROADMAP.md](ROADMAP.md): current phase, completed work and next steps.
5. [Architecture](docs/architecture.md): boundaries and pending serving work.
6. [Data dictionary](docs/data_dictionary.md) and
   [model card](docs/model_card.md): inputs, evaluation protocol and limitations.

For data/model changes also read the current [quality report](docs/data_quality_report.md),
[dataset provenance](docs/dataset_provenance.json) and
[baseline results](docs/baseline_results.json). Consult [notebook guidance](notebooks/README.md)
when working on EDA. Inspect Git status and relevant implementation before editing.

These files are the project memory. Read their current contents rather than relying
on a previous conversation or duplicating their full text here. ROADMAP owns phase
status; CODESTYLE owns engineering conventions; SPECS owns required behavior.
Resolve contradictions explicitly and keep affected documents consistent with changes.

## Project intent

Build an educational portfolio spanning Data Science, ML Engineering and Backend:
reproducible training, justified model comparison, explanations and a working API.
Keep scope proportionate. Prefer completing the current phase over adding speculative
infrastructure. Customer CRUD, a dashboard and advanced infrastructure are post-MVP.

Work on the selected checkout, preserving existing changes. Follow CODESTYLE for
repository language and commit conventions.

## Constraints worth preserving

- Keep identical raw model inputs in one split/fold, including conflicting labels.
- Serialize feature engineering, preprocessing and estimator together; verify the
  API-to-source mapping against that raw-input pipeline.
- Select the model based on evidence. XGBoost is a candidate, not a guaranteed winner.
- Do not score the final test partition until model/feature/calibration/threshold
  choices are frozen under the documented protocol.
- Do not fabricate missing source fields, provenance, metrics or successful checks.
- Distinguish implemented scaffolds from operational serving. No real lending claims.
- Preserve existing experiment outputs; use a fresh output directory for another run.

## Working commands

Use Python 3.12 and the committed uv.lock. Setup and data acquisition are in README.

```bash
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src scripts
uv run --locked pytest
```

Run checks appropriate to the change. Documentation-only edits need consistency and
link review, not model retraining. Read ROADMAP at handoff and update actual progress
there; do not maintain a second phase checklist or metric table in this file.
