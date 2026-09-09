# CreditRisk

A portfolio project for estimating loan default probability and demonstrating
Data Science, ML Engineering, and Backend development with public tabular data.

**Current status: Phase 4 complete.** Grouped cross-validation, bounded Optuna
tuning, calibration comparison and threshold selection are implemented. A selected
XGBoost pipeline was frozen and evaluated once on test. SHAP explanations and an
operational prediction API are still pending. See [ROADMAP.md](ROADMAP.md).

## Scope

The MVP is a reproducible training workflow, an evaluated model, SHAP explanations,
a FastAPI prediction endpoint, PostgreSQL prediction/model persistence, tests,
Docker and CI. XGBoost was selected from the documented comparison with both baselines.
Customer/loan CRUD, a frontend and advanced infrastructure are outside the MVP.

## Architecture

```text
Public CSV -> validation/report -> validated Parquet
  -> grouped train/validation/test split
  -> raw inputs -> derived features -> preprocessing -> estimator
  -> validation metrics + pipeline artifact + run manifest

Planned serving:
FastAPI -> request-to-source mapping -> same pipeline -> probability/risk score
  -> SHAP -> PostgreSQL -> response
```

Details: [architecture](docs/architecture.md), [specification](SPECS.md),
[engineering conventions](CODESTYLE.md).

## Setup

Python 3.12 is the development and CI reference runtime. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) to use the committed lockfile.

```bash
git clone https://github.com/MGimenez059/CreditRisk.git
cd CreditRisk
uv sync --locked --python 3.12 --extra dev --extra viz
```

Download the CSV manually following the [data dictionary](docs/data_dictionary.md)
and save it at `data/raw/credit_risk_dataset.csv`. The dataset is not bundled.

```bash
uv run --locked python scripts/ingest_data.py
uv run --locked python scripts/train_baselines.py
```

The baseline command writes both pipelines, JSON sidecars, `run.json` and
`validation.md` to `models/baselines-v1/`. The manifest records the input file's
SHA-256, split positions, seed, features, estimator parameters, dependency versions
and validation metrics. It does not compute test predictions or test metrics.
Use `--output models/baselines-v2` for another run; existing runs are not overwritten.

Phase 4 development selection (five grouped folds, 12 Optuna trials):

```bash
uv run --locked python scripts/train_model.py
```

This creates `models/selected-v1/` with the selected raw-input pipeline,
metadata, CV/trial evidence, protocol and frozen configuration. For another
**development-only** run, choose a new `--output` directory. Existing runs are
never overwritten. `--initial-only` retains the earlier fixed-candidate workflow.

The final evaluation command is separate:

```bash
uv run --locked python scripts/evaluate_model.py --run models/selected-v1
```

**The recorded run has already been evaluated. Read its saved `test.json` or the
[evaluation report](docs/evaluation_report.md); do not score it again.** The command
refuses repeated attempts in that run. Final evaluation is for a newly frozen,
not-yet-scored experiment under the [selection protocol](docs/selection_protocol.md).
A new output directory does not make this dataset's test results unseen again.

## Model status

Both baselines use class weights and the same grouped split. Identical raw model
inputs remain in one partition, including rows with conflicting labels. The
70/15/15 proportions apply to groups; actual row proportions are reported per run.
Median imputation and scaling are fitted only on training data, followed by
one-hot encoding and classification. Derived features are inside the saved pipeline.

Metrics: ROC-AUC, average precision (`pr_auc`), F1, precision, recall, log loss and
Brier score. Threshold 0.5 is a baseline convention. The selected XGBoost uses
threshold 0.43; sigmoid calibration was evaluated and rejected on separate decision
rows because it worsened log loss. Test ROC-AUC: 0.940042; F1: 0.798310.
See [model card](docs/model_card.md), [evaluation report](docs/evaluation_report.md)
and [machine-readable selection evidence](docs/selection_results.json).

## API contract (serving not yet complete)

Liveness is `GET /health` at the root. Under `/api/v1`, route scaffolds are
`GET /models/active`, `POST /predictions` and `POST /predictions/batch`.
Health reports process availability, not model/database readiness.

Example synthetic request:

```json
{
  "age": 34,
  "income": 60000,
  "employment_years": 6,
  "home_ownership": "RENT",
  "loan_amount": 12000,
  "interest_rate": 12.5,
  "loan_intent": "PERSONAL",
  "credit_history_years": 7,
  "previous_defaults": 0
}
```

`previous_defaults` is a required 0/1 indicator. Employment duration and interest
rate may explicitly be null, matching missing values in the dataset. Unknown
fields are rejected. Unsupported bureau fields are not accepted or fabricated.
Income must be positive. Monetary inputs must use the source dataset's units;
its currency is not verified, so examples are not localized currency conversions.

The intended response includes probability, score, level, model name/version and
SHAP contributions. `risk_score = round(default_probability * 100)`;
0–30 LOW, 31–70 MEDIUM, 71–100 HIGH, based on the rounded score.
These illustrative categories are separate from a classification threshold.
SHAP units/base value and the relationship to calibration must be finalized in Phase 5.

## Local service and Docker

Copy `.env.example` to `.env`. For a local Python process use a PostgreSQL URL
with `localhost`; the example hostname `db` is for Compose networking.

```bash
uv run --locked uvicorn credit_risk.main:app --reload
docker compose up --build
```

Compose defines PostgreSQL, a migration command and the API. No Alembic revisions
exist yet, so `alembic upgrade head` currently creates no application tables.
The selected model is a local artifact, not bundled in Git. SHAP is a placeholder:
a running container does not yet mean predictions work. The configured artifact is selected through
`MODEL_PATH`; `.env.example` lists the remaining configuration.

## Quality

```bash
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src scripts
uv run --locked pytest
```

GitHub Actions runs quality checks and a Docker build for pushes to `main` and
pull requests targeting `main`. Database integration coverage is still pending.
This is CI; automated deployment is outside the current scope.

## Ethical considerations

Educational/portfolio use only. No real lending decisions, automated loan approvals
or claims about an individual's actual creditworthiness. The public dataset's
population, collection process, default observation horizon and synthetic origin
are not independently verified. Do not claim it represents real-world performance.
Use synthetic API examples and no direct personal identifiers. Limitations and
potential bias belong in the [model card](docs/model_card.md).

## Documentation

- [ROADMAP.md](ROADMAP.md): the single source of phase status and acceptance criteria.
- [SPECS.md](SPECS.md): required behavior and scope.
- [CODESTYLE.md](CODESTYLE.md): engineering conventions.
- [Data dictionary](docs/data_dictionary.md): provenance, fields and mappings.
- [Data quality report](docs/data_quality_report.md): regenerated ingestion evidence.
- [Dataset provenance](docs/dataset_provenance.json): source metadata and snapshot hashes.
- [Baseline results](docs/baseline_results.json): verified parameters, environment and metrics.
- [Model card](docs/model_card.md): evaluation status and limitations.
- [Architecture](docs/architecture.md): implementation boundaries and remaining work.

## License

Project code: MIT, see [LICENSE](LICENSE). Dataset redistribution rights are separate.
