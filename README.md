# CreditRisk

A portfolio project for estimating loan default probability and demonstrating
Data Science, ML Engineering, and Backend development with public tabular data.

**Current status: educational MVP complete.** Grouped cross-validation, bounded Optuna
tuning, calibration comparison and threshold selection are implemented. A selected
XGBoost pipeline was frozen and evaluated once on test. Global and local SHAP
explanations, transactional PostgreSQL prediction serving, readiness and correlated
request logging are implemented. See [ROADMAP.md](ROADMAP.md).

## Scope

The MVP is a reproducible training workflow, an evaluated model, SHAP explanations,
a FastAPI prediction endpoint, PostgreSQL prediction/model persistence, tests,
Docker and CI. XGBoost was selected from the documented comparison with both baselines.
Customer/loan CRUD, a frontend and advanced infrastructure are outside the MVP.

## Explore the project

Start with the [model comparison and evaluation](docs/evaluation_report.md), then
the [explanation report](docs/explainability.md) and [working API demo](docs/backend.md).
The [model card](docs/model_card.md) explains intended use, feature choices and limitations.
These reports and images can be viewed directly on GitHub without installing Python.

The global chart summarizes model attributions on 500 development examples:

![Global SHAP importance and signed contributions](docs/shap_global_summary.png)

The local chart explains one synthetic example. Contributions are in log-odds,
not probability percentage points; they describe model behavior, not causal effects.

![SHAP explanation for one synthetic example](docs/shap_local_explanation.png)

To interact with the API, complete the setup below and start Compose, then open
[Swagger UI](http://localhost:8000/docs). Expand `POST /api/v1/predictions`, click
**Try it out**, paste the synthetic JSON below and click **Execute**. A successful
response shows probability, display score, model identity and SHAP contributions.
The startup needs a local trained artifact and sidecar; cloning alone does not
include them. Training instructions below create them from the downloaded dataset.
Swagger is a technical browser interface; a custom visual demo and public hosting
are possible follow-up work.

## Architecture

```text
Public CSV -> validation/report -> validated Parquet
  -> grouped train/validation/test split
  -> raw inputs -> derived features -> preprocessing -> estimator
  -> validation metrics + pipeline artifact + run manifest

Serving:
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

## API contract

Liveness is `GET /health` at the root. `GET /ready` verifies PostgreSQL serving columns
and the configured immutable serving artifact without changing registry state.
Under `/api/v1`, endpoints are
`GET /models/active`, `POST /predictions` and `POST /predictions/batch`.
Successful and failed responses carry `X-Request-ID`. A bounded caller-provided value
is propagated; otherwise the application generates a UUID for request-wide logs.

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

The response includes probability, score, level, model name/version and
SHAP contributions. `risk_score = round(default_probability * 100)`;
0–30 LOW, 31–70 MEDIUM, 71–100 HIGH, based on the rounded score.
These illustrative categories are separate from a classification threshold.
The explanation object includes its base, raw output and all grouped contributions
in log-odds. See [explainability](docs/explainability.md) for semantics and plots.

## Local service and Docker

Copy `.env.example` to `.env`. For a local Python process use a PostgreSQL URL
with `localhost`; the example hostname `db` is for Compose networking.

```bash
uv run --locked uvicorn credit_risk.main:app --reload
docker compose up --build
```

For a local Python process, set `DATABASE_URL` to your local PostgreSQL instance
and apply the schema before starting the API:

```bash
uv run --locked alembic upgrade head
```

Compose applies the same migration before starting the API. The selected model
is a local artifact, not bundled in Git. `MODEL_PATH` must point to its joblib file
with the matching JSON sidecar. Only trusted local artifacts should be loaded.
The backend verifies the frozen hashes when `frozen.json` is present.
See [backend verification and transaction semantics](docs/backend.md).
The container health check calls `/ready`, while `/health` remains process liveness.
A clean Compose demo with the selected artifact is recorded in the backend guide.

## Quality

```bash
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src scripts
uv run --locked pytest
```

If Windows denies access to `Temp/pytest-of-<user>`, use a fresh test directory
inside the ignored `.pytest-tmp/` folder. In PowerShell:

```powershell
$testTemp = ".pytest-tmp/run-" + [guid]::NewGuid().ToString("N")
uv run --locked pytest --basetemp=$testTemp
```

GitHub Actions runs quality checks and a Docker build for pushes to `main` and
pull requests targeting `main`, including PostgreSQL integration tests.
This is CI; automated deployment is outside the current scope.

To run integration tests locally, create a dedicated PostgreSQL database whose
name ends in `_test` and set `TEST_DATABASE_URL` explicitly. See
[the backend guide](docs/backend.md) for the isolated Docker setup. Tests create
and remove their own random schemas. Without this variable, integration tests
are skipped; unit tests still run. CI always sets it.

## Ethical considerations

Educational/portfolio use only. No real lending decisions, automated loan approvals
or claims about an individual's actual creditworthiness. Dataset provenance and the
default observation horizon have documented limitations; results apply to this
dataset, with no claim of real-world or Argentine-population validation.
See the [provenance review](docs/data_dictionary.md#provenance-review-and-scope) and
[model card](docs/model_card.md) for details. API examples use synthetic inputs
without direct personal identifiers.

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
