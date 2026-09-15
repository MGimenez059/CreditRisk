# Architecture

## Layering

- `api/`: HTTP routing, validation and response mapping.
- `schemas/`: Pydantic request/response contracts.
- `services/`: orchestration; may use ORM entities internally, but does not write queries.
- `repositories/` and `db/`: database queries, models, session and engine.
- `ml/`: framework-independent training, inference, features and serialization.
- `config/`: application settings; command-line scripts may accept explicit path arguments.

Repository protocols support fakes at persistence boundaries. Plain ML functions do
not need an interface hierarchy. Dormant customer CRUD routes still require a service
and integration tests before being enabled; they are outside the MVP.

## Phase 3 training (implemented)

```text
validated Parquet -> raw-input groups -> 70/15/15 group split
 -> training: derived features -> imputation/encoding/scaling -> baseline classifier
 -> validation metrics
 -> joblib pipeline + metadata + split positions + environment/run manifest
```

`ml.train.split_dataset` hashes the nine raw feature columns, excluding the target,
loan_grade, source loan_percent_income and already-derived columns. Identical model
inputs remain together even when labels differ. Stratification uses group-majority
labels (positive on ties); row ratios can differ from 70/15/15. Inspect actual sizes.
No rows are dropped by splitting. This also protects against identical inputs hidden
by differences in unused source fields. Keep this grouping policy in Phase 4 CV.

The saved pipeline calls `ml.features.add_derived_features` before preprocessing.
It accepts raw source fields; extra source/derived columns do not supply independent
features. Imputers, encoders and scaling fit only on training rows.

`scripts/train_baselines.py` saves two pipelines and sidecars, exact input SHA-256,
Python/dependency versions, seed, hyperparameters and row-position split assignments.
The test partition is allocated but never scored by this command. JSON manifests
and artifacts are local outputs; publish reviewed validation summaries in the model card.

## Request flow (implemented)

```text
POST /api/v1/predictions -> PredictionRequest
 -> PredictionService -> explicit API-to-source mapping
 -> saved pipeline -> probability -> risk score
 -> SHAP -> transaction: model registration + predictions -> commit -> response
```

`previous_defaults` is required and restricted to 0/1, translated to N/Y. Four
unsupported bureau/loan fields were removed from the unreleased API and ORM scaffold.
The schema rejects unknown fields and non-finite numbers. Missing employment years
or interest rate are represented explicitly as null and handled by the fitted imputer.

Synchronous route functions run blocking database/model work through FastAPI's
threadpool. Async SQLAlchemy is not required for this MVP. See
[FastAPI concurrency documentation](https://fastapi.tiangolo.com/async/).

Every HTTP request receives a bounded `X-Request-ID`, either propagated from the
caller or generated as a UUID. Structlog context variables attach it to request,
model-load and prediction-commit events without logging applicant payloads. One
completion or failure event records method, path, status and request duration.

`GET /health` is process liveness. `GET /ready` queries the serving tables and mapped
columns without fetching rows, through the repository layer, and fully
loads and validates the configured artifact, including frozen hashes when present.
It performs no model registration or prediction write. The Docker health check uses
readiness so an API with a missing model or unavailable database is not marked healthy.

## Persistence and artifacts

`MODEL_PATH` selects the artifact. Each single/batch request loads it once; prediction
and SHAP share that pipeline. Byte hashes detect changes during loading and reuse
of an existing name/version with different content. Stored JSON preserves all
sidecar metadata, including calibration, threshold and metric partition.
Frozen artifacts are checked against their saved artifact/sidecar hashes.

Services own `Session.begin()`; repositories only query/flush. A transaction-level
PostgreSQL advisory lock serializes registry registration and activation; a unique
partial index permits at most one active model. Inference happens before the lock.
Any insert/commit failure rolls back the entire batch and any activation changes.
`GET /models/active` also reconciles the configured artifact, allowing lazy registration.
All workers in one deployment must share MODEL_PATH and immutable artifact contents.

Predictions can be anonymous. Customer, Loan and CreditHistory are optional retained
scaffolds, not a requirement to ingest the training CSV into PostgreSQL. Training
uses Parquet. Only model and prediction persistence is required for the MVP; a model
version alone is not enough to reconstruct an input that was never retained.

## Phase 4 selection (implemented)

`scripts/train_model.py` uses `ml.selection` and `ml.selection_cv` to compare weighted
and unweighted candidates with five grouped folds inside the original training set.
Twelve seeded Optuna trials tune XGBoost. `ml.calibration` fits a sigmoid on one
half of validation and selects calibration/threshold on the other half. The final
estimator remains fitted only on training rows. The selected run retained the base
XGBoost probabilities, without sigmoid calibration.

`ml.selection_state` records hashes and partitions in `frozen.json`.
`scripts/evaluate_model.py` verifies that state before one test prediction pass;
a persistent exclusive marker prevents accidental repetition within the run.
`--initial-only` preserves the initial comparison through `ml.experiments`.
See [protocol](selection_protocol.md) and [evaluation](evaluation_report.md).

## Packaging verification

`ml.explain` now computes grouped Tree SHAP in log-odds and checks additivity
against both the raw margin and pipeline probability. The service passes the same
loaded pipeline to inference and explanation. See [explainability](explainability.md).
Alembic revision `0001` creates the persistence schema, including the retained
customer/loan/history tables required by existing ORM relationships. Their CRUD
routes remain disabled. PostgreSQL integration tests verify commits, rollback,
concurrent registration, identity conflicts and migration upgrade/downgrade.
See [backend verification](backend.md). Phase 7 verified this path in a freshly built
Compose project with a new PostgreSQL volume, real migration and the selected artifact.
