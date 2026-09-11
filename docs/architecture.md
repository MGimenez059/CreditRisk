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

## Request flow (partially implemented)

```text
POST /api/v1/predictions -> PredictionRequest
 -> PredictionService -> explicit API-to-source mapping
 -> saved pipeline -> probability -> risk score
 -> SHAP (implemented) -> prediction repository -> response
```

`previous_defaults` is required and restricted to 0/1, translated to N/Y. Four
unsupported bureau/loan fields were removed from the unreleased API and ORM scaffold.
The schema rejects unknown fields and non-finite numbers. Missing employment years
or interest rate are represented explicitly as null and handled by the fitted imputer.

Synchronous route functions run blocking database/model work through FastAPI's
threadpool. Async SQLAlchemy is not required for this MVP. See
[FastAPI concurrency documentation](https://fastapi.tiangolo.com/async/).

## Persistence and artifacts

`MODEL_PATH` selects the operational artifact. A JSON sidecar identifies its model
and data/feature versions. The service resolves a matching models-table row before
storing predictions. The `is_active` database flag is scaffold state, not the serving
selector; Phase 6 must make promotion and transaction behavior consistent.
Loading/registration currently happens per request; lifecycle loading can be addressed
when completing serving, with tests for missing or incompatible artifacts.

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

## Remaining work

`ml.explain` now computes grouped Tree SHAP in log-odds and checks additivity
against both the raw margin and pipeline probability. The service passes the same
loaded pipeline to inference and explanation. See [explainability](explainability.md).
Alembic has no revisions yet; Compose runs
its migration command but creates no application tables. Integration tests and operational prediction
serving are not complete. Follow [ROADMAP.md](../ROADMAP.md), not file presence, for status.
