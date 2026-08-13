# Architecture

## Layering

```
api/            HTTP only: routing, request validation, response mapping.
                No business logic. Depends on services/ and schemas/.
services/       Business logic and orchestration. No FastAPI or SQLAlchemy
                imports. Depends on repositories/ (via Protocol) and ml/.
repositories/   The only place SQLAlchemy queries are written. Returns ORM
                models to services/ — never returns them to api/.
db/             Engine, session, declarative base, ORM models.
schemas/        Pydantic I/O contracts. Independent of db/ — see the enum
                duplication note in db/models/enums.py.
ml/             Training, inference, explainability. No FastAPI or database
                dependency; must remain importable and testable standalone.
config/         The single Settings object. No other module reads os.environ.
```

A request never skips a layer: `api/routes/predictions.py` calls
`PredictionService`, which calls `ml.predict` and a
`PredictionRepositoryProtocol` implementation — it does not import
SQLAlchemy or `ml.registry.load_model_artifact`'s internals directly. This
mirrors CODESTYLE.md §3 exactly.

## Request flow (single prediction)

```
POST /api/v1/predictions
  → Pydantic validates PredictionRequest
  → api/routes/predictions.create_prediction
  → services/prediction_service.PredictionService.predict
      → ml/registry.load_model_artifact (raises ModelNotFoundError if untrained)
      → PredictionService._resolve_model_metadata (get-or-register the models-table row)
      → ml/predict.predict (raises InvalidFeatureSchemaError on schema mismatch)
      → services/risk_service.calculate_risk_score
      → services/explanation_service.ExplanationService.explain (Phase 5)
      → repositories/prediction.SQLAlchemyPredictionRepository.add
  → api/routes/predictions._to_response maps PredictionResult → PredictionResponse
  → api/exception_handlers translates any CreditRiskError to an HTTP response
```

## Key design decisions

**Model artifacts are self-describing.** `ml.registry` pairs every
`.joblib` pipeline with a `.json` metadata sidecar (name, version, algorithm,
dataset version, feature version, metrics, training timestamp).

**`Prediction.model_id` is a normalized FK into the `models` table, per
SPECS.md §6** — not the denormalized `model_name`/`model_version` string
columns an earlier draft of this schema used. Since the sidecar JSON (read
from `Settings.model_path`) is the operational source of truth for "what
artifact is currently configured," and the `models` table is the
historical registry SPECS.md §26 (Experiment Tracking) describes,
`PredictionService._resolve_model_metadata` bridges the two: it looks up a
`models` row matching the loaded artifact's `(name, version)`, and
registers one automatically on first use if none exists yet. This means
`scripts/train_model.py` is not a strict prerequisite for the first
prediction to succeed once an artifact file is manually placed at
`Settings.model_path` — a deliberate simplification for a portfolio-scale
project. Promoting a different model to "active" (flipping
`ModelMetadata.is_active`) remains a distinct, explicit operation, per
`ModelMetadata`'s docstring.

**Repositories are `Protocol`s, not base classes.** `services/` depends on
`repositories.interfaces.PredictionRepositoryProtocol` and
`ModelRepositoryProtocol`, so a unit test can hand a service an in-memory
fake with the same method signatures, with no inheritance and no test
database required.

**Sync SQLAlchemy inside `async def` route handlers.** Route handlers are
declared `async def` to match the API reference examples in `README.md`, but
`PredictionService` and the repositories underneath it are synchronous
(`sqlalchemy.orm.Session`, not `AsyncSession`). This means a DB-touching
request blocks the event loop for the duration of that call. This is a
deliberate, documented trade-off for a portfolio-scale service, not an
oversight — revisit if this project is ever load-tested under concurrent
traffic; the fix is switching to `sqlalchemy.ext.asyncio` end-to-end, not a
partial patch.

**Nullable columns instead of a narrower schema.** `CreditHistory` and
`Loan` include four columns (`late_payments`, `credit_utilization`,
`active_credit_lines`, `term_months`) that have no source in the Phase 0
Kaggle dataset. They are kept nullable rather than removed because the
public API contract in `README.md` already documents them. See
`docs/data_dictionary.md` for the full gap analysis and
`docs/model_card.md` for how the trained pipeline ultimately handles them.

**`Prediction.customer_id` replaces an earlier `loan_id` FK.** SPECS.md §6
ties predictions to a (nullable) customer, not a loan — `POST /predictions`
is designed to work for ad-hoc, anonymous applications with no persisted
`Customer`/`Loan` row at all, which is exactly today's MVP flow. A `Loan`
row (with its own historical `loan_status` label) is a separate concept:
training data ingested in Phase 2, or a future `customers`/`loans` CRUD
API — not something every live prediction request creates.

**`api/routes/customers.py` exists but is not mounted in `main.py`.**
SPECS.md §4 lists it as part of the Phase 1 file structure; ROADMAP.md
bundles the *live* `customers` endpoint into Phase 6 alongside the rest of
the persistence layer. The handlers are fully implemented against
`repositories/customer.py` (not stubs), so wiring them in is a one-line
`app.include_router(...)` addition in `main.py` when Phase 6 starts.

## TODO convention

CODESTYLE.md §8 requires every `TODO` to reference an issue. This project
does not yet have an issue tracker, so TODOs reference `ROADMAP.md` phases
instead, e.g. `# TODO(ROADMAP-P4): ...`. Replace this convention with real
issue references once the project has a tracker — this is a known,
temporary substitution, not the long-term convention.

## What's deliberately not implemented yet

Every module under `ml/` other than `ml.registry` and `ml.predict` raises
`NotImplementedError` with a docstring pointing at the roadmap phase that
implements it. This is intentional: the Phase 1 goal is a skeleton that
fails loudly and traceably when asked to do Phase 3-5 work, not a skeleton
that fakes results. See `ROADMAP.md` for the phase breakdown and
`SPECS.md` for the full technical specification this skeleton was built
against.
