# Backend serving and verification

## Behavior

`MODEL_PATH` is the serving selector, not the fallback name/version settings.
Each request loads one trusted local joblib pipeline and its sidecar; a batch uses
that same object for every item. Inference and SHAP use the complete fitted raw-input
pipeline. Loading requires an uncalibrated fitted binary XGBoost pipeline and the
canonical feature schema. If the run has `frozen.json`, its artifact/sidecar hashes
must match. Files are hashed before and after loading to reject concurrent changes.
Deploy immutable artifacts; do not replace a joblib and sidecar in place while serving.

A successful single request commits one prediction. A batch accepts 1..500 items
by default (MAX_BATCH_SIZE), preserves input order and commits all items together.
Inference finishes before taking a registry lock. The service then registers or
checks the model, activates it and inserts predictions in one transaction. Any
insert or commit failure rolls back that transaction, including model activation.
A prediction response is returned only after commit succeeds.

The registry stores the binary SHA-256 and full sidecar metadata, including metric
partition, threshold and calibration. Reusing a name/version with different bytes
or metadata is rejected. A PostgreSQL transaction advisory lock serializes registry
changes; a unique partial index permits at most one active row. All workers sharing
this registry must use the same configured deployment artifact.

`GET /api/v1/models/active` loads and reconciles the configured model too, so its
first use can register/activate it. The response reports actual name, version,
algorithm, ROC-AUC (null when unavailable), threshold, calibration and metric partition.
The registry is descriptive; changing its active flag does not select another file.

Missing files return 404. Unsupported/corrupt artifacts and SHAP failures return
500. Invalid payloads and empty/oversized batches return 422. Database failures
return 503 with a generic message; SQL, credentials and applicant fields are not
included in that response or error log.

`GET /health` reports liveness only. `GET /ready` checks the mapped columns of both
serving tables with zero-row queries against PostgreSQL
and loads the configured model through the same compatibility and hash validation used
for predictions. It does not register a model or write a prediction. A failed check
returns a generic 503. Docker uses this endpoint for its container health check.

Every request returns `X-Request-ID`. Safe caller identifiers up to 128 characters are
propagated; missing or unsafe values are replaced with a UUID. The identifier is bound
to all structured application events in that request, including model loading and the
post-commit prediction event. Request logs contain method, path, status and duration,
but never the request body or applicant fields. Handled and unexpected failures are
logged and unexpected responses use a generic 500 body.
Unexpected exception messages and tracebacks are excluded from logs because they may
contain sensitive values. The request event retains the exception type and request ID.
Readiness checks schema readability and artifact compatibility, not write permissions
or a complete inference/SHAP round trip; the container demo verifies that full path.

## Schema and setup

Revision `0001` creates `models` and `predictions`, with foreign keys, indexes and
probability/score/latency checks. `training_dataset` holds the full `sha256:` prefix
plus 64 hexadecimal characters. Explanation JSONB retains the base, units, method
and every grouped signed contribution. Anonymous input payloads are not stored.
Historical explanations are retained, but replaying an original request is not promised.

The migration also includes the existing customer, loan and credit-history tables
because the retained ORM relationships and nullable customer FK reference them.
Their CRUD endpoints remain disabled and no applicant records are created by inference.
Downgrade removes the tables and PostgreSQL enums; it is destructive and is only
exercised against isolated test schemas during verification.

For a local Python process, configure `.env` with your PostgreSQL URL (localhost,
not the Compose hostname `db`) and the selected artifact path, then run:

```bash
uv run --locked alembic upgrade head
uv run --locked uvicorn credit_risk.main:app --reload
```

Open `/docs` to submit the synthetic request in README to `/api/v1/predictions`.
For `/api/v1/predictions/batch`, send a JSON array of such requests.
Compose runs `alembic upgrade head` before starting the API; its model mount is read-only.
The model artifact and dataset remain local and are not bundled in Git.

## Integration tests

A separate disposable PostgreSQL instance can be started with:

```bash
docker run --detach --name credit-risk-phase6-test --publish 127.0.0.1:55432:5432 --env POSTGRES_USER=creditrisk --env POSTGRES_PASSWORD=creditrisk --env POSTGRES_DB=creditrisk_test postgres:16-alpine
```

If that test container already exists, use `docker start credit-risk-phase6-test`.
Use only dedicated test databases, never the development/production database.
In PowerShell:

```powershell
$env:TEST_DATABASE_URL = "postgresql+psycopg://creditrisk:creditrisk@127.0.0.1:55432/creditrisk_test"
$testTemp = ".pytest-tmp/run-" + [guid]::NewGuid().ToString("N")
uv run --locked pytest --basetemp=$testTemp --cov --cov-report=term-missing
```

On Linux, export the same TEST_DATABASE_URL and run `uv run --locked pytest --cov`.
Without TEST_DATABASE_URL, integration tests explicitly skip. An explicitly invalid
or unreachable URL fails the tests. The database name must end in `_test`; each test
creates a unique random schema, applies real Alembic migrations and drops only its
own schema afterward. The fixture overrides HTTP dependencies with sessions connected
to that schema. It trains a small synthetic XGBoost fixture, not the real dataset.
CI supplies a PostgreSQL service and TEST_DATABASE_URL, so these tests run on every
configured push/PR. Tests verify migration parity, downgrade/re-upgrade, persisted
responses, batch order, concurrent first registration, artifact/metadata conflicts,
SHAP failures, database constraint failures and commit rollback.

## Phase 6 execution evidence (2026-09-14)

A local FastAPI TestClient run against PostgreSQL 16 used the frozen `selected-v1`
artifact and freshly migrated isolated schema. Single and two-item batch endpoints
returned HTTP 200 and committed three predictions linked to one model. The active
model endpoint agreed with the response and stored artifact hash.

The synthetic request from the SHAP report (age 34, income 60000, amount 12000,
RENT/PERSONAL, history 7, prior default 1, null employment and interest rate) returned
probability 0.10521303117275238. The matching batch item returned the identical result.
Hashes of every selected-run file were unchanged afterward; no final-test scoring
or model training took place in this smoke run. Its temporary schema was removed.

This verified HTTP routing and real PostgreSQL persistence in the local Python
runtime. At the time of that check, a clean-container demo, correlation and readiness
were still pending; the Phase 7 evidence below verifies them. `/health` remains
liveness only.

Quality verification at the Phase 6 checkpoint covered 141 tests: 118 unit tests
and 23 PostgreSQL integration tests. The 140-test coverage run reported 82%, followed
by a passing inference-failure regression. Ruff formatting/lint
and strict MyPy passed. Four existing dependency deprecations remain visible
(Starlette and optional SHAP/Matplotlib); no warnings are suppressed.

A second smoke run used the normal application dependencies (no HTTP dependency
overrides) after CLI `alembic upgrade head` and `alembic check` against the dedicated
test database. Single/batch/active endpoints returned 200, three synthetic predictions
were committed, and selected-run hashes remained unchanged. Verification used
synthetic requests in the dedicated test database.

The runtime Docker image also built successfully as `credit-risk:phase6`.

## Phase 7 clean-container evidence (2026-09-15)

Compose project `credit-risk-phase7-demo` was built from the current Dockerfile and
started with a newly created network and PostgreSQL volume. The migration service
completed revision `0001` before the API started. The selected-v1 directory was mounted
read-only; the artifact is not embedded in the image.

`GET /health`, `GET /ready`, `POST /api/v1/predictions` and
`GET /api/v1/models/active` all returned HTTP 200 from the containers. Readiness named
`credit-risk-xgboost` version `selected-v1`. The SHAP report's synthetic request returned
probability `0.10521303117275238`, score 11 and LOW, with the complete SHAP explanation.
PostgreSQL contained one model, one prediction and Alembic revision `0001` afterward.
Hashes of all selected-v1 files were unchanged.
The rebuilt final API image remained healthy via `/ready`. A subsequent two-item
batch returned 200, loaded the model once and brought the prediction count to three
without creating another model row.

The caller identifier `phase7-clean-demo` appeared on the response and on the matching
request-start, model-load, prediction-commit and request-completion JSON events. A
separate invalid request returned 422 and emitted `request_failed` with its correlation
identifier; its body and applicant field names were absent from application logs.

Final locked-environment verification passed Ruff format and lint, strict MyPy and
149 Pytest tests, including PostgreSQL integration tests, with 83% total coverage.
The four visible third-party deprecation warnings remain unchanged and unsuppressed.

Follow-up review on 2026-09-15 replaced the connectivity-only probe with zero-row
queries for serving tables/columns and sanitized unexpected exceptions at the HTTP
middleware boundary. A missing-column integration test verifies readiness 503 while
liveness remains 200. An unexpected-error regression verifies one correlated failure
event without the sensitive exception message and without propagating to the server.
All 150 tests passed with a fresh ignored temporary directory, along with Ruff and
MyPy. These review changes were verified by the test suite; the container execution
above predates them. To build and run the current checkout, use
`docker compose up --build -d` after completing the setup instructions.
