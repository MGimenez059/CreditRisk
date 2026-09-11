# CreditRisk — Technical Specification

Status is tracked in [ROADMAP.md](ROADMAP.md). Section numbers are retained for existing code references.

## 1. Project Overview

CreditRisk is an educational portfolio project spanning Data Science, ML Engineering
and Backend. Build a reproducible default-classification workflow and serve an
evaluated model with explanations. Non-goals: real lending decisions, direct PII,
automated approvals, arbitrary uploaded training data, authentication/billing and
multi-tenancy. XGBoost is a candidate; model selection remains evidence-based.

## 2. High-Level Architecture

Validated dataset -> grouped split -> complete training pipeline -> evaluation
-> artifact/metadata -> FastAPI -> explanations -> PostgreSQL predictions.
See [architecture](docs/architecture.md) for implemented versus pending behavior.

## 3. Proposed Technology Stack

Python 3.12 reference runtime; Pandas, NumPy, PyArrow; scikit-learn, XGBoost,
SHAP, Optuna and joblib; FastAPI/Pydantic; SQLAlchemy/Alembic/PostgreSQL;
Ruff/MyPy/Pytest; Docker and GitHub Actions. Matplotlib and ipykernel are optional
notebook dependencies. Resolved dependencies live in `uv.lock`. Do not add Polars,
DuckDB or a frontend framework without a use case.

## 4. Repository Structure

`src/credit_risk/` contains api, schemas, services, repositories, db, ml and config.
`scripts/` holds ingestion, training, evaluation and explanation CLIs. `tests/`,
`notebooks/`, `docs/`, `models/` and `data/` have distinct purposes. Do not create
placeholder directories or notebooks merely to reproduce an illustrative tree.

## 5. Data Source

Source: Kaggle `laotse/credit-risk-dataset`. Record source version, download date,
license evidence and target semantics when obtaining the data. Record a content
hash for each training input. Missing provenance must be stated as unknown, not
replaced with invented values. See [data dictionary](docs/data_dictionary.md).

## 6. Canonical Data Model

MVP persistence stores model identity/metadata and predictions linked by model_id.
Predictions may be anonymous. Customer, Loan and CreditHistory are optional retained
scaffolds; they do not require CRUD endpoints or training-data ingestion into SQL.
Unsupported bureau fields are removed. In existing storage scaffolds Loan.amount
and Loan.purpose correspond to API loan_amount and loan_intent.
Migrations and tested transaction boundaries are required before serving works.

## 7. Target Variable

`loan_status`: 0 = no default, 1 = default. This is the local target convention;
verify the source definition and observation horizon. Never include target values
or target-derived information in inputs.

## 8. Feature Engineering

Use the nine source fields in `ml.preprocessing.RAW_FEATURE_COLUMNS`.
The serialized pipeline derives loan_to_income, income_per_employment_year and
credit_age_ratio, then preprocesses and classifies. Income must be positive.
Retain missing employment/rate for training-only imputation. Exclude source
loan_percent_income in favor of the recomputed ratio; rounding may differ.
No debt_to_income or late_payment_rate without actual source information.

## 9. Data Leakage Prevention

Split before fitting imputers, encoders, scalers or models. Group identical raw
model inputs across all partitions and CV folds. Fit learned transformations only
on training rows. Exclude outcome information and verify feature availability at
prediction time. Grade is conservatively excluded; correlation alone does not prove
leakage. Review interest rate as well. Historical EDA used the full dataset; record
that limitation rather than calling the holdout completely unseen.

## 10. Dataset Splitting

Split raw-input groups 70% train, 15% validation, 15% test with random_state=42.
Groups use identical values in the nine model-input columns, excluding labels.
Stratify by group-majority target; positive wins ties. Keep all group members,
including conflicting outcomes, in one partition. Report actual row sizes and
development class balance because group sizes differ. Save positional assignments
with the exact input hash. No final test scoring before model selection is frozen.
Use temporal validation instead if a future dataset supports it.

## 11. Class Imbalance

Measure positive_rate, negative_rate and class_ratio. Both Phase 3 baselines
start with class_weight="balanced". Compare weighting against unweighted models
in Phase 4, especially for probability quality. Threshold optimization is model
agnostic; scale_pos_weight is an XGBoost option. Sampling is not required.

## 12. Baseline Models

Train Logistic Regression and Random Forest using the same split/preprocessing
protocol. Save artifacts, parameters and validation results through
`scripts/train_baselines.py`. A real-data rerun is required after changing splits;
historical numbers must not be reused as new-protocol results.

## 13. Candidate Model — XGBoost

Train XGBClassifier as the next candidate, with a fixed seed and a modest initial
configuration. Compare against both baselines; choose the best justified model,
considering discrimination, probability quality and complexity. Do not require
XGBoost to win to satisfy the project goal. The `scripts/train_model.py --initial-only`
comparison preserves the Phase 3 split and shared preprocessing. The default CLI
performs selection and freezing under [the selection protocol](docs/selection_protocol.md);
final test evaluation is a separate guarded command.

## 14. Hyperparameter Optimization

Use a seeded Optuna sampler after a stable XGBoost baseline. Set an explicit trial
budget and tune a small justified parameter space. Primary objective: ROC-AUC;
report average precision and probability metrics as secondary evidence. Do not
optimize against the test set.

## 15. Cross Validation

Use group-aware stratified CV on development data, preserving the Phase 3 grouping.
Use five folds within the original training partition when group/class counts allow. Fit preprocessing inside each
fold. Report mean and standard deviation for discrimination/probability metrics;
threshold metrics must state the threshold. Ordinary StratifiedKFold alone does
not protect duplicate groups.

## 16. Model Evaluation

Report ROC-AUC, average precision (`pr_auc` = average_precision_score), precision,
recall, F1, log loss and Brier score. State threshold and evaluated partition.
Inspect a calibration curve in Phase 4. Brier score is overall probability error,
not a standalone proof of calibration. Accuracy is not the primary metric.

## 17. Decision Threshold

Use 0.5 for comparable baseline F1/precision/recall. The Phase 4 demo maximizes
F1 on decision rows over 0.05..0.95 in steps of 0.01, breaking ties by precision
then higher threshold. Freeze this choice before final test scoring. No claim that an arbitrary threshold is a lending policy.

## 18. Risk Score

`risk_score = round(default_probability * 100)`. Classify the rounded score:
0–30 LOW, 31–70 MEDIUM, 71–100 HIGH. These illustrative display bands are independent
of the classification threshold and are not an industry credit score.

## 19. Explainability — SHAP

Phase 5: global importance, summary plot and local contributions. Define SHAP
output space, base value, encoded-feature aggregation and additivity checks.
XGBoost raw values are log-odds; do not label them probability increments.
The selected uncalibrated model uses tree_path_dependent Tree SHAP. Return the
base, raw output, method, output space and all grouped contributions as specified
in [explainability](docs/explainability.md). Calibrated wrappers are rejected until
a contract for their output is implemented. Explanations are not causal effects.

## 20. Prediction API

Root liveness: GET /health. Under `/api/v1`: GET /models/active, POST /predictions
and POST /predictions/batch. See README for a request using only supported fields.
Batch input is a JSON array; output wraps predictions in `results`.
Results identify the actual artifact's name/version. Customer CRUD is outside MVP.
The current route scaffolds do not mean operational inference is complete.

## 21. API Validation

Reject unknown fields and non-finite numbers. Require age 18–100, positive income
and loan amount, known categories, non-negative credit history and previous_defaults
as a required 0/1 flag. Employment duration 0–70 and rate 0–100 may be explicitly null.
Invalid requests return HTTP 422. Keep raw-input and serving constraints consistent.

## 22. Database Layer

SQLAlchemy queries belong in repositories. Services orchestrate writes and may use
ORM entities internally; HTTP responses use schemas. Use Alembic, indexed foreign
keys and database constraints. Add tested transactions before completing Phase 6.

## 23. ML Service Architecture

Request -> service -> API-to-source mapping -> full saved pipeline -> probability
-> display score -> explanation -> persistence -> response. ML code has no FastAPI
or database dependency. Use synchronous routes for the synchronous service stack.

## 24. Model Artifact

Serialize feature engineering + preprocessing + estimator together in joblib.
Use a JSON sidecar with name/version, algorithm, input snapshot, feature version,
metrics, training timestamp and runtime/dependency versions. Confirm save/load
preserves raw-input predictions through a regression test.

## 25. Model Versioning

Every deployed artifact has a name/version and immutable provenance. Store the
model_id/version with predictions. An artifact version identifies the model but
does not reconstruct a historical request without its input. Do not overclaim
replayability of anonymous predictions whose inputs are not retained.

## 26. Experiment Tracking

Use local JSON run manifests and reviewed evaluation reports. Record snapshot hash,
seed, feature schema, parameters, split assignments, environment and artifacts.
MLflow is optional post-MVP; no database experiment-tracking service is needed now.

## 27. Testing Strategy

Unit tests cover data cleaning, transformations, grouped splits, raw-input artifact
round trips, API mapping, validation and metrics. Use tmp_path for isolated file I/O.
Integration tests must cover serving plus a dedicated PostgreSQL database before
the API is declared operational. Test meaningful boundaries and failure paths.

## 28. Data Validation

Validate columns/types/nulls/categories/ranges. Structural errors stop ingestion.
Configured row violations are excluded with reasons; more than 5% exclusions stop
the pipeline. Preserve ordinary statistical outliers unless justified otherwise.
Report duplicates and enforce partition isolation when splitting. Undefined ratios
(non-positive income) fail model feature construction instead of creating infinities.

## 29. Data Quality Report

Generate counts, missingness, duplicate count, numerical/categorical distributions,
target distribution, unique values, outliers and a numeric correlation screen.
Label the screen's limits: it does not check categorical leakage or feature timing.
Published reports identify their snapshot/status and are not manually fabricated.

## 30. Reproducibility

Use the committed uv.lock and Python 3.12 reference environment. Record runtime
versions and the input content hash per run. `scripts/train_baselines.py` must work
after environment setup and ingestion with no notebook execution. Record split
positions and report actual sizes. Reproducibility requires more than a seed.

## 31. Docker

Compose provides PostgreSQL, a migration command and the API. Build success is
not a prediction smoke test. Real Alembic revisions and a compatible model/explainer
are required for an operational demo. No frontend container required.

## 32. Environment Variables

Use `.env.example` as the environment variable reference. `db` is the Compose
hostname; use `localhost` for a local Python process. MODEL_PATH selects the loaded
artifact. Do not duplicate model identity from config when artifact metadata exists.

## 33. Logging

Use structured logs for startup, model loading, errors and inference latency/model
version, with request correlation. Never log full applicant payloads or credentials.
CLI progress/errors may use stdout/stderr. Complete request-wide tracing in Phase 7.

## 34. Observability

MVP observability is structured logs, correlation, latency and meaningful readiness
checks. Current /health is liveness only. Advanced telemetry is optional post-MVP.

## 35. Model Monitoring — Future

Data/prediction drift and performance monitoring need a sustained serving use case
and actual outcome availability. They are not Phase 3 or MVP requirements.

## 36. Security / Privacy

Use public non-identifying data and synthetic examples. Do not commit credentials
or direct personal identifiers. Public data does not establish synthetic origin.
Authentication, rate limits and deployment hardening are future deployment work,
not an excuse to claim the demo is production-ready.

## 37. Ethical / Responsible ML

State educational use, uncertain representativeness, potential bias and absence
of real-world/compliance validation. Review sensitive/proxy features before portfolio
sign-off; category correlations cannot prove fairness. No real creditworthiness claims.

## 38. Model Card

Maintain docs/model_card.md with purpose, data/provenance, features, protocol,
model selection, metrics, runtime, limitations and bias considerations. Clearly
separate historical results, current verified results and pending work.

## 39. CI/CD

GitHub Actions installs the locked environment, checks Ruff formatting/lint, runs
MyPy/Pytest and builds Docker only after quality passes. Database integration tests
are added with Phase 6. Automated deployment is not currently implemented.

## 40. Git Strategy

Use focused changes, English Conventional Commits and reviewable descriptions.
Do not manufacture commit history or claim a check ran when it did not.

## 41. Development Milestones Reference

Phase status lives only in [ROADMAP.md](ROADMAP.md). Do not duplicate phase checklists here.

## 42. MVP Definition

Reproducible ingestion/training, a justified selected model, documented evaluation,
SHAP, a working FastAPI prediction endpoint, PostgreSQL prediction/model persistence,
tests, Docker, CI and documentation. No frontend, customer CRUD or production deployment required.

## 43. Definition of Done

See the Definition of Done in ROADMAP.md. Code existence and scaffolds are not
execution evidence. Pending real-data or integration validation remains explicit.

## 44. Suggested First Dataset Schema

The chosen dataset and `RAW_FEATURE_COLUMNS` determine the actual schema, documented
in the data dictionary. Do not impose a generic credit-bureau schema or impute
entirely unavailable fields.

## 45. Success Criteria

Demonstrate sound data handling, leakage-aware evaluation, reproducible artifacts,
model comparison, explanations and maintainable serving. Additional infrastructure
does not compensate for missing evaluation or a non-working prediction path.

## 46. Long-Term Architecture

Optional extensions are listed once in ROADMAP.md. Add infrastructure only when
an actual requirement justifies it; no speculative feature-store architecture.

## 47. Guiding Principle

Prioritize reproducibility, data quality, correct evaluation, explainability,
maintainable code, testing and incremental delivery.

Phase 4 partition roles, calibration selection, tuning budget and final-evaluation
guards are specified in [the selection protocol](docs/selection_protocol.md).
