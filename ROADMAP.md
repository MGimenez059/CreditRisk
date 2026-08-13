# CreditRisk — Work Roadmap

> Based on `SPECS.md`. Guiding principle: **build a real ML system, not a notebook with an API glued on.**

---

## Phase 0 — Planning
- [ ] Define MVP scope and non-goals (no real PII, no automated loan approval)
- [ ] Select a public credit risk dataset (Kaggle / UCI / OpenML / LendingClub-derived)
- [ ] Document dataset provenance (source, version, license, URL, record count, target)
- [ ] Define the binary target `loan_status` (0 = no default, 1 = default) and verify the real mapping
- [ ] Define initial features (see suggested schema: age, income, employment, amount, rate, credit history, etc.)

## Phase 1 — Repository
- [ ] Initialize a Python 3.12+ project with `pyproject.toml`
- [ ] Create the `src/credit_risk/` package structure (api, config, db, schemas, repositories, services, ml)
- [ ] Configure Ruff, MyPy, and pre-commit
- [ ] Configure Pytest
- [ ] Configure environment variables (`.env.example`)
- [ ] Write the initial README

## Phase 2 — Data
- [ ] Download and store the dataset (evaluate whether to commit it based on license/size)
- [ ] Implement the ingestion script (`scripts/ingest_data.py`)
- [ ] Validate the dataset schema (types, ranges, nulls)
- [ ] Generate a data quality report
- [ ] Perform EDA (notebook `01_data_exploration.ipynb`)
- [ ] Document the data dictionary (`docs/data_dictionary.md`)
- [ ] Prevent data leakage (exclude post-outcome variables from the feature set)

## Phase 3 — ML Baseline
- [ ] Create a train/validation/test split (stratified by target)
- [ ] Build the preprocessing and feature engineering pipeline (`ml/preprocessing.py`, `ml/features.py`)
- [ ] Handle class imbalance
- [ ] Train Logistic Regression as a baseline
- [ ] Evaluate the baseline with discrimination metrics (ROC-AUC, PR-AUC)
- [ ] Add Random Forest as a second baseline

## Phase 4 — XGBoost (primary model)
- [ ] Train the first XGBoost model
- [ ] Evaluate (ROC-AUC, PR-AUC, F1, Brier score)
- [ ] Add cross-validation
- [ ] Hyperparameter tuning with Optuna
- [ ] Compare against the baselines
- [ ] Select the candidate model and define the decision threshold
- [ ] Calibrate probabilities
- [ ] Define the risk score derived from the probability

## Phase 5 — Explainability
- [ ] Integrate SHAP (`ml/explain.py`)
- [ ] Global feature importance
- [ ] Local explanations (per individual prediction)
- [ ] Define the explanation schema for the API

## Phase 6 — Backend
- [ ] Stand up PostgreSQL
- [ ] Define SQLAlchemy models: Customer, Loan, CreditHistory, Prediction, Model
- [ ] Configure Alembic (migrations)
- [ ] Implement repositories (repository pattern)
- [ ] Implement `prediction_service`, `risk_service`, `explanation_service`
- [ ] Implement FastAPI endpoints: health, predictions (single + batch), customers, models
- [ ] Validate requests/responses with Pydantic v2
- [ ] Persist model metadata (`registry.py`, `models` table)

## Phase 7 — Productionization
- [ ] Dockerfile for the API
- [ ] Docker Compose (API + PostgreSQL)
- [ ] Structured logging
- [ ] Centralized error handling
- [ ] Health checks
- [ ] Integration tests (API + DB)

## Phase 8 — Testing & Quality
- [ ] Unit tests (preprocessing, features, services)
- [ ] Integration tests (endpoints, DB)
- [ ] Reusable fixtures (`tests/fixtures/`)
- [ ] Verify training reproducibility (seeds, data/artifact versioning)

## Phase 9 — CI/CD
- [ ] GitHub Actions: install dependencies → Ruff → MyPy → Pytest → build Docker
- [ ] Fail the pipeline if tests or quality checks fail
- [ ] (Future) integration tests + security scan + deploy

## Phase 10 — Documentation & Portfolio
- [ ] Model card (`docs/model_card.md`): purpose, intended use, data, metrics, limitations, biases
- [ ] Document the architecture (`docs/architecture.md`)
- [ ] Clean README with architecture diagram and demo/screenshots
- [ ] Keep a clean Git history with semantic commits (`feat:`, `fix:`, `docs:`, `ci:`)
- [ ] Review ethical/responsible considerations (do not claim real creditworthiness, review sensitive proxies)

## Phase 11 — Dashboard (optional, post-MVP)
- [ ] Risk distribution
- [ ] Individual prediction form
- [ ] Prediction + SHAP visualization
- [ ] Model metrics
- [ ] Batch predictions

## Phase 12 — Advanced (future, out of MVP scope)
- [ ] MLflow for experiment tracking
- [ ] Redis + Celery/RQ for asynchronous batch jobs
- [ ] MinIO/S3 for model artifacts
- [ ] Prometheus/Grafana for observability
- [ ] Feature store
- [ ] Data drift / prediction drift detection
- [ ] Automated retraining pipeline

---

## MVP Definition
The system is functional end-to-end when:

```
Dataset → Validation → Feature Engineering → XGBoost → Evaluation
→ Saved model → FastAPI → POST /predictions → Probability + Risk Score + Explanation
```

Must include: reproducible training, XGBoost model, documented evaluation,
SHAP explanation, FastAPI prediction endpoint, PostgreSQL, tests, Docker, and this README.
**The frontend is not required for the MVP.**

## Definition of "Done" (per feature)
- Implementation exists
- Unit tests where applicable
- Lint/type checks pass
- Documentation updated
- No committed secrets
- Follows the project architecture
- Reproducible behavior
- Commit clearly describes the change
