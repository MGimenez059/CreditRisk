# CreditRisk — Technical Specification

## 1. Project Overview

**CreditRisk** is an end-to-end machine learning system for estimating the probability that a borrower will default on a loan.

The project is designed as a portfolio-grade **Data Science / ML Engineering / Backend** project rather than a standalone notebook.

### Primary goals

- Build a reproducible data ingestion and preprocessing pipeline.
- Perform exploratory data analysis and data quality checks.
- Engineer predictive features from borrower and loan data.
- Train and compare baseline and tree-based classification models.
- Use **XGBoost** as the primary production model.
- Evaluate discrimination, calibration, and business-relevant metrics.
- Explain individual predictions with **SHAP**.
- Persist application data and model metadata in PostgreSQL.
- Expose predictions through a FastAPI REST API.
- Support single and batch predictions.
- Containerize the system with Docker.
- Automate testing and quality checks with GitHub Actions.
- Maintain a clean Git history showing incremental development.

### Non-goals for MVP

- Real financial institution integration.
- Real customer PII.
- Automated loan approval.
- Real monetary lending decisions.
- Production-grade authentication/billing/multi-tenancy.
- Training directly from arbitrary user-uploaded datasets.

---

# 2. High-Level Architecture

```text
                    ┌────────────────────┐
                    │   Public Dataset   │
                    │  CSV / Parquet     │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Data Validation     │
                    │ + Profiling         │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Feature Engineering│
                    │ + Preprocessing     │
                    └─────────┬──────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
          ┌─────────────────┐   ┌─────────────────┐
          │ Training        │   │ Evaluation      │
          │ XGBoost         │   │ Metrics         │
          └────────┬────────┘   └────────┬────────┘
                   │                     │
                   └──────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ Model Artifact      │
                    │ + Metadata          │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ FastAPI             │
                    │ Prediction Service  │
                    └───────┬───────┬────┘
                            │       │
                 ┌──────────┘       └──────────┐
                 ▼                             ▼
        ┌─────────────────┐           ┌─────────────────┐
        │ PostgreSQL      │           │ Web Dashboard   │
        │ Customers       │           │ Analytics       │
        │ Loans           │           │ Predictions     │
        │ Predictions     │           │ Explanations    │
        └─────────────────┘           └─────────────────┘
```

---

# 3. Proposed Technology Stack

## Backend

- Python 3.12+
- FastAPI
- Pydantic v2
- SQLAlchemy 2.x
- Alembic
- PostgreSQL
- Uvicorn

## Data

- Polars
- Pandas where ecosystem compatibility requires it
- NumPy
- PyArrow
- DuckDB for analytical/local data exploration when useful

## Machine Learning

- XGBoost
- scikit-learn
- SHAP
- Optuna for hyperparameter optimization
- joblib for model artifact serialization

## Visualization

- Plotly
- Optional frontend: React + TypeScript

## Quality / Tooling

- Pytest
- Ruff
- MyPy
- pre-commit
- GitHub Actions

## Infrastructure

- Docker
- Docker Compose
- PostgreSQL container
- FastAPI container

## Optional later additions

- MLflow for experiment tracking
- Redis + Celery/RQ for asynchronous batch jobs
- MinIO/S3-compatible storage for model artifacts
- Prometheus/Grafana for observability

---

# 4. Repository Structure

```text
credit-risk/
│
├── src/
│   └── credit_risk/
│       ├── __init__.py
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── dependencies.py
│       │   └── routes/
│       │       ├── health.py
│       │       ├── predictions.py
│       │       ├── customers.py
│       │       └── models.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       │
│       ├── db/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── session.py
│       │   └── models/
│       │       ├── customer.py
│       │       ├── loan.py
│       │       ├── prediction.py
│       │       └── model.py
│       │
│       ├── schemas/
│       │   ├── customer.py
│       │   ├── loan.py
│       │   ├── prediction.py
│       │   └── model.py
│       │
│       ├── repositories/
│       │   ├── customer.py
│       │   ├── loan.py
│       │   ├── prediction.py
│       │   └── model.py
│       │
│       ├── services/
│       │   ├── prediction_service.py
│       │   ├── risk_service.py
│       │   └── explanation_service.py
│       │
│       ├── ml/
│       │   ├── preprocessing.py
│       │   ├── features.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   ├── predict.py
│       │   ├── explain.py
│       │   └── registry.py
│       │
│       └── main.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_analysis.ipynb
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── scripts/
│   ├── ingest_data.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── alembic/
├── docs/
│   ├── model_card.md
│   ├── data_dictionary.md
│   └── architecture.md
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── docker/
│   └── ...
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
└── SPECS.md
```

---

# 5. Data Source

## Initial source

The initial model will use a **public, anonymized credit-risk dataset**.

The exact dataset must be selected before implementation begins.

### Dataset requirements

The dataset should contain:

- borrower demographic information
- income information
- employment information
- loan information
- credit history variables
- a binary default/loan-status target
- enough records to support train/validation/test splits
- no direct PII

### Candidate sources

- Kaggle
- UCI Machine Learning Repository
- OpenML
- Public academic datasets
- Public LendingClub-derived datasets where licensing and availability permit

### Data provenance

The repository must document:

```text
Source:
Dataset:
Version:
Download date:
License:
Original URL:
Number of records:
Target variable:
Known limitations:
```

The raw dataset must **not** be committed to Git if its license or size makes that inappropriate.

---

# 6. Canonical Data Model

The application will maintain an internal relational model independent from the original dataset schema.

## Customer

```text
customers
---------
id UUID PK
age INTEGER
income NUMERIC
employment_years NUMERIC
home_ownership VARCHAR
created_at TIMESTAMP
updated_at TIMESTAMP
```

## Loan

```text
loans
-----
id UUID PK
customer_id UUID FK
amount NUMERIC
interest_rate NUMERIC
term_months INTEGER
purpose VARCHAR
grade VARCHAR NULL
loan_status INTEGER
created_at TIMESTAMP
```

## Credit History

```text
credit_histories
----------------
id UUID PK
customer_id UUID FK
credit_history_years NUMERIC
late_payments INTEGER
previous_defaults INTEGER
credit_utilization NUMERIC
active_credit_lines INTEGER
created_at TIMESTAMP
```

## Prediction

```text
predictions
-----------
id UUID PK
customer_id UUID NULL
model_id UUID FK
default_probability NUMERIC
risk_score INTEGER
risk_level VARCHAR
prediction_version VARCHAR
created_at TIMESTAMP
```

## Model

```text
models
------
id UUID PK
name VARCHAR
version VARCHAR
algorithm VARCHAR
training_dataset VARCHAR
roc_auc NUMERIC
pr_auc NUMERIC
f1 NUMERIC
brier_score NUMERIC NULL
artifact_path VARCHAR
is_active BOOLEAN
created_at TIMESTAMP
```

---

# 7. Target Variable

Primary target:

```text
loan_status
```

Binary classification:

```text
0 = No default
1 = Default
```

The exact mapping must be verified against the selected dataset.

The target must never be included as a model feature.

---

# 8. Feature Engineering

Initial feature groups:

## Borrower features

- age
- income
- employment years
- home ownership

## Loan features

- loan amount
- interest rate
- term
- loan purpose
- grade

## Credit history

- credit history length
- previous defaults
- late payments
- credit utilization
- active credit lines

## Derived features

Examples:

```text
loan_to_income =
    loan_amount / income

debt_to_income =
    total_debt / income

income_per_employment_year =
    income / max(employment_years, 1)

credit_age_ratio =
    credit_history_years / max(age, 1)

late_payment_rate =
    late_payments / max(credit_history_length, 1)
```

Feature engineering must be implemented as reusable Python code rather than only notebook cells.

---

# 9. Data Leakage Prevention

This is a critical requirement.

The following rules must be enforced:

1. Split data before fitting transformations that learn parameters.
2. Fit preprocessing only on the training set.
3. Never use post-loan outcome information as a feature.
4. Never include the target or target-derived variables.
5. Avoid features that would only be available after the credit decision.
6. Keep train/validation/test datasets isolated.
7. Document suspicious features during EDA.

The final pipeline should use scikit-learn-compatible transformers where possible.

---

# 10. Dataset Splitting

Default split:

```text
Train       70%
Validation  15%
Test        15%
```

For classification:

```python
train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)
```

The final test set must remain untouched until model selection is complete.

If the selected dataset contains meaningful temporal information, a **time-based split** should be preferred over a random split.

---

# 11. Class Imbalance

Credit default datasets commonly contain fewer defaults than non-defaults.

The project must explicitly measure:

```text
positive_rate
negative_rate
class_ratio
```

Potential strategies:

- `scale_pos_weight`
- class weights
- threshold optimization
- stratified cross-validation

Oversampling techniques such as SMOTE are optional and must only be applied inside the training pipeline to avoid leakage.

---

# 12. Baseline Models

Before XGBoost, train simple baselines.

Required:

### Logistic Regression

Purpose:

- interpretable baseline
- sanity check
- comparison against nonlinear model

Optional:

### Random Forest

Purpose:

- tree-based baseline
- compare ensemble performance

The final report must compare all models using the same validation protocol.

---

# 13. Primary Model — XGBoost

Primary algorithm:

```python
XGBClassifier
```

Initial configuration should prioritize reproducibility.

Example starting parameters:

```python
XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
```

These values are starting points, not final values.

The final configuration must be selected using validation data/cross-validation.

---

# 14. Hyperparameter Optimization

Use **Optuna** after a stable baseline exists.

Candidate parameters:

```text
n_estimators
max_depth
learning_rate
min_child_weight
subsample
colsample_bytree
gamma
reg_alpha
reg_lambda
```

Optimization objective:

```text
Primary:
ROC-AUC

Secondary:
PR-AUC
Brier Score
F1
Calibration
```

The optimization process must be reproducible with fixed seeds.

---

# 15. Cross Validation

Default:

```text
StratifiedKFold
n_splits = 5
shuffle = True
random_state = 42
```

For each fold record:

- ROC-AUC
- PR-AUC
- F1
- precision
- recall
- log loss

Report:

```text
mean ± standard deviation
```

---

# 16. Model Evaluation

Accuracy must **not** be the primary metric.

Required metrics:

## ROC-AUC

Measures ranking/discrimination across thresholds.

## PR-AUC

Especially useful under class imbalance.

## Precision

Of predicted defaults, how many were actually defaults?

## Recall

Of actual defaults, how many did the model identify?

## F1

Harmonic mean of precision and recall.

## Log Loss

Measures quality of predicted probabilities.

## Brier Score

Measures probabilistic accuracy/calibration.

## Calibration Curve

Compare predicted probability against observed default frequency.

---

# 17. Decision Threshold

The default classification threshold of `0.5` must not automatically be considered optimal.

The system should support:

```text
threshold = 0.50
```

as a baseline and evaluate alternatives.

Example:

```text
Threshold  Precision  Recall
0.30       0.41       0.82
0.40       0.49       0.75
0.50       0.57       0.66
0.60       0.65       0.54
0.70       0.72       0.43
```

The final threshold should be selected based on the project's stated business objective.

---

# 18. Risk Score

Convert predicted probability into a portfolio-friendly score.

Initial MVP:

```text
risk_score = round(default_probability * 100)
```

Risk levels:

```text
0–30   LOW
31–70  MEDIUM
71–100 HIGH
```

This is a presentation layer and should not be confused with an industry-standard credit score.

---

# 19. Explainability — SHAP

Use SHAP for model interpretation.

Required outputs:

### Global explanation

- mean absolute SHAP importance
- feature ranking
- summary plot

### Local explanation

For an individual prediction:

```text
default_probability = 0.73

Top contributors:

debt_to_income       +0.21
late_payments        +0.14
credit_utilization   +0.08
income               -0.06
employment_years     -0.04
```

The API should return machine-readable explanation data.

---

# 20. Prediction API

Base URL:

```text
/api/v1
```

## Health

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

## Model information

```http
GET /models/active
```

Response:

```json
{
  "name": "credit-risk-xgboost",
  "version": "1.0.0",
  "algorithm": "XGBoost",
  "roc_auc": 0.82
}
```

## Single prediction

```http
POST /predictions
```

Request:

```json
{
  "age": 34,
  "income": 1450000,
  "employment_years": 6,
  "home_ownership": "RENT",
  "loan_amount": 500000,
  "interest_rate": 12.5,
  "term_months": 36,
  "loan_intent": "PERSONAL",
  "credit_history_years": 7,
  "late_payments": 1,
  "previous_defaults": 0,
  "credit_utilization": 0.42,
  "active_credit_lines": 4
}
```

Response:

```json
{
  "default_probability": 0.183,
  "risk_score": 18,
  "risk_level": "LOW",
  "model": {
    "name": "credit-risk-xgboost",
    "version": "1.0.0"
  },
  "explanation": [
    {
      "feature": "debt_to_income",
      "impact": 0.18,
      "direction": "positive"
    }
  ]
}
```

## Batch prediction

```http
POST /predictions/batch
```

Input:

- JSON array initially
- CSV upload in later version

Output:

- predictions
- probabilities
- risk levels
- model version

---

# 21. API Validation

Pydantic must validate:

- non-negative income
- positive loan amount
- valid age range
- valid categorical values
- reasonable employment duration
- probability range
- valid UUIDs
- required fields

Invalid requests must return HTTP 422.

---

# 22. Database Layer

Use:

- SQLAlchemy 2.x
- PostgreSQL
- Alembic

Requirements:

- typed ORM models
- repository/service separation
- migrations
- transactions
- indexes on foreign keys
- database constraints where appropriate

Do not put raw SQL throughout API route handlers.

---

# 23. ML Service Architecture

Prediction flow:

```text
FastAPI endpoint
      ↓
Pydantic validation
      ↓
Prediction service
      ↓
Feature builder
      ↓
Preprocessing pipeline
      ↓
XGBoost model
      ↓
Probability
      ↓
Risk score
      ↓
SHAP explanation
      ↓
Persist prediction
      ↓
API response
```

The FastAPI layer must not contain model implementation details.

---

# 24. Model Artifact

The model artifact must contain the complete inference pipeline where practical:

```text
preprocessing
+
feature engineering
+
model
```

Possible artifact:

```text
models/
└── credit_risk_xgboost_v1.joblib
```

Metadata:

```json
{
  "model_name": "credit-risk-xgboost",
  "version": "1.0.0",
  "algorithm": "XGBoost",
  "training_date": "YYYY-MM-DD",
  "dataset_version": "v1",
  "roc_auc": 0.82,
  "pr_auc": 0.65
}
```

---

# 25. Model Versioning

Every deployed model must have:

```text
model name
version
training dataset version
feature version
metrics
training timestamp
artifact location
```

Example:

```text
credit-risk-xgboost:v1.0.0
credit-risk-xgboost:v1.1.0
```

Predictions must store the model version used.

This makes historical predictions reproducible.

---

# 26. Experiment Tracking

MVP:

- store experiments in structured files/JSON
- commit configuration
- save evaluation reports

Later:

- integrate MLflow

Each experiment should record:

```text
experiment_id
model
hyperparameters
features
dataset version
metrics
random seed
artifact
timestamp
```

---

# 27. Testing Strategy

## Unit tests

Test:

- feature engineering
- validation
- risk scoring
- preprocessing
- prediction service
- repository methods

Example:

```text
test_risk_score_low()
test_risk_score_medium()
test_risk_score_high()
test_invalid_income()
test_feature_engineering()
```

## Integration tests

Test:

```text
FastAPI
    ↓
Service
    ↓
Database
```

Use a dedicated test database/container.

## ML tests

At minimum:

- pipeline can train
- model produces probability in `[0, 1]`
- prediction schema is stable
- feature columns match training schema
- no unexpected NaNs reach inference

---

# 28. Data Validation

Before training, validate:

- required columns
- dtypes
- null percentages
- duplicate rows
- invalid categorical values
- impossible numerical values
- target distribution
- feature ranges

A validation failure must stop the pipeline rather than silently corrupting data.

---

# 29. Data Quality Report

Generate a report containing:

```text
Rows
Columns
Missing values
Duplicates
Unique values
Numerical distributions
Categorical distributions
Target distribution
Potential outliers
Potential leakage
```

This report should be versioned as part of the project documentation, not necessarily committed for every dataset execution.

---

# 30. Reproducibility

All training runs must define:

```text
random_state = 42
```

The repository must specify:

- Python version
- dependency versions
- dataset version
- model version
- feature version

A fresh clone should be able to reproduce the baseline training run.

---

# 31. Docker

Services:

```text
app
db
```

Optional later:

```text
frontend
mlflow
redis
worker
```

Example:

```text
docker compose up --build
```

The application should start with:

```text
FastAPI → PostgreSQL
```

and run database migrations automatically or through an explicit migration command.

---

# 32. Environment Variables

`.env.example`:

```env
APP_ENV=development

DATABASE_URL=postgresql+psycopg://creditrisk:creditrisk@db:5432/creditrisk

MODEL_PATH=models/credit_risk_xgboost_v1.joblib

LOG_LEVEL=INFO
```

Secrets must never be committed.

---

# 33. Logging

Use structured application logging.

Log:

- startup
- database connection
- model loading
- prediction request ID
- prediction latency
- model version
- errors

Never log sensitive customer information.

---

# 34. Observability

MVP:

- structured logs
- request IDs
- prediction latency

Later:

- Prometheus metrics
- Grafana dashboard
- model drift monitoring

Potential metrics:

```text
prediction_count
prediction_latency
model_error_count
risk_distribution
```

---

# 35. Model Monitoring — Future

Monitor:

## Data drift

Compare production feature distributions against training distributions.

## Prediction drift

Monitor changes in:

```text
LOW / MEDIUM / HIGH
```

distribution.

## Performance drift

If actual outcomes become available, calculate:

- ROC-AUC
- PR-AUC
- calibration
- recall
- precision

over time.

---

# 36. Security / Privacy

The project must use synthetic or public anonymized data.

Never include:

- names
- addresses
- phone numbers
- email addresses
- government IDs
- bank account numbers
- real financial records

API input must be treated as untrusted.

Production deployment should eventually include:

- authentication
- rate limiting
- HTTPS
- secret management
- audit logging

---

# 37. Ethical / Responsible ML

Credit risk is a **high-impact domain**.

The project is educational/portfolio-oriented and must not claim to be suitable for real-world lending decisions.

The README and model card should explicitly state:

- dataset limitations
- potential bias
- absence of real-world validation
- lack of regulatory/compliance review
- model uncertainty
- intended educational/research use

Potentially sensitive/proxy features must be reviewed carefully.

Do not claim that a prediction represents a person's actual creditworthiness.

---

# 38. Model Card

Create:

```text
docs/model_card.md
```

Include:

```text
Model name
Version
Purpose
Intended use
Out-of-scope use
Training data
Features
Target
Algorithm
Metrics
Limitations
Bias considerations
Explainability
Reproducibility
```

---

# 39. CI/CD

GitHub Actions pipeline:

```text
Push / Pull Request
        ↓
Install dependencies
        ↓
Ruff
        ↓
MyPy
        ↓
Pytest
        ↓
Build Docker image
```

Later:

```text
        ↓
Integration tests
        ↓
Security scan
        ↓
Deploy
```

CI must fail if tests or quality checks fail.

---

# 40. Git Strategy

Commits should represent real units of work.

Examples:

```text
chore: initialize project structure
feat: add database configuration
feat: add customer and loan models
feat: add alembic migrations
feat: implement dataset ingestion
feat: add data validation
feat: add feature engineering pipeline
feat: train logistic regression baseline
feat: train xgboost model
feat: add cross validation
feat: add model calibration
feat: add shap explanations
feat: implement prediction service
feat: add prediction endpoint
test: add prediction service tests
feat: add batch predictions
feat: add risk analytics endpoint
ci: add github actions
docs: add model card
docs: document architecture
```

Avoid meaningless commits such as:

```text
update
fix
changes
stuff
final
final2
```

---

# 41. Development Milestones

## Phase 0 — Planning

- [ ] Define scope
- [ ] Select dataset
- [ ] Document data source
- [ ] Define target
- [ ] Define initial features

## Phase 1 — Repository

- [ ] Initialize Python project
- [ ] Configure Ruff
- [ ] Configure Pytest
- [ ] Configure environment
- [ ] Create package structure
- [ ] Create README

## Phase 2 — Data

- [ ] Download dataset
- [ ] Implement ingestion
- [ ] Validate schema
- [ ] Generate data quality report
- [ ] Perform EDA
- [ ] Document data dictionary

## Phase 3 — ML baseline

- [ ] Create train/validation/test split
- [ ] Build preprocessing pipeline
- [ ] Train Logistic Regression
- [ ] Evaluate baseline
- [ ] Add Random Forest

## Phase 4 — XGBoost

- [ ] Train first XGBoost model
- [ ] Evaluate
- [ ] Add cross-validation
- [ ] Tune hyperparameters
- [ ] Compare against baselines
- [ ] Select candidate model

## Phase 5 — Explainability

- [ ] Integrate SHAP
- [ ] Global feature importance
- [ ] Local explanations
- [ ] Explanation API schema

## Phase 6 — Backend

- [ ] PostgreSQL
- [ ] SQLAlchemy
- [ ] Alembic
- [ ] Customer model
- [ ] Loan model
- [ ] Prediction model
- [ ] Model metadata
- [ ] Prediction service
- [ ] FastAPI endpoints

## Phase 7 — Productionization

- [ ] Docker
- [ ] Docker Compose
- [ ] Logging
- [ ] Error handling
- [ ] Health checks
- [ ] Integration tests

## Phase 8 — Dashboard

- [ ] Risk distribution
- [ ] Prediction form
- [ ] Individual prediction
- [ ] SHAP visualization
- [ ] Model metrics
- [ ] Batch predictions

## Phase 9 — CI/CD

- [ ] GitHub Actions
- [ ] Lint
- [ ] Type checking
- [ ] Unit tests
- [ ] Integration tests
- [ ] Docker build

## Phase 10 — Advanced

- [ ] Optuna
- [ ] MLflow
- [ ] Model registry
- [ ] Data drift
- [ ] Prediction drift
- [ ] Automated retraining pipeline

---

# 42. MVP Definition

The first usable version is complete when all of the following work:

```text
Dataset
   ↓
Validation
   ↓
Feature Engineering
   ↓
XGBoost
   ↓
Evaluation
   ↓
Saved Model
   ↓
FastAPI
   ↓
POST /predictions
   ↓
Probability + Risk Score + Explanation
```

MVP must have:

- reproducible training
- XGBoost model
- documented evaluation
- SHAP explanation
- FastAPI prediction endpoint
- PostgreSQL
- tests
- Docker
- README

The frontend is not required for MVP.

---

# 43. Definition of Done

A feature is considered complete when:

- implementation exists
- unit tests exist where applicable
- type/lint checks pass
- documentation is updated
- no secrets are committed
- code follows project architecture
- behavior is reproducible
- Git commit clearly describes the change

---

# 44. Suggested First Dataset Schema

If the selected public dataset provides compatible fields, normalize them toward:

```text
age
income
employment_years
home_ownership
loan_amount
interest_rate
term_months
loan_intent
loan_grade
loan_percent_income
credit_history_years
previous_default
late_payments
credit_utilization
active_credit_lines
loan_status
```

Do not force this schema if the chosen dataset does not contain the required information. The final feature set must be based on actual available data.

---

# 45. Success Criteria

The project is successful when it demonstrates all of the following:

### Data Science

- meaningful EDA
- robust preprocessing
- feature engineering
- class imbalance handling
- appropriate evaluation metrics

### Machine Learning

- baseline comparison
- XGBoost
- hyperparameter tuning
- probability calibration
- SHAP explainability

### Backend

- FastAPI
- Pydantic
- PostgreSQL
- SQLAlchemy
- Alembic
- service/repository architecture

### Engineering

- tests
- Docker
- CI
- logging
- reproducibility
- versioned model artifacts

### Portfolio

- clean README
- architecture diagram
- model card
- meaningful Git history
- screenshots/demo
- documented experiments

---

# 46. Long-Term Architecture

```text
                    ┌─────────────────┐
                    │ Public / Batch  │
                    │ Data Sources    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Data Pipeline   │
                    │ Validation      │
                    │ Feature Store   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Training        │
                    │ XGBoost         │
                    │ Optuna          │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Model Registry  │
                    │ MLflow          │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ FastAPI         │
                    │ Prediction API  │
                    └───────┬─────────┘
                            ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
       ┌──────────────┐            ┌──────────────┐
       │ PostgreSQL   │            │ Dashboard    │
       └──────────────┘            └──────────────┘
              │
              ↓
       ┌──────────────┐
       │ Monitoring   │
       │ Drift        │
       └──────────────┘
```

The architecture should evolve incrementally. Do not implement the entire long-term architecture before the MVP.

---

# 47. Guiding Principle

> **Build a real ML system, not a notebook with an API attached.**

The project should prioritize:

1. reproducibility
2. data quality
3. correct evaluation
4. explainability
5. clean architecture
6. testing
7. incremental delivery

The model is only one component of the system.
