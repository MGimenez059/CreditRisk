# CreditRisk

**An end-to-end machine learning system for estimating the probability of loan default.**

CreditRisk is a portfolio-grade Data Science / ML Engineering / Backend project. It goes from raw tabular data to a served, explainable, containerized prediction API — not just a notebook with a model in it.

> ⚠️ **Educational / portfolio project.** This system uses public, anonymized data and synthetic examples. It does **not** connect to any real financial institution, does not process real PII, and must not be used to make real lending decisions. See [Ethical Considerations](#ethical-considerations).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Model](#model)
- [Explainability](#explainability)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Documentation](#documentation)
- [Ethical Considerations](#ethical-considerations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

CreditRisk predicts the probability that a borrower will default on a loan, and exposes that prediction — together with a risk score and a human-readable explanation — through a REST API.

**Highlights:**

- 🔄 Reproducible data ingestion, validation, and preprocessing pipeline
- 📊 Exploratory data analysis and automated data quality reports
- 🧠 Baseline models (Logistic Regression, Random Forest) compared against a tuned **XGBoost** production model
- 🎯 Evaluation beyond accuracy: ROC-AUC, PR-AUC, F1, Log Loss, Brier Score, calibration
- 🔍 Per-prediction explanations powered by **SHAP**
- ⚡ **FastAPI** service with single and batch prediction endpoints
- 🗄️ **PostgreSQL** persistence for customers, loans, predictions, and model metadata
- 🐳 Fully containerized with **Docker Compose**
- ✅ Automated linting, type-checking, and testing via **GitHub Actions**
- 📈 Optional dashboard for risk analytics and explanation visualization

---

## Architecture

```text
                    ┌────────────────────┐
                    │   Public Dataset    │
                    │   CSV / Parquet     │
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
                    │ Feature Engineering │
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

Inference request flow:

```text
FastAPI endpoint → Pydantic validation → Prediction service → Feature builder
→ Preprocessing pipeline → XGBoost model → Probability → Risk score
→ SHAP explanation → Persist prediction → API response
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Backend** | Python 3.12+, FastAPI, Pydantic v2, SQLAlchemy 2.x, Alembic, Uvicorn |
| **Data** | Polars, Pandas, NumPy, PyArrow, DuckDB |
| **Machine Learning** | XGBoost, scikit-learn, SHAP, Optuna, joblib |
| **Database** | PostgreSQL |
| **Visualization** | Plotly, (optional) React + TypeScript |
| **Quality** | Pytest, Ruff, MyPy, pre-commit |
| **Infrastructure** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Optional / future** | MLflow, Redis + Celery/RQ, MinIO/S3, Prometheus + Grafana |

---

## Repository Structure

```text
credit-risk/
├── src/credit_risk/
│   ├── api/                # FastAPI routes & dependencies
│   ├── config/              # Settings
│   ├── db/                  # SQLAlchemy models, session, base
│   ├── schemas/              # Pydantic schemas
│   ├── repositories/         # Data access layer
│   ├── services/              # prediction / risk / explanation services
│   ├── ml/                    # preprocessing, features, train, evaluate, predict, explain, registry
│   └── main.py
├── data/                   # raw / interim / processed (not fully committed)
├── models/                 # serialized model artifacts
├── notebooks/               # EDA, feature engineering, model analysis
├── tests/                   # unit / integration / fixtures
├── scripts/                 # ingest_data.py, train_model.py, evaluate_model.py
├── alembic/                 # DB migrations
├── docs/                     # model_card.md, data_dictionary.md, architecture.md
├── .github/workflows/ci.yml
├── docker/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Python 3.12+ (for local development outside Docker)

### Run with Docker (recommended)

```bash
git clone https://github.com/<your-user>/credit-risk.git
cd credit-risk
cp .env.example .env

docker compose up --build
```

This starts the FastAPI service and PostgreSQL, and applies database migrations. The API will be available at `http://localhost:8000`.

Interactive API docs: `http://localhost:8000/docs`

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"

# start PostgreSQL separately, then:
alembic upgrade head

uvicorn credit_risk.main:app --reload
```

### Train the model

```bash
python scripts/ingest_data.py
python scripts/train_model.py
python scripts/evaluate_model.py
```

Training artifacts are written to `models/` and evaluation reports to `docs/`.

---

## Configuration

Environment variables (`.env.example`):

```env
APP_ENV=development

DATABASE_URL=postgresql+psycopg://creditrisk:creditrisk@db:5432/creditrisk

MODEL_PATH=models/credit_risk_xgboost_v1.joblib

LOG_LEVEL=INFO
```

Secrets are never committed to the repository.

---

## API Reference

Base URL: `/api/v1`

### Health check

```http
GET /health
```

```json
{ "status": "ok" }
```

### Active model info

```http
GET /models/active
```

```json
{
  "name": "credit-risk-xgboost",
  "version": "1.0.0",
  "algorithm": "XGBoost",
  "roc_auc": 0.82
}
```

### Single prediction

```http
POST /predictions
```

**Request**

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

**Response**

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

### Batch prediction

```http
POST /predictions/batch
```

Accepts a JSON array of loan applications (CSV upload planned for a later version) and returns predictions, probabilities, risk levels, and the model version used for each record.

Invalid requests return `HTTP 422` with field-level validation errors (via Pydantic).

---

## Model

**Primary algorithm:** `XGBClassifier`

| Metric | Value |
|---|---|
| ROC-AUC | 0.82 |
| PR-AUC | 0.65 |
| F1 | 0.58 |
| Brier Score | 0.09 |

*(Illustrative values — actual numbers depend on the selected dataset and are documented per model version in `docs/model_card.md`.)*

- Compared against Logistic Regression and Random Forest baselines using the same validation protocol
- Hyperparameters tuned with **Optuna**, optimizing primarily for ROC-AUC (secondary: PR-AUC, Brier Score, F1, calibration)
- Evaluated with **StratifiedKFold** cross-validation (5 folds)
- Probabilities calibrated; decision threshold selected against the project's stated business objective (not a default `0.5`)
- Class imbalance handled via `scale_pos_weight` / class weights / threshold optimization
- Every model artifact is versioned (`credit-risk-xgboost:v1.0.0`) with training dataset version, feature version, and metrics; every stored prediction records the model version used

**Risk score**

```text
risk_score = round(default_probability * 100)

0–30   LOW
31–70  MEDIUM
71–100 HIGH
```

This is a presentation-layer score, not an industry-standard credit score.

---

## Explainability

Every prediction includes a **SHAP**-based explanation showing which features pushed the probability up or down:

```text
default_probability = 0.73

Top contributors:
debt_to_income       +0.21
late_payments        +0.14
credit_utilization   +0.08
income               -0.06
employment_years     -0.04
```

Global feature importance and summary plots are available in `notebooks/03_model_analysis.ipynb` and `docs/model_card.md`.

---

## Testing

```bash
pytest
ruff check .
mypy src
```

- **Unit tests** — feature engineering, validation, risk scoring, preprocessing, prediction service, repositories
- **Integration tests** — FastAPI → service → database, against a dedicated test database
- **ML tests** — the pipeline trains successfully, probabilities fall in `[0, 1]`, prediction schema is stable, feature columns match the training schema, no unexpected NaNs reach inference

---

## CI/CD

GitHub Actions runs on every push and pull request:

```text
Install dependencies → Ruff → MyPy → Pytest → Build Docker image
```

The pipeline fails the build if linting, type-checking, or tests fail.

---

## Documentation

| Document | Description |
|---|---|
| [`SPECS.md`](SPECS.md) | Full technical specification this project is built against |
| [`docs/model_card.md`](docs/model_card.md) | Purpose, intended use, training data, metrics, limitations, bias considerations |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Field-by-field description of the canonical data model |
| [`docs/architecture.md`](docs/architecture.md) | Detailed system architecture and design decisions |
| [`ROADMAP.md`](ROADMAP.md) | Phased development roadmap |

---

## Ethical Considerations

Credit risk is a high-impact domain. This project is **educational and portfolio-oriented**:

- Uses public, anonymized data only — no real names, addresses, government IDs, or financial records
- Does **not** represent an individual's actual creditworthiness
- Has not undergone real-world validation or regulatory/compliance review
- Should not be used to make real lending or credit decisions
- Potential dataset bias and proxy features are documented in the model card

---

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the full phased plan (data → baseline models → XGBoost → explainability → backend → productionization → dashboard → CI/CD → advanced monitoring).

**MVP scope:** reproducible training, XGBoost model, documented evaluation, SHAP explanations, FastAPI prediction endpoint, PostgreSQL, tests, Docker, and this README. The frontend dashboard is out of scope for the MVP.

---

## License

MIT — see [`LICENSE`](LICENSE) for details.
