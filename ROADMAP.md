# CreditRisk — Work Roadmap

This is the single source of phase status. A scaffold is not an operational feature.
Phase 3 is complete. Two real-data runs reproduced identical grouped splits and
validation metrics; test predictions/metrics were not computed.

## Phase 0 — Planning
- [x] Define portfolio scope, target convention and non-goals.
- [x] Select Kaggle `laotse/credit-risk-dataset`.
- [x] Document known provenance and unknowns in the data dictionary.
- [x] Verify source version 1, CC0 license and target 0/1 mapping via Kaggle metadata.
- [ ] Observation horizon and collection/timing remain undocumented upstream; retain as limitations.

## Phase 1 — Repository
- [x] Python package, Ruff, MyPy, Pytest and pre-commit configuration.
- [x] Central settings and environment example.
- [x] Lockfile and documented Python 3.12 setup.

## Phase 2 — Data
- [x] Manual CSV acquisition instructions and ingestion script.
- [x] Structural validation, bounded row exclusions and quality report generator.
- [x] Historical EDA execution and data dictionary.
- [x] Document conservative exclusion of loan_grade and unresolved feature timing.
- [x] Regenerate the expanded quality report; record source metadata and raw/validated hashes.

## Phase 3 — ML Baseline
- [x] Group identical raw model inputs before stratified 70/15/15 group splitting.
- [x] Preserve groups across partitions, even with conflicting outcomes.
- [x] Serialize feature engineering, preprocessing and estimator together.
- [x] Train Logistic Regression and Random Forest with the same protocol.
- [x] Validation metrics, class weights and baseline threshold 0.5.
- [x] Runnable baseline CLI with artifact metadata and a split/run manifest.
- [x] Tests for raw-input serialization, API mapping and duplicate isolation.
- [x] Run the CLI twice on real data: 22,811 train / 4,889 validation / 4,874 test rows. Publish identical repeated validation results.

## Phase 4 — Model Selection
- [ ] Train XGBoost and compare it with both baselines; retain the best justified candidate.
- [ ] Group-aware cross-validation on development data; no duplicate groups across folds.
- [ ] Bounded, seeded Optuna tuning after a stable XGBoost baseline exists.
- [ ] Review loan_grade and loan_int_rate availability at the declared prediction time.
- [ ] Evaluate probability calibration on data separate from estimator fitting.
- [ ] State a demo decision objective and compare thresholds without using test results.
- [ ] Freeze features, candidate, calibration and threshold, then evaluate the test set once.
- [ ] Produce the selected artifact, configuration and complete evaluation report.

## Phase 5 — Explainability
- [ ] Global and local SHAP explanations for the selected model.
- [ ] Define base value, output units and aggregation of encoded features.
- [ ] Explain explicitly whether SHAP describes raw model output or the calibrated predictor.
- [ ] Finalize explanation response schema and verify additivity in the declared output space.

## Phase 6 — Backend
- [x] FastAPI route/service/repository and ORM scaffolds exist.
- [x] Prediction request maps available dataset fields to the raw model schema.
- [ ] Implement Alembic revisions for the MVP prediction/model persistence flow.
- [ ] Complete artifact loading, SHAP orchestration and transaction behavior.
- [ ] Ensure model selection and stored model metadata agree.
- [ ] Exercise single/batch requests against a dedicated PostgreSQL test database.

## Phase 7 — Packaging
- [x] Dockerfile, Compose and migration startup command exist.
- [x] Structured logging and centralized error-handler scaffolds exist.
- [ ] Verify a clean-container prediction demo using a trained artifact and real migrations.
- [ ] Complete request correlation, failure logging and readiness checks.

## Phase 8 — Testing & Quality
- [x] Unit tests for data, feature engineering, preprocessing, baselines and risk scoring.
- [ ] Database integration tests and complete prediction-service failure coverage.
- [x] Reproduce baseline training twice from the locked environment and pinned input snapshot.

## Phase 9 — CI
- [x] Workflow runs Ruff, MyPy, Pytest and Docker build.
- [ ] Include the operational database integration path once Phase 6 is complete.

## Phase 10 — Documentation & Portfolio
- [x] Separate implemented behavior from planned features; remove duplicate phase lists.
- [x] Publish real-data baseline validation and updated model card.
- [ ] Publish the selected-model evaluation and operational serving demo.
- [ ] Review sensitive/proxy features and document uncertainty without compliance claims.

## Post-MVP (optional)

Customer/loan CRUD, a dashboard, MLflow, asynchronous batch jobs, remote artifact
storage and drift/monitoring infrastructure need a concrete use case before implementation.
Existing customer/loan scaffolds are not MVP acceptance criteria.

## MVP Definition

Dataset -> validated data -> reproducible training/evaluation -> saved pipeline
-> FastAPI prediction -> probability + risk score + SHAP -> PostgreSQL persistence.
Include tests, Docker, CI and clear documentation. No frontend or production deployment required.

## Definition of Done

Implementation works for the intended inputs; relevant tests and quality checks pass;
results are reproducible; documentation reflects actual status. No secrets or direct
personal data are committed. A checkbox does not substitute for execution evidence.
