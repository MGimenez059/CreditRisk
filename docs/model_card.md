# Model Card — CreditRisk

> **Status: template.** No model has been trained yet (roadmap Phase 4). This
> document follows the exact section list SPECS.md §38 requires, scaffolded
> now so `scripts/train_model.py` and `scripts/evaluate_model.py` have a
> fixed place to write real values into. Every `TBD` below must be filled in
> before this model is described as "trained" anywhere else in the
> repository.

## Model name

`credit-risk-xgboost` (see `credit_risk.config.settings.Settings.model_name`).

## Version

TBD — the first trained artifact will be `1.0.0`, per SPECS.md §25's versioning example (`credit-risk-xgboost:v1.0.0`).

## Purpose

Estimate the probability that a loan applicant will default, and expose that estimate with a human-readable explanation, for portfolio and educational purposes. See [Ethical Considerations](../README.md#ethical-considerations) in the README for the full non-goals statement.

## Intended use

Demonstrating an end-to-end ML engineering workflow — data validation, baseline comparison, tuned gradient boosting, calibration, explainability, and a served API — as a portfolio artifact.

## Out-of-scope use

Any real lending, credit, or underwriting decision. The model has not been reviewed for regulatory compliance (e.g., fair lending laws), has not been validated on real-world outcomes, and is trained on a single public, synthetic-scale dataset. See SPECS.md §1 (Non-goals).

## Training data

See `docs/data_dictionary.md` for full provenance, the raw column reference, and the known gap between the public API's field set and the Phase 0 dataset's actual columns.

| | |
|---|---|
| Dataset | Kaggle `laotse/credit-risk-dataset` |
| Dataset version pinned at training time | TBD — populated at training time from `ModelArtifactMetadata.dataset_version`, persisted to `ModelMetadata.training_dataset` |
| Split | 70% train / 15% validation / 15% test, stratified by `loan_status`, `random_state=42` — per SPECS.md §10 |
| Class balance handling | TBD — `scale_pos_weight` / class weights / threshold tuning, per SPECS.md §11 |

## Features

TBD — will list only the columns the fitted pipeline actually consumes, which may be a subset of `docs/data_dictionary.md`'s application schema per the known gap fields (`term_months`, `late_payments`, `credit_utilization`, `active_credit_lines`). Derived features under consideration, per SPECS.md §8: `loan_to_income`, `debt_to_income`, `income_per_employment_year`, `credit_age_ratio`, `late_payment_rate`.

## Target

`loan_status` — binary, `0` = no default, `1` = default. Never used as a model input feature (SPECS.md §7).

## Algorithm

XGBoost (`XGBClassifier`), compared against Logistic Regression and Random Forest baselines (SPECS.md §12). Starting hyperparameters, per SPECS.md §13 (not final — selected via validation/cross-validation):

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

Hyperparameter tuning: Optuna, over `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda` — primary objective ROC-AUC, secondary PR-AUC / Brier Score / F1 / calibration (SPECS.md §14). Cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` (SPECS.md §15). Decision threshold: TBD, selected against a stated business objective, not a default 0.5 (SPECS.md §17). Random seed: `credit_risk.ml.train.DEFAULT_RANDOM_STATE` (currently `42`).

## Metrics

All values below are placeholders until roadmap Phase 4 runs `scripts/evaluate_model.py` against a held-out test set. Accuracy is deliberately not tracked as a primary metric (SPECS.md §16).

| Metric | Value |
|---|---|
| ROC-AUC | TBD |
| PR-AUC | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 | TBD |
| Log Loss | TBD |
| Brier Score | TBD |

## Explainability

Per-prediction explanations are produced via SHAP (`credit_risk.ml.explain`, roadmap Phase 5): global mean-absolute SHAP importance and feature ranking, plus per-prediction local contributions, per SPECS.md §19. Populated once training exists to compute it against.

## Reproducibility

`random_state=42` fixed throughout (splitting, cross-validation, XGBoost, Optuna). A fresh clone can reproduce the baseline training run via `scripts/ingest_data.py` → `scripts/train_model.py`, per SPECS.md §30. Every trained artifact records Python version, dependency versions (`pyproject.toml`), dataset version, model version, and feature version — see `credit_risk.ml.registry.ModelArtifactMetadata`.

## Limitations

- Synthetic-scale public dataset; no guarantee it reflects any real population's actual default behavior.
- Four fields in the public API contract (`term_months`, `late_payments`, `credit_utilization`, `active_credit_lines`) have no source column in the Phase 0 dataset — see `docs/data_dictionary.md`. Whatever the trained pipeline actually does with them (drop vs. impute) must be recorded here once Phase 3/4 lands.
- No fairness/bias audit has been performed. `person_home_ownership` and `loan_intent` are retained as features; whether either functions as a proxy for a protected characteristic has not been evaluated and should be, before any claim beyond "portfolio demo" is made (SPECS.md §37).
- No temporal validation: the dataset is a single static snapshot, so the model's behavior under population or economic drift is unknown.

## Bias considerations

TBD — to be completed alongside the fairness audit noted above, before roadmap Phase 10 sign-off. Potentially sensitive/proxy features must be reviewed carefully, per SPECS.md §37.
