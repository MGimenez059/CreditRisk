# Model Card — CreditRisk

> **Status: baselines trained (roadmap Phase 3), production model pending
> (roadmap Phase 4).** Logistic Regression and Random Forest baselines
> below are trained and evaluated against the real Kaggle dataset. The
> XGBoost sections remain `TBD` until Phase 4. `scripts/train_model.py`
> and `scripts/evaluate_model.py` will populate those sections when run.

## Model name

`credit-risk-xgboost` (see `credit_risk.config.settings.Settings.model_name`) will be the production model, Phase 4. The baselines below are not registered in `ml.registry` — they exist to set the comparison bar, per SPECS.md §12.

## Version

TBD — the first trained XGBoost artifact will be `1.0.0`, per SPECS.md §25's versioning example (`credit-risk-xgboost:v1.0.0`).

## Purpose

Estimate the probability that a loan applicant will default, and expose that estimate with a human-readable explanation, for portfolio and educational purposes. See [Ethical Considerations](../README.md#ethical-considerations) in the README for the full non-goals statement.

## Intended use

Demonstrating an end-to-end ML engineering workflow — data validation, baseline comparison, tuned gradient boosting, calibration, explainability, and a served API — as a portfolio artifact.

## Out-of-scope use

Any real lending, credit, or underwriting decision. The model has not been reviewed for regulatory compliance (e.g., fair lending laws), has not been validated on real-world outcomes, and is trained on a single public, synthetic-scale dataset. See SPECS.md §1 (Non-goals).

## Training data

See `docs/data_dictionary.md` for full provenance and `docs/data_quality_report.md` for the real, current data quality report (row counts, missingness, excluded rows).

| | |
|---|---|
| Dataset | Kaggle `laotse/credit-risk-dataset` |
| Rows after cleaning | 32,574 of 32,581 (7 rows excluded — impossible `person_age`/`person_emp_length` values; see `docs/data_quality_report.md`'s "Excluded rows" section) |
| Split | 70% train (22,801) / 15% validation (4,886) / 15% test (4,887), stratified by `loan_status`, `random_state=42` — per SPECS.md §10. **The test set has not been touched** — every metric below is on the validation split only, per §10's "final test set must remain untouched until model selection is complete." |
| Class balance | 21.82% positive (default) / 78.18% negative — measured via `credit_risk.ml.train.measure_class_balance`, per SPECS.md §11. Ratio ≈ 3.6 negatives per positive: moderate imbalance, not extreme. |
| Class balance handling | `class_weight="balanced"` for both baselines below (SPECS.md §11's "class weights" strategy). Threshold optimization and `scale_pos_weight` are XGBoost-specific choices deferred to Phase 4. |

## Features

Baselines use `credit_risk.ml.preprocessing.ALL_FEATURE_COLUMNS`: `person_age`, `person_income`, `person_emp_length`, `loan_amnt`, `loan_int_rate`, `cb_person_cred_hist_length`, `loan_to_income`, `income_per_employment_year`, `credit_age_ratio` (numeric), plus `person_home_ownership`, `loan_intent`, `cb_person_default_on_file` (categorical, one-hot encoded).

**`loan_grade` is deliberately excluded** — see Limitations below for the full reasoning; this is the single most consequential feature decision made so far and materially affects how these numbers compare to public analyses of this dataset that include it.

Two of SPECS.md §8's five suggested derived features are not implemented: `debt_to_income` (needs a `total_debt` column this dataset doesn't have) and `late_payment_rate` (needs `late_payments`, one of the fields documented as absent from this dataset in `docs/data_dictionary.md`).

## Target

`loan_status` — binary, `0` = no default, `1` = default. Never used as a model input feature (SPECS.md §7).

## Baselines (Phase 3)

Trained via `credit_risk.ml.train.train_baseline`, evaluated via `credit_risk.ml.evaluate.evaluate_model` on the **validation split**:

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| ROC-AUC | 0.8491 | **0.9211** |
| PR-AUC | 0.6691 | **0.8556** |
| F1 (threshold 0.5) | 0.6009 | **0.7712** |
| Log Loss | 0.4738 | **0.2743** |
| Brier Score | 0.1552 | **0.0772** |

Random Forest outperforms Logistic Regression on every metric — expected, given it can capture non-linear feature interactions the linear model can't, and does so without leaning on `loan_grade` as a near-shortcut. This sets the bar Phase 4's XGBoost must clear to justify the added complexity.

## Algorithm

XGBoost (`XGBClassifier`), compared against the Logistic Regression and Random Forest baselines above (SPECS.md §12). Starting hyperparameters, per SPECS.md §13 (not final — selected via validation/cross-validation):

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

The table below is for the **production XGBoost model** and remains `TBD` until roadmap Phase 4 runs `scripts/evaluate_model.py` against the (still-untouched) test set. See "Baselines (Phase 3)" above for the real, current baseline numbers. Accuracy is deliberately not tracked as a primary metric (SPECS.md §16).

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

Per-prediction explanations are produced via SHAP (`credit_risk.ml.explain`, roadmap Phase 5): global mean-absolute SHAP importance and feature ranking, plus per-prediction local contributions, per SPECS.md §19. Populated once the Phase 4 model exists to compute it against.

## Reproducibility

`random_state=42` fixed throughout (splitting, baselines, and — once implemented — XGBoost/Optuna). A fresh clone can reproduce the baseline numbers above via `scripts/ingest_data.py` (with the raw CSV manually placed, see `docs/data_dictionary.md`) followed by `credit_risk.ml.train.split_dataset` + `train_baseline`, per SPECS.md §30. Every trained artifact records Python version, dependency versions (`pyproject.toml`), dataset version, model version, and feature version — see `credit_risk.ml.registry.ModelArtifactMetadata`.

## Limitations

- **`loan_grade` is excluded as a likely leakage risk, despite SPECS.md §8 listing it as a candidate feature.** In the real dataset, default rate by grade ranges from 9.96% (A) to 98.4% (G) — an almost perfect split. The lender assigns `loan_grade` at origination using their own risk assessment, so training on it risks the model re-deriving the lender's original decision rather than learning independent signal — SPECS.md §9 rule 5 ("avoid features that would only be available after the credit decision") is treated as taking precedence over §8's illustrative feature list here. **This means the baseline numbers above are not directly comparable to public analyses of this dataset that include `loan_grade`, which typically report higher scores.** A with-grade ablation is a reasonable Phase 4 side experiment, kept clearly separate from the production model.
- Synthetic-scale public dataset; no guarantee it reflects any real population's actual default behavior.
- Four fields in the public API contract (`term_months`, `late_payments`, `credit_utilization`, `active_credit_lines`) have no source column in this dataset — see `docs/data_dictionary.md`. The trained pipeline does not consume them at all (they aren't in `ALL_FEATURE_COLUMNS`); this must be revisited if a richer data source is ever ingested.
- No fairness/bias audit has been performed. `person_home_ownership` and `loan_intent` are retained as features; whether either functions as a proxy for a protected characteristic has not been evaluated and should be, before any claim beyond "portfolio demo" is made (SPECS.md §37).
- No temporal validation: the dataset is a single static snapshot, so the model's behavior under population or economic drift is unknown.
- The data quality report shows 165 duplicate rows (post-cleaning) — not deduplicated for the baselines above. Worth deciding explicitly in Phase 4 rather than carrying forward silently.

## Bias considerations

TBD — to be completed alongside the fairness audit noted above, before roadmap Phase 10 sign-off. Potentially sensitive/proxy features must be reviewed carefully, per SPECS.md §37.
