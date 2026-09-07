# Model Card — CreditRisk

**Status: Phase 3 complete; real-data grouped baselines reproduced twice on
2026-09-07. No production model has been selected.**

## Purpose and intended use

Demonstrate reproducible default classification, evaluation, explainability and
backend serving for a portfolio. This is not a lending/underwriting system.
See [ethical considerations](../README.md#ethical-considerations).

## Data and target

Source: Kaggle `laotse/credit-risk-dataset`; see the [data dictionary](data_dictionary.md).
The regenerated report retained 32,574 of 32,581 rows after seven exclusions,
with 165 exact duplicate rows. Kaggle metadata confirms source version 1, CC0 license
and target 0 = no default, 1 = default. Population, synthetic origin and observation
horizon remain unverified. See [recorded provenance](dataset_provenance.json).

## Features and prediction time

Nine raw fields: age, income, employment duration, loan amount, interest rate,
credit-history length, home ownership, loan intent and the prior-default indicator
(source names are defined by `ml.preprocessing.RAW_FEATURE_COLUMNS`). Three derived
ratios are computed in the saved pipeline. Numeric values receive median imputation
and scaling; categorical values receive most-frequent imputation and one-hot encoding.
Only training rows fit these transformations.

The demo assumes a priced offer or explicitly missing rate. `loan_grade` is excluded
as a conservative availability/derivation decision, not proven leakage. Historical
default rates differ strongly across grades; this does not demonstrate an almost
perfect predictor. Source timing for both grade and interest rate needs verification.
The four unsupported API fields and two unavailable derived features are not used.

## Validation protocol

Identical raw model inputs share a group, including conflicting labels. Unique
groups are split 70/15/15 with seed 42 and stratification by majority label (positive
on ties). Row sizes and class balance may differ from nominal ratios; the run
manifest reports actual development proportions. All rows are retained.
Phase 4 cross-validation must preserve groups as well. The baseline command does
not compute test predictions or metrics. Historical full-data EDA was already viewed,
so this is a held-out scoring partition, not a claim of completely unseen data.

Both baselines use `class_weight="balanced"`; this is a baseline choice, not proof
that weighting improves probability estimates. Phase 4 should compare alternatives.
F1/precision/recall use threshold 0.5. `pr_auc` means scikit-learn average precision,
not trapezoidal integration. Brier score measures probabilistic error; inspect
calibration curves separately before claiming calibrated probabilities.

## Verified baseline results — grouped protocol

Run date: 2026-09-07. Train: 22,811 rows; validation: 4,889; test: 4,874.
Train positive rate: 21.814%; validation: 21.845%. Test was allocated but not scored.
The two independent runs produced identical split assignments and validation metrics.

| Validation metric | Logistic Regression | Random Forest |
|---|---:|---:|
| ROC-AUC | 0.860188 | 0.932167 |
| Average precision | 0.688847 | 0.874833 |
| F1 at 0.5 | 0.606686 | 0.785679 |
| Log loss | 0.473660 | 0.259069 |
| Brier score | 0.155617 | 0.073457 |
| Precision at 0.5 | 0.489106 | 0.824923 |
| Recall at 0.5 | 0.798689 | 0.750000 |

Random Forest improves ranking and probability error on this validation split;
Logistic Regression has higher recall at threshold 0.5. Do not claim Random Forest
wins on every metric. The older row-split figures in Git history are not directly
comparable: the partition and environment changed.

## Reproducibility

```bash
uv sync --locked --python 3.12 --extra dev --extra viz
uv run --locked python scripts/ingest_data.py
uv run --locked python scripts/train_baselines.py
```

Use a fresh --output directory for another run. Full run manifests and pipelines
are stored locally under models/baselines-v1 and models/baselines-v2. The committed
[run summary](baseline_results.json) records exact parameters, dependency versions,
input hash, split sizes and validation metrics. [Dataset provenance](dataset_provenance.json)
records the source version and both raw/validated hashes. Artifacts accept raw
source inputs; feature engineering is serialized with preprocessing and estimator.
Same seed alone does not promise bit-identical results across platforms/runtimes.

## Model selection, calibration and explanations

XGBoost is a Phase 4 candidate, compared against these baselines under a common
protocol. Selection must consider discrimination, probability quality and complexity.
The selected model may be Random Forest if it is better justified.
Calibrate using development data separate from fitting the estimator; freeze all
choices before final test scoring. A demo threshold objective remains to be stated.
Risk-score bands are illustrative presentation categories, not validated decisions.

SHAP is pending Phase 5. Specify output space, base value and encoded-feature
aggregation. XGBoost raw SHAP values are log-odds, not probability increments; explain
whether values refer to the base estimator or calibrated predictor. See
[TreeExplainer documentation](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html).

## Limitations and bias considerations

- No temporal validation or evidence of real-world performance.
- Dataset representativeness, source timing and observation horizon are uncertain.
- Prior-default data is a binary indicator, not a frequency/count.
- Grouping prevents identical inputs crossing partitions but cannot identify repeated
  people without entity identifiers, and does not resolve every leakage risk.
- No fairness audit has been performed. Review age and potential proxy features;
  differences in category default rates alone cannot establish absence of bias.
- No real lending, regulatory compliance or individual creditworthiness claims.
