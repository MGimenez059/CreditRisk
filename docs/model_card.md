# Model Card — CreditRisk

**Status: Phase 4 complete, 2026-09-09. XGBoost selected, frozen and evaluated
once on test. Phase 5 SHAP is complete; operational serving remains pending.**

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
that weighting improves probability estimates. Phase 4 compared weighted and unweighted alternatives.
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

## Initial XGBoost comparison

Run: `models/candidates-v1`, 2026-09-09, on the same snapshot and exact split
positions as Phase 3. Both baseline results match the earlier run within 1e-12.
Full parameters, runtime and metrics: [candidate results](candidate_results.json).

XGBoost uses CPU histogram trees, seed 42, one thread, 200 trees, depth 3 and
learning rate 0.05. It starts without class weighting (`scale_pos_weight=1`);
Logistic Regression and Random Forest retain their balanced class weights.
This is an initial comparison of configurations, not an isolated comparison of
algorithms. Parameters were fixed before validation; no early stopping or tuning.
See the [XGBoost parameter reference](https://xgboost.readthedocs.io/en/stable/parameter.html).

| Validation metric | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| roc_auc | 0.860188 | 0.932167 | 0.927015 |
| pr_auc | 0.688847 | 0.874833 | 0.871481 |
| f1 | 0.606686 | 0.785679 | 0.790123 |
| log_loss | 0.473660 | 0.259069 | 0.236647 |
| brier_score | 0.155617 | 0.073457 | 0.066314 |
| precision | 0.489106 | 0.824923 | 0.925786 |
| recall | 0.798689 | 0.750000 | 0.689139 |

Random Forest retains higher ROC-AUC and average precision. XGBoost has lower
log loss and Brier score, and higher precision but lower recall at threshold 0.5.
These validation results do not establish calibration or a final winner.
At this initial stage, CV, weighting comparisons, tuning, calibration and
threshold selection were pending; test had not yet been scored. The completed
selection below supersedes these preliminary comparisons.

## Final selection and test evaluation

The [selection protocol](selection_protocol.md) was fixed before CV/tuning and test
scoring. Five grouped folds within original training compared weighted/unweighted
Logistic Regression, Random Forest and XGBoost. Twelve sequential Optuna trials
selected XGBoost trial 6 (CV ROC-AUC 0.933849). Other model families were outside
the fixed 0.002 ROC-AUC tolerance; probability metrics supported the same choice.

The base estimator uses 200 depth-5 trees, learning rate 0.1407256738281049,
min_child_weight=1, scale_pos_weight=1, CPU hist, one thread and seed 42.
It fitted 22,811 training rows. Validation groups were split into 2,449 calibration
and 2,440 decision rows. Sigmoid calibration was rejected: decision log loss rose
from 0.201327 to 0.208473 and Brier score from 0.056903 to 0.058168.

Threshold **0.43** maximized decision-set F1 on the fixed grid. The model was not
refitted after choosing calibration/threshold. Display risk bands remain separate.
This threshold is a portfolio objective, not a lending policy.

The original **4,874 test rows were scored once**, after the artifact, threshold,
features and calibration were frozen and their hashes verified:

| Test metric | Value |
|---|---:|
| roc_auc | 0.940042 |
| pr_auc | 0.884125 |
| f1 | 0.798310 |
| log_loss | 0.212963 |
| brier_score | 0.061381 |
| precision | 0.909747 |
| recall | 0.711195 |

Artifact: `models/selected-v1/model.joblib` (version `selected-v1`), with metadata
in `model.json`. The artifact contains the full raw-input pipeline. Its metadata
metrics are explicitly from decision rows; final test metrics are in `test.json`.
See the [complete evaluation report](evaluation_report.md),
[reliability curves](calibration_curves.svg) and [run evidence](selection_results.json).
CV and decision scores were used for selection and may be optimistic. Final test
results are now known and cannot be treated as unseen in subsequent development.

## Explanations

Tree SHAP explains the frozen uncalibrated XGBoost margin in log-odds.
One-hot impacts are summed by source field; derived numeric variables remain
separate. Base plus all impacts reconstructs the margin, whose sigmoid matches
pipeline probability. These are model attributions, not causal effects.
See [the explanation report](explainability.md) for the fixed 500-row development
sample, global summary, synthetic example and reconstruction checks.

## Limitations and bias considerations

- No temporal validation or evidence of real-world performance.
- Dataset representativeness, source timing and observation horizon are uncertain.
- Prior-default data is a binary indicator, not a frequency/count.
- Grouping prevents identical inputs crossing partitions but cannot identify repeated
  people without entity identifiers, and does not resolve every leakage risk.
- No fairness audit has been performed. Review age and potential proxy features;
  differences in category default rates alone cannot establish absence of bias.
- No real lending, regulatory compliance or individual creditworthiness claims.
