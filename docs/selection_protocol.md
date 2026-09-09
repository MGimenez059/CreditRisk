# Phase 4 selection protocol

Fixed before running Phase 4 CV/tuning or scoring test, 2026-09-09.

## Data boundaries

Retain the Phase 3 raw-input grouped 70/15/15 split and seed 42. CV and
hyperparameter tuning use only the original 70% training partition. Use five
stratified folds of unique raw-input groups, with group-majority labels (positive
on ties). Fit all transformations within each fold. Weight ratios use fold-training
labels only. Report mean and sample standard deviation of all seven metrics.

Split the original validation groups in half, seed 42, with the same group strata:
one half fits a sigmoid calibrator on the frozen selected estimator; the other
half chooses calibration and threshold. The estimator is fitted only on the
original training partition. No refit after calibration or threshold selection.
Earlier baseline/EDA analysis has seen development data; CV estimates used for
selection are not unbiased performance estimates. The final test is the reserved
scoring partition, not a claim that full-data EDA was never viewed.

## Candidates and tuning

Compare Logistic Regression, Random Forest and initial XGBoost, each weighted and
unweighted. Keep Phase 3 baseline parameters; XGBoost uses the initial candidate
configuration. Run 12 sequential Optuna TPE trials, seed 42, five startup trials,
maximizing mean training-CV ROC-AUC. No pruning or early stopping.
Search XGBoost n_estimators in {100, 200, 300}, max_depth 2..6,
learning_rate 0.03..0.15 (log), min_child_weight in {1, 5, 10}, and weighting
in {false, true}. All other parameters remain fixed. Record every trial.

Compare the tuned XGBoost winner with the six initial configurations. Candidates
within 0.002 of the highest mean ROC-AUC are eligible; choose lowest mean log loss,
then simpler family (Logistic Regression, Random Forest, XGBoost), then name.
Average precision, Brier score and threshold metrics remain secondary evidence.
The 0.002 tolerance is a fixed demo tradeoff, not a statistical significance test.

## Calibration and threshold

Compare no calibration with sigmoid calibration, fitted on calibration rows only.
Choose lower log loss on the separate decision rows; exact ties retain no
calibration. Report Brier score and ten-bin uniform reliability curves for both.
This comparison does not guarantee perfect calibration.

Demo classification objective: maximize F1 on decision rows over thresholds
0.05..0.95 inclusive in increments of 0.01. Ties prefer higher precision, then
higher threshold. Store the full grid and chosen threshold. This objective does
not encode real lending costs. Display risk-score bands remain unchanged.

## Feature availability decision

Retain the nine raw fields and three derived ratios. Continue excluding loan_grade:
its derivation and availability are unverified. Retain loan_int_rate only under
the already-declared priced-offer scenario; explicit missing rate is accepted.
The recorded source metadata supplies no timing guarantees. This review closes
the demo feature decision, not the upstream provenance uncertainty. A pre-pricing
use case requires a new protocol and independent evaluation data.

## Freeze and test

Save selection evidence, fitted pipeline (including calibration if selected),
metadata and a frozen manifest with input/artifact/sidecar hashes, exact row
positions and threshold. Final evaluation verifies those hashes and the original
split, then creates an exclusive evaluation marker before predicting. Repeated
scoring in that run is refused, including after interrupted attempts; inspect any
failure rather than clearing the marker to try another model. The marker is an
accidental-rerun guard, not a security or cross-checkout access control.

Only after verification, score the original test once with frozen decisions.
Publish all seven metrics, confusion matrix and reliability bins. No changes to
features/model/calibration/threshold in response to test results. Future research
on this snapshot must disclose that test results are already known.

References: [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html),
[Optuna TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).
