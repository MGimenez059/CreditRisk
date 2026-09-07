# Data Dictionary

## Dataset provenance

| Item | Value |
|---|---|
| Source | [Kaggle: laotse/credit-risk-dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) |
| Historical raw size | 32,581 rows, 12 columns |
| Target convention | loan_status: 0 = no default, 1 = default |
| License | Kaggle metadata reports CC0: Public Domain; verified 2026-09-07 |
| Source version/download date | Version 1, downloaded 2026-09-07; source last updated 2020-06-02 |
| Snapshot identifier | Baseline CLI records SHA-256 of the validated Parquet actually used |
| Population and collection process | Not independently verified; do not assert synthetic origin or real-world representativeness |
| Default definition/horizon | Source confirms 0 = non-default, 1 = default; observation horizon unspecified |

The code's MIT license does not establish dataset redistribution rights. Raw data is
not committed. [Source version/license evidence](dataset_provenance.json) and a local file hash serve different
purposes; a hash identifies the input bytes but does not prove upstream provenance.

## Getting the raw file

1. Download the CSV manually from the source page (sign in if requested).
2. Save it as `data/raw/credit_risk_dataset.csv`.
3. Run `uv run --locked python scripts/ingest_data.py`.
4. Run `uv run --locked python scripts/train_baselines.py`.

Ingestion validates structural schema errors first. Defined row-level value violations
are excluded with a reason; more than 5% exclusions stop the pipeline. Unexpected
structural failures also stop it. This explicit cleaning policy is different from
silently ignoring validation errors. Null employment duration and rate are retained
for training-only imputation. Zero income is rejected by the feature builder/API
because the loan-to-income ratio would be undefined.

The script writes validated Parquet and a generated quality report. The committed
report was regenerated on 2026-09-07; see [provenance](dataset_provenance.json) for
source metadata and raw/validated SHA-256 values. The public download API allowed
unauthenticated access during this run; the ingestion command itself remains local-file only.

## Source columns and API mapping

| Source | API field | Meaning |
|---|---|---|
| person_age | age | Age in years; accepted range 18–100 |
| person_income | income | Annual income, positive for modeling; currency unverified |
| person_home_ownership | home_ownership | RENT, OWN, MORTGAGE, OTHER |
| person_emp_length | employment_years | Employment duration, 0–70 years; nullable |
| loan_intent | loan_intent | PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION |
| loan_grade | Excluded | Source grade A–G; derivation/availability unverified |
| loan_amnt | loan_amount | Positive loan amount, same monetary unit as income |
| loan_int_rate | interest_rate | Annual percent, 0–100; nullable |
| loan_percent_income | Not accepted | Source ratio; replaced by recomputed loan_to_income |
| cb_person_default_on_file | previous_defaults | N/Y maps to required 0/1 indicator, not a count |
| cb_person_cred_hist_length | credit_history_years | Non-negative history length in years |
| loan_status | Never accepted | Historical target, never a feature |

The API adapter maps public names to the nine `RAW_FEATURE_COLUMNS`. PostgreSQL's
optional Loan scaffold uses `amount` and `purpose` for `loan_amount` and `loan_intent`;
this storage mapping is independent of model training. Customer/loan CRUD is deferred.

## Derived features

- `loan_to_income = loan_amnt / person_income`.
- `income_per_employment_year = person_income / max(person_emp_length, 1)`.
- `credit_age_ratio = cb_person_cred_hist_length / max(person_age, 1)`.

Missing employment values propagate to the derived ratio and are imputed using
training medians. Derivation occurs inside the serialized pipeline. The source
ratio may have rounding differences and is not treated as an identical stored copy.
`debt_to_income` and `late_payment_rate` are not available from this dataset.

## Unsupported fields

`term_months`, `late_payments`, `credit_utilization` and `active_credit_lines` were
removed from the unreleased contract and ORM scaffold. Do not impute features with
no observed source values or invent proxies merely to satisfy an illustrative schema.

## Prediction time and leakage

The current demo assumes a loan offer with a known rate, or an explicitly missing
rate represented as null. It does not claim to score an application before pricing.
The actual source timing for loan_int_rate and loan_grade remains unverified.
Grade is excluded conservatively; strong target association alone does not establish
leakage. Reassess both fields under a pre-pricing use case.

Identical raw model inputs form one split group, including conflicting target labels.
This preserves rows while preventing duplicate inputs from crossing evaluation boundaries.
