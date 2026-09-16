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
| Population and collection process | Publisher describes columns as simulating credit bureau data; generation method and population are not established |
| Default definition/horizon | Source confirms 0 = non-default, 1 = default; observation horizon unspecified |

The code's MIT license does not establish dataset redistribution rights. Raw data is
not committed. [Source version/license evidence](dataset_provenance.json) and a local file hash serve different
purposes; a hash identifies the input bytes but does not prove upstream provenance.

## Provenance review and scope

Rechecked the [Kaggle data card](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
via indexed page content and its [metadata API](https://www.kaggle.com/api/v1/datasets/view/laotse/credit-risk-dataset)
on 2026-09-16. The publisher describes the columns as simulating credit bureau data.
The API confirms that subtitle, version 1 and the binary target mapping. The reviewed
description does not specify an observation horizon, collection dates, sampling or
generation method, or feature measurement timeline. The simulation description is a
publisher statement, not independent verification of how every record was generated.

The discussion page returned no readable thread content to the retrieval tool;
targeted searches did not establish those missing facts. This review does not claim
to have inspected every discussion. The [recorded provenance](dataset_provenance.json)
retains the original download evidence and the dated follow-up findings separately.

These unknowns are retained as dataset limitations rather than unfinished development
tasks. The educational model estimates the dataset's default label; it cannot be
described as a 6- or 12-month default forecast or as validated for Argentine borrowers.
The source update date is not a collection date. Real-world use would require verified
provenance, a defined prediction horizon and evaluation on the intended population.
Reopen this review if new primary-source evidence becomes available.

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

## Phase 4 feature review (2026-09-09)

Reviewed the recorded source metadata and the declared priced-offer scenario.
Keep `loan_grade` excluded because neither its derivation nor timing is verified.
Keep `loan_int_rate` under the explicit assumption that an offer rate is known,
or explicitly null, when scoring. No upstream timing guarantees were found in
the recorded provenance. This closes the demo feature choice, while preserving
that limitation; it does not validate a pre-pricing application model.
The nine raw inputs and three derived features remain unchanged.
