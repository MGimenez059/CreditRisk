# Data Dictionary

## Dataset provenance

| | |
|---|---|
| **Source** | Kaggle — [`laotse/credit-risk-dataset`](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) |
| **Records** | 32,581 rows |
| **Columns** | 12 (11 features + 1 target) |
| **Target** | `loan_status` (0 = no default, 1 = default) |
| **License** | Not machine-readable on the Kaggle listing at the time of writing — confirm the current license on the dataset page before redistributing the raw file. The raw CSV is **not** committed to this repository (see `.gitignore`); `scripts/ingest_data.py` downloads or expects a manually placed local copy. |
| **Dataset version** | Pinned per training run in `ModelArtifactMetadata.dataset_version` (see `docs/architecture.md`), not hardcoded here, so this document does not go stale when the source dataset is updated upstream. |

This is a synthetic/anonymized public dataset used for portfolio and educational purposes. It does not represent real applicants — see [Ethical Considerations](../README.md#ethical-considerations) in the README.

## Raw source columns

| Column | Type | Description | Notes |
|---|---|---|---|
| `person_age` | int | Applicant age in years | |
| `person_income` | float | Annual income | Currency unit as provided by the source, not specified upstream |
| `person_home_ownership` | categorical | `RENT`, `OWN`, `MORTGAGE`, `OTHER` | |
| `person_emp_length` | float | Years in current employment | Contains nulls in the raw file |
| `loan_intent` | categorical | `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` | |
| `loan_grade` | categorical | `A` through `G`, lender-assigned | |
| `loan_amnt` | float | Requested loan amount | |
| `loan_int_rate` | float | Annual interest rate (%) | Contains nulls in the raw file |
| `loan_status` | int | **Target.** 0 = no default, 1 = default | |
| `loan_percent_income` | float | `loan_amnt` as a fraction of `person_income` | Derivable; kept as-is from source rather than recomputed, pending a Phase 2 leakage review |
| `cb_person_default_on_file` | categorical | `Y` / `N` — prior default on credit bureau file | |
| `cb_person_cred_hist_length` | int | Credit history length, in years | |

## Mapping across the three naming layers

This project uses three distinct naming layers by design, matching SPECS.md exactly rather than collapsing them into one:

1. **Source dataset columns** (Kaggle) — `person_age`, `loan_amnt`, etc.
2. **API field names** (`schemas/prediction.py`, SPECS.md §20) — `age`, `loan_amount`, `loan_intent`. This is the public JSON contract in `README.md`.
3. **Canonical DB column names** (`db/models/`, SPECS.md §6) — `age`, `amount`, `purpose`. Mostly identical to the API names, but `Loan.amount` and `Loan.purpose` are deliberately renamed from the API's `loan_amount` / `loan_intent`; translating between the two is a repository/service-layer concern, documented directly in `db/models/loan.py`.

| Source column | API field (`PredictionRequest`) | Canonical DB column |
|---|---|---|
| `person_age` | `age` | `Customer.age` |
| `person_income` | `income` | `Customer.income` |
| `person_home_ownership` | `home_ownership` | `Customer.home_ownership` |
| `person_emp_length` | `employment_years` | `Customer.employment_years` |
| `loan_intent` | `loan_intent` | `Loan.purpose` |
| `loan_grade` | — (not yet in `PredictionRequest`) | `Loan.grade` |
| `loan_amnt` | `loan_amount` | `Loan.amount` |
| `loan_int_rate` | `interest_rate` | `Loan.interest_rate` |
| `loan_percent_income` | — | `Loan.loan_percent_income` (kept for source parity, not in SPECS.md §6's canonical model) |
| `cb_person_cred_hist_length` | `credit_history_years` | `CreditHistory.credit_history_years` |
| `cb_person_default_on_file` (`Y`/`N`) | `previous_defaults` (int) | `CreditHistory.previous_defaults` (int; source boolean mapped to `0`/`1`) |
| `loan_status` | — (never a request field, per SPECS.md §7) | `Loan.loan_status` — historical training label only |

## Known gap: fields with no source column

The public API contract in `README.md` / `SPECS.md` §20 and `PredictionRequest` (`src/credit_risk/schemas/prediction.py`) includes four fields that **do not exist** in the Phase 0 dataset:

- `term_months`
- `late_payments`
- `credit_utilization`
- `active_credit_lines`

These are modeled as **nullable** columns on `CreditHistory` and `Loan` (see `src/credit_risk/db/models/`) rather than removed, because they are realistic and commonly available fields in a production credit bureau feed, and the API is designed against that eventual reality. Until a richer dataset is ingested or these are engineered as proxies, the Phase 3/4 preprocessing pipeline must either:

1. drop them from the trained feature set entirely (recommended for the first XGBoost baseline), or
2. impute them and flag the imputation, clearly documented in `docs/model_card.md`'s Limitations section once a model is trained.

This gap is intentional and tracked here rather than silently papered over with fabricated values.
