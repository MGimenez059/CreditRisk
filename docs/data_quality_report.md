# Data Quality Report

Generated from `data/raw/credit_risk_dataset.csv`. See SPECS.md §29 for the field list this report follows.

- **Rows (after exclusions):** 32574
- **Columns:** 12
- **Duplicate rows:** 165

## Excluded rows (impossible values)

7 row(s) excluded before this report was computed:
- Row 0: person_emp_length=123.0 above maximum 70
- Row 81: person_age=144 above maximum 100
- Row 183: person_age=144 above maximum 100
- Row 210: person_emp_length=123.0 above maximum 70
- Row 575: person_age=123 above maximum 100
- Row 747: person_age=123 above maximum 100
- Row 32297: person_age=144 above maximum 100

## Missing values

- `person_emp_length`: 895 (2.75%)
- `loan_int_rate`: 3115 (9.56%)

## Target distribution (`loan_status`)

- `0`: 78.18%
- `1`: 21.82%

## Potential outliers (IQR method, per numeric column)

- `person_age`: 1489 rows
- `person_income`: 1480 rows
- `person_emp_length`: 851 rows
- `loan_amnt`: 1688 rows
- `loan_int_rate`: 6 rows
- `loan_percent_income`: 650 rows
- `cb_person_cred_hist_length`: 1141 rows

## Potential leakage (|correlation with target| ≥ 0.95)

None flagged.

## Unique values per column

- `person_age`: 56
- `person_income`: 4294
- `person_home_ownership`: 4
- `person_emp_length`: 35
- `loan_intent`: 6
- `loan_grade`: 7
- `loan_amnt`: 753
- `loan_int_rate`: 348
- `loan_status`: 2
- `loan_percent_income`: 77
- `cb_person_default_on_file`: 2
- `cb_person_cred_hist_length`: 29
