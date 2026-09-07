# Data Quality Report

Generated from `data/raw/credit_risk_dataset.csv`. See SPECS.md §29 for the field list.
Raw CSV SHA-256: `ce3c6d2167717bf1627d1c0c81cbccd28323cd4aa7b96d542599366d5ff6aac8`.

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

## Numeric leakage screen (|correlation with target| ≥ 0.95)

None flagged by this numeric-only screen. This does not rule out leakage.

Categorical leakage and feature availability require manual review.

## Numerical distributions

| Feature | Count | Mean | Std | Min | 25% | Median | 75% | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| person_age | 32574.0000 | 27.7184 | 6.2050 | 20.0000 | 23.0000 | 26.0000 | 30.0000 | 94.0000 |
| person_income | 32574.0000 | 65878.4808 | 52531.9388 | 4000.0000 | 38500.0000 | 55000.0000 | 79200.0000 | 2039784.0000 |
| person_emp_length | 31679.0000 | 4.7821 | 4.0349 | 0.0000 | 2.0000 | 4.0000 | 7.0000 | 41.0000 |
| loan_amnt | 32574.0000 | 9588.0181 | 6320.2496 | 500.0000 | 5000.0000 | 8000.0000 | 12200.0000 | 35000.0000 |
| loan_int_rate | 29459.0000 | 11.0115 | 3.2405 | 5.4200 | 7.9000 | 10.9900 | 13.4700 | 23.2200 |
| loan_percent_income | 32574.0000 | 0.1702 | 0.1068 | 0.0000 | 0.0900 | 0.1500 | 0.2300 | 0.8300 |
| cb_person_cred_hist_length | 32574.0000 | 5.8041 | 4.0539 | 2.0000 | 3.0000 | 4.0000 | 8.0000 | 30.0000 |

## Categorical distributions

- `person_home_ownership`: {'RENT': 16442, 'MORTGAGE': 13441, 'OWN': 2584, 'OTHER': 107}
- `loan_intent`: {'EDUCATION': 6451, 'MEDICAL': 6071, 'VENTURE': 5716, 'PERSONAL': 5519, 'DEBTCONSOLIDATION': 5212, 'HOMEIMPROVEMENT': 3605}
- `loan_grade`: {'A': 10776, 'B': 10448, 'C': 6456, 'D': 3625, 'E': 964, 'F': 241, 'G': 64}
- `cb_person_default_on_file`: {'N': 26830, 'Y': 5744}

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
