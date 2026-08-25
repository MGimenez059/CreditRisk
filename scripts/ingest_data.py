#!/usr/bin/env python3
"""CLI entrypoint for data ingestion.

Validates a manually placed copy of the training dataset against the
schema in `docs/data_dictionary.md`, writes a data quality report, and (if
validation passes) writes a cleaned copy to `data/interim/` ready for
Phase 3 feature engineering.

This dataset (Kaggle `laotse/credit-risk-dataset`) is not auto-downloaded:
Kaggle requires an authenticated account to fetch it, and no Kaggle client
is in this project's dependencies (see `README.md`'s Tech Stack table).
Download it manually from
https://www.kaggle.com/datasets/laotse/credit-risk-dataset and place the
CSV at `data/raw/credit_risk_dataset.csv` before running this script.

Per SPECS.md §28, the eight checks it lists split into two tiers here:
required columns, dtypes, invalid categories, and impossible numerical
values are hard failures that stop the pipeline (`validate_dataset`).
Null percentages, duplicate rows, target distribution, and outliers have
no single "correct" value to enforce — they are surfaced in the data
quality report (SPECS.md §29) for a human to review, not auto-rejected.

Usage:
    python scripts/ingest_data.py
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk.exceptions import DataValidationError

RAW_DATA_PATH = Path("data/raw/credit_risk_dataset.csv")
INTERIM_DATA_PATH = Path("data/interim/credit_risk_validated.parquet")
QUALITY_REPORT_PATH = Path("docs/data_quality_report.md")

KAGGLE_URL = "https://www.kaggle.com/datasets/laotse/credit-risk-dataset"

# Required columns and their expected pandas dtype kind, per
# docs/data_dictionary.md. "i"/"f" = integer/float, "O" = object (string).
REQUIRED_COLUMNS: dict[str, str] = {
    "person_age": "if",
    "person_income": "if",
    "person_home_ownership": "O",
    "person_emp_length": "if",
    "loan_intent": "O",
    "loan_grade": "O",
    "loan_amnt": "if",
    "loan_int_rate": "if",
    "loan_status": "if",
    "loan_percent_income": "if",
    "cb_person_default_on_file": "O",
    "cb_person_cred_hist_length": "if",
}

ALLOWED_CATEGORICAL_VALUES: dict[str, set[str]] = {
    "person_home_ownership": {"RENT", "OWN", "MORTGAGE", "OTHER"},
    "loan_intent": {
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    },
    "loan_grade": {"A", "B", "C", "D", "E", "F", "G"},
    "cb_person_default_on_file": {"Y", "N"},
}

# (min, max) inclusive bounds. `None` means unbounded on that side. Chosen
# to catch data-entry errors (negative amounts, ages outside any plausible
# adult borrower range), not to reject legitimate outliers — see
# `compute_data_quality_report`'s "potential_outliers" field for that.
NUMERIC_RANGES: dict[str, tuple[float | None, float | None]] = {
    "person_age": (18, 100),
    "person_income": (0, None),
    "person_emp_length": (0, None),
    "loan_amnt": (0, None),
    "loan_int_rate": (0, 100),
    "loan_percent_income": (0, 1),
    "cb_person_cred_hist_length": (0, None),
    "loan_status": (0, 1),
}

TARGET_COLUMN = "loan_status"

# Correlation threshold above which a numeric feature is flagged as a
# possible leakage risk (SPECS.md §29's "Potential leakage"). This is a
# heuristic, not a determination — a flagged column needs a human look,
# per SPECS.md §28's "stop the pipeline rather than silently corrupting
# data": this script reports the flag but does not auto-drop the column.
LEAKAGE_CORRELATION_THRESHOLD = 0.95


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a raw dataset against the expected schema.

    Attributes:
        schema_issues: Missing columns, wrong dtypes, or a required column
            that is entirely null. Any entry here is fatal (SPECS.md §28).
        value_issues: Out-of-range values or invalid categories. Any entry
            here is fatal.
    """

    schema_issues: list[str] = field(default_factory=list)
    value_issues: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Whether the dataset passed every check."""
        return not self.schema_issues and not self.value_issues


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame.

    Args:
        path: Path to the raw dataset CSV.

    Returns:
        The loaded, unvalidated DataFrame.

    Raises:
        DataValidationError: If no file exists at `path`.
    """
    if not path.exists():
        raise DataValidationError(
            f"No raw dataset found at '{path}'. Download it manually from "
            f"{KAGGLE_URL} and save the CSV to this path."
        )
    return pd.read_csv(path)


def validate_schema(dataframe: pd.DataFrame) -> list[str]:
    """Check that every required column is present, non-empty, with a compatible dtype.

    Args:
        dataframe: The raw, loaded dataset.

    Returns:
        Human-readable issue descriptions. Empty if the schema is valid.
    """
    issues: list[str] = []

    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        issues.append(f"Missing required columns: {sorted(missing)}")

    for column, expected_kind in REQUIRED_COLUMNS.items():
        if column not in dataframe.columns:
            continue
        non_null = dataframe[column].dropna()
        if non_null.empty:
            # A required column that is 100% null is structurally the same
            # problem as a missing column — dtype inference on an empty
            # selection would otherwise hide this from the check below.
            issues.append(f"Column '{column}' is present but entirely null")
            continue
        actual_kind = non_null.dtype.kind
        if actual_kind not in expected_kind:
            issues.append(
                f"Column '{column}' has dtype kind '{actual_kind}', expected one of "
                f"'{expected_kind}'"
            )

    return issues


def validate_values(dataframe: pd.DataFrame) -> list[str]:
    """Check categorical values and numeric ranges for impossible values.

    Args:
        dataframe: The raw, loaded dataset. Assumed to have already passed
            `validate_schema`.

    Returns:
        Human-readable issue descriptions. Empty if all values are within
        the expected domain.
    """
    issues: list[str] = []

    for column, allowed_values in ALLOWED_CATEGORICAL_VALUES.items():
        if column not in dataframe.columns:
            continue
        observed = set(dataframe[column].dropna().unique())
        unexpected = observed - allowed_values
        if unexpected:
            issues.append(f"Column '{column}' has unexpected categories: {sorted(unexpected)}")

    for column, (low, high) in NUMERIC_RANGES.items():
        if column not in dataframe.columns:
            continue
        series = dataframe[column].dropna()
        if low is not None and (series < low).any():
            issues.append(f"Column '{column}' has values below the allowed minimum ({low})")
        if high is not None and (series > high).any():
            issues.append(f"Column '{column}' has values above the allowed maximum ({high})")

    return issues


def validate_dataset(dataframe: pd.DataFrame) -> ValidationResult:
    """Run schema and value validation together.

    Value validation only runs if the schema is valid — checking ranges on
    a column with the wrong dtype, or a missing column, isn't meaningful.

    Args:
        dataframe: The raw, loaded dataset.

    Returns:
        The combined validation result.
    """
    schema_issues = validate_schema(dataframe)
    value_issues = validate_values(dataframe) if not schema_issues else []
    return ValidationResult(schema_issues=schema_issues, value_issues=value_issues)


def _raise_if_invalid(result: ValidationResult) -> None:
    """Raise `DataValidationError` if validation found any issues.

    The single point where a `ValidationResult` becomes a stopped pipeline,
    per CODESTYLE.md §9: data pipeline validation failures must stop the
    pipeline, never be logged and ignored or silently coerced.

    Args:
        result: Output of `validate_dataset`.

    Raises:
        DataValidationError: If any schema or value issue was found.
    """
    if result.is_valid:
        return
    issues = "\n".join(f"  - {issue}" for issue in [*result.schema_issues, *result.value_issues])
    raise DataValidationError(
        f"Data validation failed — stopping the pipeline (SPECS.md §28):\n{issues}"
    )


# The report mixes scalar counts, nested per-column distributions, and
# floating-point correlations — a precise TypedDict would need a distinct
# value type per key, which pandas' `.to_dict()` output (numpy scalar keys
# and values) doesn't map onto cleanly. `Any` is used deliberately here
# rather than forcing an imprecise or overly complex type (CODESTYLE.md §6).
DataQualityReport = dict[str, Any]


def _compute_outliers(dataframe: pd.DataFrame, numeric_columns: list[str]) -> dict[str, int]:
    """Count IQR-method outliers per numeric column.

    Args:
        dataframe: The raw, loaded dataset.
        numeric_columns: Columns to check, excluding the target.

    Returns:
        Outlier row count per column.
    """
    outliers: dict[str, int] = {}
    for column in numeric_columns:
        series = dataframe[column].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers[column] = int(((series < lower) | (series > upper)).sum())
    return outliers


def _compute_leakage_flags(dataframe: pd.DataFrame, numeric_columns: list[str]) -> dict[str, float]:
    """Flag numeric features highly correlated with the target.

    Reported, not blocking: a feature this correlated with the target is a
    leakage *risk*, not proof of leakage (e.g., a genuinely predictive
    feature can also score high) — see the module docstring's tiering.

    Args:
        dataframe: The raw, loaded dataset.
        numeric_columns: Columns to check, excluding the target.

    Returns:
        `{column: correlation}` for every column at or above
        `LEAKAGE_CORRELATION_THRESHOLD`.
    """
    if TARGET_COLUMN not in dataframe.columns:
        return {}

    correlations = dataframe[[*numeric_columns, TARGET_COLUMN]].corr(numeric_only=True)[
        TARGET_COLUMN
    ]
    return {
        str(feature_name): round(float(correlation), 4)
        for feature_name, correlation in correlations.items()
        if feature_name != TARGET_COLUMN and abs(correlation) >= LEAKAGE_CORRELATION_THRESHOLD
    }


def compute_data_quality_report(dataframe: pd.DataFrame) -> DataQualityReport:
    """Compute the data quality report fields required by SPECS.md §29.

    Args:
        dataframe: The raw, loaded dataset.

    Returns:
        A dict with rows, columns, missing values (count and percentage),
        duplicates, unique values, numerical and categorical distributions,
        target distribution, potential outliers, and potential leakage.
    """
    numeric_columns = [
        column
        for column, kind in REQUIRED_COLUMNS.items()
        if kind in {"i", "if"} and column in dataframe.columns and column != TARGET_COLUMN
    ]
    categorical_columns = [
        column for column in ALLOWED_CATEGORICAL_VALUES if column in dataframe.columns
    ]

    row_count = len(dataframe)
    missing_counts = {
        column: int(count) for column, count in dataframe.isna().sum().items() if count > 0
    }

    return {
        "rows": row_count,
        "columns": len(dataframe.columns),
        "missing_values": missing_counts,
        "missing_percentages": {
            column: round(count / row_count * 100, 2) for column, count in missing_counts.items()
        },
        # Reported, not blocking: legitimate duplicate applications are
        # plausible in this domain (two people can share every recorded
        # attribute) — a human decides whether to deduplicate in Phase 3.
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "unique_values": {column: int(dataframe[column].nunique()) for column in dataframe.columns},
        "numerical_distributions": {
            column: dataframe[column].describe().to_dict() for column in numeric_columns
        },
        "categorical_distributions": {
            column: dataframe[column].value_counts().to_dict() for column in categorical_columns
        },
        "target_distribution": (
            dataframe[TARGET_COLUMN].value_counts(normalize=True).to_dict()
            if TARGET_COLUMN in dataframe.columns
            else {}
        ),
        "potential_outliers": _compute_outliers(dataframe, numeric_columns),
        "potential_leakage": _compute_leakage_flags(dataframe, numeric_columns),
    }


def render_quality_report_markdown(report: DataQualityReport, source_path: Path) -> str:
    """Render the data quality report dict as Markdown.

    Args:
        report: Output of `compute_data_quality_report`.
        source_path: Path to the raw file the report was generated from,
            included for traceability.

    Returns:
        The report as a Markdown document, ready to write to disk.
    """
    lines = [
        "# Data Quality Report",
        "",
        f"Generated from `{source_path}`. See SPECS.md §29 for the field list this report follows.",
        "",
        f"- **Rows:** {report['rows']}",
        f"- **Columns:** {report['columns']}",
        f"- **Duplicate rows:** {report['duplicate_rows']}",
        "",
        "## Missing values",
        "",
    ]
    if report["missing_values"]:
        lines += [
            f"- `{col}`: {count} ({report['missing_percentages'][col]}%)"
            for col, count in report["missing_values"].items()
        ]
    else:
        lines.append("None.")

    lines += ["", "## Target distribution (`loan_status`)", ""]
    lines += [f"- `{label}`: {share:.2%}" for label, share in report["target_distribution"].items()]

    lines += ["", "## Potential outliers (IQR method, per numeric column)", ""]
    lines += [f"- `{col}`: {count} rows" for col, count in report["potential_outliers"].items()]

    lines += ["", "## Potential leakage (|correlation with target| ≥ 0.95)", ""]
    if report["potential_leakage"]:
        lines += [f"- `{col}`: {corr}" for col, corr in report["potential_leakage"].items()]
    else:
        lines.append("None flagged.")

    lines += ["", "## Unique values per column", ""]
    lines += [f"- `{col}`: {count}" for col, count in report["unique_values"].items()]

    return "\n".join(lines) + "\n"


def main() -> int:
    """Run data ingestion end-to-end: load, validate, report, and persist.

    Returns:
        Process exit code: 0 on success, 1 if the pipeline cannot proceed.
    """
    try:
        dataframe = load_raw_data(RAW_DATA_PATH)
        _raise_if_invalid(validate_dataset(dataframe))
    except DataValidationError as err:
        print(err, file=sys.stderr)
        return 1

    report = compute_data_quality_report(dataframe)
    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT_PATH.write_text(
        render_quality_report_markdown(report, RAW_DATA_PATH), encoding="utf-8"
    )
    print(f"Data quality report written to '{QUALITY_REPORT_PATH}'.")

    INTERIM_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(INTERIM_DATA_PATH, index=False)
    print(f"Validated dataset written to '{INTERIM_DATA_PATH}'.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
