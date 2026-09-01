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

- Structural problems (missing columns, wrong dtypes, an entirely-null
  required column) cannot be fixed by dropping rows — they are hard
  failures that stop the pipeline entirely (`validate_schema`).
- Row-level problems (an impossible value, an invalid category) are
  excluded row by row rather than failing the whole file — a handful of
  bad rows in an otherwise-valid ~32k-row file is a data engineering
  reality, not proof the whole file is untrustworthy. Every exclusion is
  logged with its row index and reason (`exclude_invalid_rows`). If too
  large a fraction of rows would be excluded, that stops being "a few
  outliers" and becomes a signal of a systemic problem, so it still stops
  the pipeline (`MAX_EXCLUDED_ROW_FRACTION`).

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
# to catch data-entry errors (negative amounts, ages/employment lengths no
# real adult borrower could have), not to reject legitimate outliers — see
# `compute_data_quality_report`'s "potential_outliers" field for those.
# `person_emp_length`'s upper bound of 70 and `person_age`'s of 100 are
# deliberately generous (not the tightest plausible bound) so only
# genuinely impossible values are excluded, not merely unusual ones —
# found necessary in practice: the real Kaggle file contains rows with
# `person_age` of 123-144 and `person_emp_length` of 123, a documented
# data-entry-error quirk of this dataset.
NUMERIC_RANGES: dict[str, tuple[float | None, float | None]] = {
    "person_age": (18, 100),
    "person_income": (0, None),
    "person_emp_length": (0, 70),
    "loan_amnt": (0, None),
    "loan_int_rate": (0, 100),
    "loan_percent_income": (0, 1),
    "cb_person_cred_hist_length": (0, None),
    "loan_status": (0, 1),
}

TARGET_COLUMN = "loan_status"

# Beyond this fraction of rows excluded by `exclude_invalid_rows`, the
# problem is treated as systemic rather than "a few outliers" and stops
# the pipeline instead of silently proceeding with a much-reduced dataset.
MAX_EXCLUDED_ROW_FRACTION = 0.05

# Correlation threshold above which a numeric feature is flagged as a
# possible leakage risk (SPECS.md §29's "Potential leakage"). This is a
# heuristic, not a determination — a flagged column needs a human look,
# per SPECS.md §28's "stop the pipeline rather than silently corrupting
# data": this script reports the flag but does not auto-drop the column.
LEAKAGE_CORRELATION_THRESHOLD = 0.95


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

    Structural checks only — nothing here can be fixed by dropping a row,
    so any issue found is a hard failure (SPECS.md §28).

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


def _raise_if_schema_invalid(issues: list[str]) -> None:
    """Raise `DataValidationError` if `validate_schema` found any issues.

    Args:
        issues: Output of `validate_schema`.

    Raises:
        DataValidationError: If `issues` is non-empty.
    """
    if not issues:
        return
    formatted = "\n".join(f"  - {issue}" for issue in issues)
    raise DataValidationError(
        f"Data validation failed — stopping the pipeline (SPECS.md §28):\n{formatted}"
    )


@dataclass(frozen=True)
class CleaningResult:
    """Outcome of removing individually invalid rows from a schema-valid dataset.

    Attributes:
        cleaned: `dataframe` with invalid rows removed and the index reset.
        excluded_row_count: How many rows were removed.
        exclusion_reasons: One human-readable reason per excluded row
            (`"Row <original index>: <reason>"`), for the data quality
            report and for manual review of what was dropped and why.
    """

    cleaned: pd.DataFrame
    excluded_row_count: int
    exclusion_reasons: list[str] = field(default_factory=list)


def _to_native(value: Any) -> Any:  # noqa: ANN401 -- genuinely any scalar type pandas may hand back
    """Convert a numpy scalar to a native Python type for clean, readable repr().

    pandas' `.at[]` accessor returns numpy scalars (`numpy.int64`,
    `numpy.float64`) whose `repr()` is `"np.int64(144)"` rather than
    `"144"` — unreadable in a human-facing log or report. Non-numpy values
    (e.g. category strings) pass through unchanged.
    """
    return value.item() if hasattr(value, "item") else value


def _flag_categorical_violations(
    dataframe: pd.DataFrame, reasons_by_row: dict[int, list[str]]
) -> None:
    """Record a reason for every row whose categorical columns fail `ALLOWED_CATEGORICAL_VALUES`."""
    for column, allowed_values in ALLOWED_CATEGORICAL_VALUES.items():
        if column not in dataframe.columns:
            continue
        invalid = dataframe[column].notna() & ~dataframe[column].isin(allowed_values)
        for row_index in dataframe.index[invalid]:
            value = _to_native(dataframe.at[row_index, column])
            reasons_by_row.setdefault(row_index, []).append(
                f"{column}={value!r} not an allowed category"
            )


def _flag_numeric_violations(dataframe: pd.DataFrame, reasons_by_row: dict[int, list[str]]) -> None:
    """Record a reason for every row whose numeric columns fail `NUMERIC_RANGES`."""
    for column, (low, high) in NUMERIC_RANGES.items():
        if column not in dataframe.columns:
            continue
        series = dataframe[column]
        bounds: list[tuple[pd.Series, str, float]] = []
        if low is not None:
            bounds.append((series.notna() & (series < low), "below minimum", low))
        if high is not None:
            bounds.append((series.notna() & (series > high), "above maximum", high))

        for invalid_mask, bound_kind, bound in bounds:
            for row_index in dataframe.index[invalid_mask]:
                value = _to_native(dataframe.at[row_index, column])
                reasons_by_row.setdefault(row_index, []).append(
                    f"{column}={value!r} {bound_kind} {bound}"
                )


def exclude_invalid_rows(dataframe: pd.DataFrame) -> CleaningResult:
    """Remove rows with impossible values or invalid categories, with a reason logged for each.

    Unlike `validate_schema`'s structural checks, a row-level violation
    here does not fail the whole file — it excludes that one row. Every
    exclusion is logged by original row index and reason so the exclusion
    is auditable, never silent, per CODESTYLE.md §8 and SPECS.md §28.

    Args:
        dataframe: The raw, loaded dataset. Assumed to have already passed
            `validate_schema`.

    Returns:
        The cleaned dataframe alongside a full log of what was excluded
        and why. Does not raise — see `_raise_if_too_many_excluded` for
        the safety-threshold check on the result.
    """
    reasons_by_row: dict[int, list[str]] = {}
    _flag_categorical_violations(dataframe, reasons_by_row)
    _flag_numeric_violations(dataframe, reasons_by_row)

    invalid_rows = sorted(reasons_by_row)
    exclusion_reasons = [
        f"Row {row_index}: {'; '.join(reasons_by_row[row_index])}" for row_index in invalid_rows
    ]
    cleaned = dataframe.drop(index=invalid_rows).reset_index(drop=True)

    return CleaningResult(
        cleaned=cleaned,
        excluded_row_count=len(invalid_rows),
        exclusion_reasons=exclusion_reasons,
    )


def _raise_if_too_many_excluded(result: CleaningResult, total_rows: int) -> None:
    """Raise `DataValidationError` if too large a fraction of rows were excluded.

    Args:
        result: Output of `exclude_invalid_rows`.
        total_rows: Row count of the dataset before exclusion.

    Raises:
        DataValidationError: If the excluded fraction exceeds
            `MAX_EXCLUDED_ROW_FRACTION`.
    """
    fraction = result.excluded_row_count / total_rows if total_rows else 0.0
    if fraction <= MAX_EXCLUDED_ROW_FRACTION:
        return
    reasons = "\n".join(f"  - {reason}" for reason in result.exclusion_reasons)
    raise DataValidationError(
        f"{result.excluded_row_count} of {total_rows} rows ({fraction:.1%}) have "
        f"impossible values — exceeds the {MAX_EXCLUDED_ROW_FRACTION:.0%} safety "
        f"threshold (SPECS.md §28), which usually means a structural problem rather "
        f"than a few outliers. Reasons:\n{reasons}"
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
        dataframe: The (already-cleaned) dataset.
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
        dataframe: The (already-cleaned) dataset.
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


def compute_data_quality_report(
    dataframe: pd.DataFrame, cleaning_result: CleaningResult | None = None
) -> DataQualityReport:
    """Compute the data quality report fields required by SPECS.md §29.

    Args:
        dataframe: The (already-cleaned) dataset that will be persisted.
        cleaning_result: Output of `exclude_invalid_rows`, if row exclusion
            ran. Adds an `excluded_rows` section when provided.

    Returns:
        A dict with rows, columns, missing values (count and percentage),
        duplicates, unique values, numerical and categorical distributions,
        target distribution, potential outliers, potential leakage, and
        (when `cleaning_result` is given) excluded row count and reasons.
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

    report: DataQualityReport = {
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

    if cleaning_result is not None:
        report["excluded_rows"] = {
            "count": cleaning_result.excluded_row_count,
            "reasons": cleaning_result.exclusion_reasons,
        }

    return report


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
        f"Generated from `{source_path.as_posix()}`. See SPECS.md §29 for the field list.",
        "",
        f"- **Rows (after exclusions):** {report['rows']}",
        f"- **Columns:** {report['columns']}",
        f"- **Duplicate rows:** {report['duplicate_rows']}",
        "",
    ]

    if "excluded_rows" in report:
        lines += ["## Excluded rows (impossible values)", ""]
        excluded = report["excluded_rows"]
        if excluded["count"]:
            lines.append(f"{excluded['count']} row(s) excluded before this report was computed:")
            lines += [f"- {reason}" for reason in excluded["reasons"]]
        else:
            lines.append("None excluded.")
        lines.append("")

    lines += ["## Missing values", ""]
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
    """Run data ingestion end-to-end: load, validate, clean, report, and persist.

    Returns:
        Process exit code: 0 on success, 1 if the pipeline cannot proceed.
    """
    try:
        dataframe = load_raw_data(RAW_DATA_PATH)
        _raise_if_schema_invalid(validate_schema(dataframe))

        cleaning_result = exclude_invalid_rows(dataframe)
        _raise_if_too_many_excluded(cleaning_result, total_rows=len(dataframe))
    except DataValidationError as err:
        print(err, file=sys.stderr)
        return 1

    if cleaning_result.excluded_row_count:
        print(
            f"Excluded {cleaning_result.excluded_row_count} row(s) with impossible values:",
            file=sys.stderr,
        )
        for reason in cleaning_result.exclusion_reasons:
            print(f"  - {reason}", file=sys.stderr)

    cleaned = cleaning_result.cleaned
    report = compute_data_quality_report(cleaned, cleaning_result)
    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT_PATH.write_text(
        render_quality_report_markdown(report, RAW_DATA_PATH), encoding="utf-8"
    )
    print(f"Data quality report written to '{QUALITY_REPORT_PATH}'.")

    INTERIM_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(INTERIM_DATA_PATH, index=False)
    print(f"Validated dataset written to '{INTERIM_DATA_PATH}'.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
