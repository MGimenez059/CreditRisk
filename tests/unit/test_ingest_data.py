"""Unit tests for `scripts/ingest_data.py`.

Uses small, inline, clearly-synthetic DataFrames — never a fixture that
could be mistaken for a sample of the real Kaggle dataset. `scripts/` is on
`pythonpath` (see `pyproject.toml`), so `ingest_data` imports directly.

Tests touching the filesystem use pytest's `tmp_path` fixture exclusively
(never a real project path), following the same pattern already used in
`tests/unit/test_registry.py` for `ml.registry`'s save/load functions.
"""

from pathlib import Path

import pandas as pd
import pytest

from credit_risk.exceptions import DataValidationError
from ingest_data import (
    ValidationResult,
    _raise_if_invalid,
    compute_data_quality_report,
    load_raw_data,
    main,
    render_quality_report_markdown,
    validate_dataset,
    validate_schema,
    validate_values,
)

_VALID_ROWS = {
    "person_age": [25, 40, 33],
    "person_income": [45_000, 82_000, 60_000],
    "person_home_ownership": ["RENT", "MORTGAGE", "OWN"],
    "person_emp_length": [2.0, 10.0, 5.0],
    "loan_intent": ["PERSONAL", "EDUCATION", "VENTURE"],
    "loan_grade": ["B", "A", "C"],
    "loan_amnt": [10_000, 25_000, 15_000],
    "loan_int_rate": [11.5, 7.9, 13.2],
    "loan_status": [0, 0, 1],
    "loan_percent_income": [0.22, 0.30, 0.25],
    "cb_person_default_on_file": ["N", "N", "Y"],
    "cb_person_cred_hist_length": [3, 12, 6],
}


def _valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(_VALID_ROWS)


# --- validate_schema ---------------------------------------------------


def test_validate_schema_accepts_a_complete_valid_frame() -> None:
    assert validate_schema(_valid_dataframe()) == []


def test_validate_schema_flags_missing_required_column() -> None:
    dataframe = _valid_dataframe().drop(columns=["loan_amnt"])

    issues = validate_schema(dataframe)

    assert any("loan_amnt" in issue for issue in issues)


def test_validate_schema_flags_wrong_dtype() -> None:
    dataframe = _valid_dataframe()
    dataframe["person_age"] = dataframe["person_age"].astype(str)

    issues = validate_schema(dataframe)

    assert any("person_age" in issue for issue in issues)


def test_validate_schema_flags_entirely_null_required_column() -> None:
    dataframe = _valid_dataframe()
    dataframe["loan_int_rate"] = None

    issues = validate_schema(dataframe)

    assert any("loan_int_rate" in issue and "entirely null" in issue for issue in issues)


# --- validate_values -----------------------------------------------------


def test_validate_values_accepts_a_valid_frame() -> None:
    assert validate_values(_valid_dataframe()) == []


def test_validate_values_flags_negative_age() -> None:
    dataframe = _valid_dataframe()
    dataframe.loc[0, "person_age"] = -5

    issues = validate_values(dataframe)

    assert any("person_age" in issue for issue in issues)


def test_validate_values_flags_unexpected_home_ownership_category() -> None:
    dataframe = _valid_dataframe()
    dataframe.loc[0, "person_home_ownership"] = "CASTLE"

    issues = validate_values(dataframe)

    assert any("person_home_ownership" in issue for issue in issues)


def test_validate_values_flags_loan_status_outside_binary_range() -> None:
    dataframe = _valid_dataframe()
    dataframe.loc[0, "loan_status"] = 2

    issues = validate_values(dataframe)

    assert any("loan_status" in issue for issue in issues)


# --- validate_dataset ------------------------------------------------------


def test_validate_dataset_is_valid_for_clean_data() -> None:
    result = validate_dataset(_valid_dataframe())

    assert result.is_valid


def test_validate_dataset_skips_value_checks_when_schema_is_invalid() -> None:
    dataframe = _valid_dataframe().drop(columns=["person_age"])

    result = validate_dataset(dataframe)

    assert not result.is_valid
    assert result.value_issues == []


# --- _raise_if_invalid -------------------------------------------------


def test_raise_if_invalid_is_a_no_op_for_a_valid_result() -> None:
    _raise_if_invalid(ValidationResult())  # must not raise


def test_raise_if_invalid_raises_data_validation_error_listing_issues() -> None:
    result = ValidationResult(schema_issues=["bad schema"], value_issues=["bad value"])

    with pytest.raises(DataValidationError, match="bad schema"):
        _raise_if_invalid(result)


# --- compute_data_quality_report / render_quality_report_markdown ----------


@pytest.mark.parametrize(
    "field_name",
    [
        "rows",
        "columns",
        "missing_values",
        "missing_percentages",
        "duplicate_rows",
        "target_distribution",
    ],
)
def test_quality_report_contains_required_fields(field_name: str) -> None:
    report = compute_data_quality_report(_valid_dataframe())

    assert field_name in report


def test_quality_report_detects_a_duplicate_row() -> None:
    dataframe = pd.concat([_valid_dataframe(), _valid_dataframe().iloc[[0]]], ignore_index=True)

    report = compute_data_quality_report(dataframe)

    assert report["duplicate_rows"] == 1


def test_quality_report_computes_missing_percentage() -> None:
    dataframe = _valid_dataframe()
    dataframe.loc[0, "person_emp_length"] = None

    report = compute_data_quality_report(dataframe)

    assert report["missing_percentages"]["person_emp_length"] == pytest.approx(33.33, abs=0.01)


def test_quality_report_renders_to_markdown() -> None:
    report = compute_data_quality_report(_valid_dataframe())

    markdown = render_quality_report_markdown(report, source_path=Path("x.csv"))

    assert "# Data Quality Report" in markdown
    assert "Target distribution" in markdown


# --- load_raw_data (filesystem I/O via tmp_path, per test_registry.py's pattern) --


def test_load_raw_data_missing_file_raises_data_validation_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(DataValidationError):
        load_raw_data(missing_path)


def test_load_raw_data_reads_a_valid_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    _valid_dataframe().to_csv(csv_path, index=False)

    loaded = load_raw_data(csv_path)

    assert len(loaded) == 3
    assert "person_age" in loaded.columns


# --- main() end-to-end (filesystem I/O via tmp_path + monkeypatch) ---------


def test_main_writes_report_and_interim_file_for_valid_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path = tmp_path / "raw.csv"
    interim_path = tmp_path / "interim.parquet"
    report_path = tmp_path / "report.md"
    _valid_dataframe().to_csv(raw_path, index=False)

    monkeypatch.setattr("ingest_data.RAW_DATA_PATH", raw_path)
    monkeypatch.setattr("ingest_data.INTERIM_DATA_PATH", interim_path)
    monkeypatch.setattr("ingest_data.QUALITY_REPORT_PATH", report_path)

    exit_code = main()

    assert exit_code == 0
    assert interim_path.exists()
    assert report_path.exists()


def test_main_stops_and_writes_nothing_for_invalid_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path = tmp_path / "raw.csv"
    interim_path = tmp_path / "interim.parquet"
    report_path = tmp_path / "report.md"
    dataframe = _valid_dataframe()
    dataframe.loc[0, "person_age"] = -5
    dataframe.to_csv(raw_path, index=False)

    monkeypatch.setattr("ingest_data.RAW_DATA_PATH", raw_path)
    monkeypatch.setattr("ingest_data.INTERIM_DATA_PATH", interim_path)
    monkeypatch.setattr("ingest_data.QUALITY_REPORT_PATH", report_path)

    exit_code = main()

    assert exit_code == 1
    assert not interim_path.exists()
    assert not report_path.exists()


def test_main_missing_raw_file_returns_exit_code_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ingest_data.RAW_DATA_PATH", tmp_path / "does_not_exist.csv")

    assert main() == 1
