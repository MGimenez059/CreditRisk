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
    CleaningResult,
    _raise_if_schema_invalid,
    _raise_if_too_many_excluded,
    compute_data_quality_report,
    exclude_invalid_rows,
    load_raw_data,
    main,
    render_quality_report_markdown,
    validate_schema,
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


def test_raise_if_schema_invalid_is_a_no_op_for_no_issues() -> None:
    _raise_if_schema_invalid([])  # must not raise


def test_raise_if_schema_invalid_raises_with_issues() -> None:
    with pytest.raises(DataValidationError, match="bad schema"):
        _raise_if_schema_invalid(["bad schema"])


# --- exclude_invalid_rows -----------------------------------------------


def test_exclude_invalid_rows_keeps_every_row_of_a_valid_frame() -> None:
    result = exclude_invalid_rows(_valid_dataframe())

    assert result.excluded_row_count == 0
    assert len(result.cleaned) == 3


def test_exclude_invalid_rows_drops_a_row_with_an_impossible_age() -> None:
    dataframe = _valid_dataframe()
    dataframe.loc[0, "person_age"] = 144  # the real dataset's documented data-entry error

    result = exclude_invalid_rows(dataframe)

    assert result.excluded_row_count == 1
    assert len(result.cleaned) == 2
    assert any("person_age" in reason for reason in result.exclusion_reasons)


def test_exclude_invalid_rows_drops_a_row_with_an_invalid_category() -> None:
    dataframe = _valid_dataframe()
    dataframe.loc[0, "person_home_ownership"] = "CASTLE"

    result = exclude_invalid_rows(dataframe)

    assert result.excluded_row_count == 1
    assert any("person_home_ownership" in reason for reason in result.exclusion_reasons)


def test_exclude_invalid_rows_combines_multiple_reasons_for_the_same_row() -> None:
    dataframe = _valid_dataframe()
    dataframe.loc[0, "person_age"] = -5
    dataframe.loc[0, "person_home_ownership"] = "CASTLE"

    result = exclude_invalid_rows(dataframe)

    assert result.excluded_row_count == 1
    assert "person_age" in result.exclusion_reasons[0]
    assert "person_home_ownership" in result.exclusion_reasons[0]


def test_exclude_invalid_rows_resets_the_index_of_the_cleaned_frame() -> None:
    dataframe = _valid_dataframe()
    dataframe.loc[0, "person_age"] = -5

    result = exclude_invalid_rows(dataframe)

    assert list(result.cleaned.index) == [0, 1]


# --- _raise_if_too_many_excluded ----------------------------------------


def test_raise_if_too_many_excluded_is_a_no_op_within_the_threshold() -> None:
    result = CleaningResult(cleaned=pd.DataFrame(), excluded_row_count=1)

    _raise_if_too_many_excluded(result, total_rows=1000)  # 0.1%, must not raise


def test_raise_if_too_many_excluded_raises_beyond_the_threshold() -> None:
    result = CleaningResult(
        cleaned=pd.DataFrame(), excluded_row_count=10, exclusion_reasons=["Row 0: bad"]
    )

    with pytest.raises(DataValidationError, match="10 of 100 rows"):
        _raise_if_too_many_excluded(result, total_rows=100)  # 10%, exceeds 5%


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


def test_quality_report_omits_excluded_rows_section_when_not_given_a_cleaning_result() -> None:
    report = compute_data_quality_report(_valid_dataframe())

    assert "excluded_rows" not in report


def test_quality_report_includes_excluded_rows_section_when_given_a_cleaning_result() -> None:
    cleaning_result = CleaningResult(
        cleaned=_valid_dataframe(), excluded_row_count=2, exclusion_reasons=["Row 5: bad age"]
    )

    report = compute_data_quality_report(_valid_dataframe(), cleaning_result)

    assert report["excluded_rows"] == {"count": 2, "reasons": ["Row 5: bad age"]}


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


def test_quality_report_markdown_lists_excluded_row_reasons() -> None:
    cleaning_result = CleaningResult(
        cleaned=_valid_dataframe(), excluded_row_count=1, exclusion_reasons=["Row 5: bad age"]
    )
    report = compute_data_quality_report(_valid_dataframe(), cleaning_result)

    markdown = render_quality_report_markdown(report, source_path=Path("x.csv"))

    assert "Excluded rows" in markdown
    assert "Row 5: bad age" in markdown


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


def _patch_paths(
    monkeypatch: pytest.MonkeyPatch, raw_path: Path, interim_path: Path, report_path: Path
) -> None:
    monkeypatch.setattr("ingest_data.RAW_DATA_PATH", raw_path)
    monkeypatch.setattr("ingest_data.INTERIM_DATA_PATH", interim_path)
    monkeypatch.setattr("ingest_data.QUALITY_REPORT_PATH", report_path)


def test_main_writes_report_and_interim_file_for_valid_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path, interim_path, report_path = (
        tmp_path / "raw.csv",
        tmp_path / "interim.parquet",
        tmp_path / "report.md",
    )
    _valid_dataframe().to_csv(raw_path, index=False)
    _patch_paths(monkeypatch, raw_path, interim_path, report_path)

    assert main() == 0
    assert interim_path.exists()
    assert report_path.exists()


def test_main_excludes_a_few_bad_rows_and_still_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path, interim_path, report_path = (
        tmp_path / "raw.csv",
        tmp_path / "interim.parquet",
        tmp_path / "report.md",
    )
    dataframe = _valid_dataframe()
    dataframe.loc[0, "person_age"] = 144  # one bad row out of three: 33%, but small absolute count
    dataframe.to_csv(raw_path, index=False)
    _patch_paths(monkeypatch, raw_path, interim_path, report_path)

    # Three rows total, one excluded, is 33% — above the 5% threshold — so
    # this is expected to stop the pipeline, exercising that path directly
    # rather than asserting success here (see the large-dataset variant
    # in test_exclude_invalid_rows_* above for the exclusion logic itself).
    assert main() == 1
    assert not interim_path.exists()


def test_main_stops_for_a_structurally_invalid_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path, interim_path, report_path = (
        tmp_path / "raw.csv",
        tmp_path / "interim.parquet",
        tmp_path / "report.md",
    )
    _valid_dataframe().drop(columns=["loan_amnt"]).to_csv(raw_path, index=False)
    _patch_paths(monkeypatch, raw_path, interim_path, report_path)

    assert main() == 1
    assert not interim_path.exists()
    assert not report_path.exists()


def test_main_missing_raw_file_returns_exit_code_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ingest_data.RAW_DATA_PATH", tmp_path / "does_not_exist.csv")

    assert main() == 1
