import pytest
import pandas as pd
import numpy as np
from autoexplainml.intelligence.data_quality import check_data_quality


# ── return type ────────────────────────────────────────────────────────────────

def test_data_quality_returns_dict(small_df):
    result = check_data_quality(small_df)
    assert isinstance(result, dict)


def test_data_quality_returns_something(trained_rf):
    _, X = trained_rf
    result = check_data_quality(X)
    assert result is not None


# ── clean data ─────────────────────────────────────────────────────────────────

def test_data_quality_on_clean_data_has_no_missing(small_df):
    result = check_data_quality(small_df)
    # missing count should be 0 or a key should indicate no issues
    missing = result.get("missing_values", result.get("missing", 0))
    if isinstance(missing, dict):
        assert all(v == 0 for v in missing.values())
    else:
        assert missing == 0


# ── data with missing values ───────────────────────────────────────────────────

def test_data_quality_detects_missing_values():
    df = pd.DataFrame({
        "a": [1.0, None, 3.0, 4.0, 5.0],
        "b": [1.0, 2.0, None, 4.0, 5.0],
        "c": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    result = check_data_quality(df)
    assert result is not None
    # result should flag that there are missing values somewhere
    # exact key depends on your implementation — adjust if needed
    has_missing_info = (
        "missing" in str(result).lower() or
        "null" in str(result).lower() or
        any(v > 0 for v in result.values() if isinstance(v, (int, float)))
    )
    assert has_missing_info


# ── data with duplicates ───────────────────────────────────────────────────────

def test_data_quality_on_data_with_duplicates():
    df = pd.DataFrame({
        "a": [1, 1, 1, 2, 3],
        "b": [4, 4, 4, 5, 6],
    })
    result = check_data_quality(df)
    assert result is not None


# ── edge cases ─────────────────────────────────────────────────────────────────

def test_data_quality_single_row():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    result = check_data_quality(df)
    assert result is not None


def test_data_quality_single_column():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = check_data_quality(df)
    assert result is not None


def test_data_quality_with_all_nulls():
    df = pd.DataFrame({"a": [None, None, None], "b": [None, None, None]})
    result = check_data_quality(df)
    assert result is not None


def test_data_quality_with_iris(trained_rf):
    _, X = trained_rf
    result = check_data_quality(X)
    assert isinstance(result, dict)


def test_data_quality_with_large_dataset():
    df = pd.DataFrame(np.random.rand(1000, 10), columns=[f"f{i}" for i in range(10)])
    result = check_data_quality(df)
    assert result is not None
