import pytest
import pandas as pd
import numpy as np
from autoexplainml.core.validator import validate
from sklearn.ensemble import RandomForestClassifier


# ── valid inputs ───────────────────────────────────────────────────────────────

def test_validate_returns_true_for_valid_inputs(trained_rf):
    model, X = trained_rf
    assert validate(model, X) is True


def test_validate_accepts_numpy_array(trained_rf):
    model, _ = trained_rf
    X = np.random.rand(10, 4)
    assert validate(model, X) is True


def test_validate_accepts_small_dataframe(trained_rf, small_df):
    model, _ = trained_rf
    assert validate(model, small_df) is True


# ── invalid model ──────────────────────────────────────────────────────────────

def test_validate_raises_if_model_is_none(small_df):
    with pytest.raises(ValueError, match="[Mm]odel"):
        validate(None, small_df)


def test_validate_raises_if_model_is_string(small_df):
    with pytest.raises((ValueError, AttributeError)):
        validate("not_a_model", small_df)


# ── invalid data ───────────────────────────────────────────────────────────────

def test_validate_raises_if_X_is_none(trained_rf):
    model, _ = trained_rf
    with pytest.raises(ValueError):
        validate(model, None)


def test_validate_raises_if_X_is_empty_dataframe(trained_rf, empty_df):
    model, _ = trained_rf
    with pytest.raises(ValueError, match="[Ee]mpty|[Dd]ataset"):
        validate(model, empty_df)


def test_validate_raises_if_X_is_empty_list(trained_rf):
    model, _ = trained_rf
    with pytest.raises(ValueError):
        validate(model, [])


# ── edge cases ─────────────────────────────────────────────────────────────────

def test_validate_accepts_single_row(trained_rf):
    model, X = trained_rf
    single_row = X.iloc[[0]]
    assert validate(model, single_row) is True


def test_validate_accepts_large_dataframe():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    X_df = pd.DataFrame(X)
    assert validate(model, X_df) is True
