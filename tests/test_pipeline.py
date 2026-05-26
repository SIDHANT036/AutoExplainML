import pytest
import pandas as pd
import numpy as np
from unittest import mock
from autoexplainml.core.pipeline import run_pipeline


# ── return type ────────────────────────────────────────────────────────────────

def test_pipeline_returns_something(trained_rf):
    model, X = trained_rf
    result = run_pipeline(model, X)
    assert result is not None


def test_pipeline_result_is_dict(trained_rf):
    model, X = trained_rf
    result = run_pipeline(model, X)
    assert isinstance(result, dict)


# ── calls all three stages ─────────────────────────────────────────────────────

def test_pipeline_calls_explain(trained_rf):
    model, X = trained_rf
    with mock.patch("autoexplainml.core.pipeline.explain") as mock_explain:
        mock_explain.return_value = {"shap_values": [0.1]}
        with mock.patch("autoexplainml.core.pipeline.check_data_quality") as mock_dq:
            mock_dq.return_value = {"missing": 0}
            with mock.patch("autoexplainml.core.pipeline.build_report") as mock_report:
                mock_report.return_value = {"status": "ok"}
                run_pipeline(model, X)
                mock_explain.assert_called_once_with(model, X)


def test_pipeline_calls_data_quality(trained_rf):
    model, X = trained_rf
    with mock.patch("autoexplainml.core.pipeline.explain") as mock_explain:
        mock_explain.return_value = {}
        with mock.patch("autoexplainml.core.pipeline.check_data_quality") as mock_dq:
            mock_dq.return_value = {}
            with mock.patch("autoexplainml.core.pipeline.build_report") as mock_report:
                mock_report.return_value = {}
                run_pipeline(model, X)
                mock_dq.assert_called_once_with(X)


def test_pipeline_calls_build_report(trained_rf):
    model, X = trained_rf
    with mock.patch("autoexplainml.core.pipeline.explain") as mock_explain:
        mock_explain.return_value = {"shap": 1}
        with mock.patch("autoexplainml.core.pipeline.check_data_quality") as mock_dq:
            mock_dq.return_value = {"missing": 0}
            with mock.patch("autoexplainml.core.pipeline.build_report") as mock_report:
                mock_report.return_value = {"done": True}
                result = run_pipeline(model, X)
                mock_report.assert_called_once()
                assert result == {"done": True}


# ── different model types ──────────────────────────────────────────────────────

def test_pipeline_with_random_forest(trained_rf):
    model, X = trained_rf
    result = run_pipeline(model, X)
    assert result is not None


def test_pipeline_with_logistic_regression(trained_lr):
    model, X = trained_lr
    result = run_pipeline(model, X)
    assert result is not None


def test_pipeline_with_decision_tree(trained_dt):
    model, X = trained_dt
    result = run_pipeline(model, X)
    assert result is not None


def test_pipeline_with_gradient_boosting(trained_gb):
    model, X = trained_gb
    result = run_pipeline(model, X)
    assert result is not None


# ── bad inputs ─────────────────────────────────────────────────────────────────

def test_pipeline_raises_on_none_model(small_df):
    with pytest.raises((ValueError, AttributeError)):
        run_pipeline(None, small_df)


def test_pipeline_raises_on_none_data(trained_rf):
    model, _ = trained_rf
    with pytest.raises((ValueError, AttributeError, TypeError)):
        run_pipeline(model, None)


def test_pipeline_raises_on_string_input(trained_rf):
    model, _ = trained_rf
    with pytest.raises(Exception):
        run_pipeline(model, "not_a_dataframe")


# ── numpy input ────────────────────────────────────────────────────────────────

def test_pipeline_accepts_numpy_array(trained_rf):
    model, _ = trained_rf
    X_np = np.random.rand(10, 4)
    result = run_pipeline(model, X_np)
    assert result is not None
