import pytest
import pandas as pd
import numpy as np
from unittest import mock
from autoexplainml.core.engine import explain


# ── tabular path ───────────────────────────────────────────────────────────────

def test_explain_returns_result_for_tabular(trained_rf):
    model, X = trained_rf
    result = explain(model, X)
    assert result is not None


def test_explain_result_is_not_empty(trained_rf):
    model, X = trained_rf
    result = explain(model, X)
    # result should be a non-empty dict or object
    assert result


def test_explain_works_with_logistic_regression(trained_lr):
    model, X = trained_lr
    result = explain(model, X)
    assert result is not None


def test_explain_works_with_gradient_boosting(trained_gb):
    model, X = trained_gb
    result = explain(model, X)
    assert result is not None


def test_explain_works_with_decision_tree(trained_dt):
    model, X = trained_dt
    result = explain(model, X)
    assert result is not None


# ── unsupported task ───────────────────────────────────────────────────────────

def test_explain_raises_on_unknown_task(trained_rf):
    model, X = trained_rf
    with mock.patch("autoexplainml.core.engine.detect_task", return_value="unknown_task"):
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            explain(model, X)


# ── shap explainer is called for tabular ──────────────────────────────────────

def test_explain_calls_shap_for_tabular(trained_rf):
    model, X = trained_rf
    with mock.patch("autoexplainml.core.engine.explain_shap") as mock_shap:
        mock_shap.return_value = {"shap_values": [0.1, 0.2]}
        explain(model, X)
        mock_shap.assert_called_once_with(model, X)


# ── numpy input ────────────────────────────────────────────────────────────────

def test_explain_accepts_numpy_array(trained_rf):
    model, _ = trained_rf
    X_np = np.random.rand(10, 4)
    result = explain(model, X_np)
    assert result is not None
