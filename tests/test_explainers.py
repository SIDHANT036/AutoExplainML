import pytest
import pandas as pd
import numpy as np

shap = pytest.importorskip("shap", reason="shap not installed — run: pip install autoexplainml[xai]")

from autoexplainml.explainers.shap_explainer import explain_shap, SHAPExplainer
from autoexplainml.explainers.permutation_explainer import explain_permutation, PermutationExplainer


# ── shap explainer — functional wrapper ───────────────────────────────────────

def test_shap_returns_dict(trained_rf):
    model, X = trained_rf
    result = explain_shap(model, X)
    assert isinstance(result, dict)


def test_shap_has_method_key(trained_rf):
    model, X = trained_rf
    result = explain_shap(model, X)
    assert result.get("method") == "shap"


def test_shap_has_features_key(trained_rf):
    model, X = trained_rf
    result = explain_shap(model, X)
    assert "features" in result
    assert isinstance(result["features"], list)


def test_shap_has_importance_key(trained_rf):
    model, X = trained_rf
    result = explain_shap(model, X)
    assert "importance" in result
    assert isinstance(result["importance"], list)


def test_shap_feature_count_matches_input(trained_rf):
    model, X = trained_rf
    result = explain_shap(model, X)
    assert len(result["features"]) == X.shape[1]
    assert len(result["importance"]) == X.shape[1]


def test_shap_importance_values_are_floats(trained_rf):
    model, X = trained_rf
    result = explain_shap(model, X)
    assert all(isinstance(v, (float, __import__("numpy").floating)) for v in result["importance"])


def test_shap_importance_values_are_non_negative(trained_rf):
    model, X = trained_rf
    result = explain_shap(model, X)
    assert all(v >= 0 for v in result["importance"])


def test_shap_works_with_logistic_regression(trained_lr):
    model, X = trained_lr
    result = explain_shap(model, X)
    assert result["method"] == "shap"


def test_shap_works_with_gradient_boosting(trained_gb):
    model, X = trained_gb
    result = explain_shap(model, X)
    assert isinstance(result, dict)


def test_shap_works_with_small_dataset(trained_rf):
    model, X = trained_rf
    result = explain_shap(model, X.iloc[:3])
    assert result is not None


# ── shap explainer — class interface ──────────────────────────────────────────

def test_shap_class_explain_returns_dict(trained_rf):
    model, X = trained_rf
    result = SHAPExplainer().explain(model, X)
    assert isinstance(result, dict)


def test_shap_class_accepts_numpy(trained_rf):
    model, _ = trained_rf
    X_np = np.random.rand(10, 4)
    result = SHAPExplainer().explain(model, X_np)
    assert "features" in result


# ── permutation explainer — functional wrapper ────────────────────────────────

def test_permutation_returns_dict(trained_rf):
    model, X = trained_rf
    result = explain_permutation(model, X)
    assert isinstance(result, dict)


def test_permutation_has_method_key(trained_rf):
    model, X = trained_rf
    result = explain_permutation(model, X)
    assert result.get("method") == "permutation"


def test_permutation_has_features_key(trained_rf):
    model, X = trained_rf
    result = explain_permutation(model, X)
    assert "features" in result
    assert isinstance(result["features"], list)


def test_permutation_has_importance_key(trained_rf):
    model, X = trained_rf
    result = explain_permutation(model, X)
    assert "importance" in result
    assert isinstance(result["importance"], list)


def test_permutation_feature_count_matches_input(trained_rf):
    model, X = trained_rf
    result = explain_permutation(model, X)
    assert len(result["features"]) == X.shape[1]
    assert len(result["importance"]) == X.shape[1]


def test_permutation_works_with_logistic_regression(trained_lr):
    model, X = trained_lr
    result = explain_permutation(model, X)
    assert result["method"] == "permutation"


def test_permutation_works_with_decision_tree(trained_dt):
    model, X = trained_dt
    result = explain_permutation(model, X)
    assert isinstance(result, dict)


# ── permutation explainer — class interface ───────────────────────────────────

def test_permutation_class_returns_dict(trained_rf):
    model, X = trained_rf
    result = PermutationExplainer().explain(model, X)
    assert isinstance(result, dict)


# ── lime explainer ─────────────────────────────────────────────────────────────

lime = pytest.importorskip("lime", reason="lime not installed — run: pip install autoexplainml[xai]")

from autoexplainml.explainers.lime_explainer import explain_lime, LimeExplainer


def test_lime_returns_dict(trained_rf):
    model, X = trained_rf
    result = explain_lime(model, X, num_samples=2)
    assert isinstance(result, dict)


def test_lime_has_method_key(trained_rf):
    model, X = trained_rf
    result = explain_lime(model, X, num_samples=2)
    assert result.get("method") == "lime"


def test_lime_has_features_key(trained_rf):
    model, X = trained_rf
    result = explain_lime(model, X, num_samples=2)
    assert "features" in result


def test_lime_has_importance_key(trained_rf):
    model, X = trained_rf
    result = explain_lime(model, X, num_samples=2)
    assert "importance" in result


def test_lime_feature_count_matches_input(trained_rf):
    model, X = trained_rf
    result = explain_lime(model, X, num_samples=2)
    assert len(result["features"]) == X.shape[1]