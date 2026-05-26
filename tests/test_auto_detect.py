import pandas as pd
import numpy as np
import pytest
from autoexplainml.core.auto_detect import detect_task
from sklearn.ensemble import RandomForestClassifier


# ── tabular detection ──────────────────────────────────────────────────────────

def test_detect_tabular_with_dataframe(trained_rf):
    model, X = trained_rf
    assert detect_task(model, X) == "tabular"


def test_detect_tabular_with_numpy(trained_rf):
    model, _ = trained_rf
    X = np.random.rand(10, 4)
    assert detect_task(model, X) == "tabular"


def test_detect_tabular_with_logistic_regression(trained_lr):
    model, X = trained_lr
    assert detect_task(model, X) == "tabular"


def test_detect_tabular_with_decision_tree(trained_dt):
    model, X = trained_dt
    assert detect_task(model, X) == "tabular"


# ── cv detection ───────────────────────────────────────────────────────────────

def test_detect_cv_with_4d_input(trained_rf):
    model, _ = trained_rf
    X_4d = np.random.rand(8, 3, 32, 32)  # batch, channels, H, W
    assert detect_task(model, X_4d) == "cv"


# ── dl detection ───────────────────────────────────────────────────────────────

def test_detect_dl_with_torch_like_model_name():
    """detect_task returns 'dl' when the model's module contains 'torch'."""
    class FakeTorchModel:
        pass
    FakeTorchModel.__module__ = "torch.nn"

    X = pd.DataFrame({"a": [1, 2, 3]})
    assert detect_task(FakeTorchModel(), X) == "dl"


def test_detect_task_returns_string(trained_rf):
    model, X = trained_rf
    result = detect_task(model, X)
    assert isinstance(result, str)
    assert result in ("tabular", "dl", "cv")
