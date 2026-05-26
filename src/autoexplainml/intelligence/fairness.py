import numpy as np
import pandas as pd


def compute_fairness(X, y=None, sensitive_col="region"):
    """
    Modern fairness metrics:
    - Statistical Parity Difference
    - Disparate Impact Ratio
    """

    if y is None or sensitive_col not in X.columns:
        return {
            "statistical_parity_difference": None,
            "disparate_impact_ratio": None,
            "note": "fairness not computed"
        }

    df = pd.DataFrame({
        "group": X[sensitive_col],
        "y": y
    })

    rates = df.groupby("group")["y"].mean()

    if len(rates) < 2:
        return {
            "statistical_parity_difference": 0.0,
            "disparate_impact_ratio": 1.0
        }

    max_rate = rates.max()
    min_rate = rates.min()

    return {
        "statistical_parity_difference": float(max_rate - min_rate),
        "disparate_impact_ratio": float(min_rate / (max_rate + 1e-9))
    }


# =========================================================
# BACKWARD COMPATIBILITY FIX (🔥 REQUIRED)
# =========================================================
def fairness_check(X, y=None):
    """
    Old API support for project_mode.py
    """
    return compute_fairness(X, y)