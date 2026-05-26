import numpy as np
import pandas as pd


def compute_fairness(X, y_pred=None):

    if "region" not in X.columns:
        return {
            "statistical_parity_difference": None,
            "disparate_impact_ratio": None
        }

    groups = X["region"].astype(str)
    unique = groups.unique()

    if len(unique) < 2:
        return {"status": "not_applicable"}

    base = groups.value_counts(normalize=True)

    spd = base.max() - base.min()
    dir_ratio = base.min() / (base.max() + 1e-9)

    return {
        "statistical_parity_difference": float(spd),
        "disparate_impact_ratio": float(dir_ratio),
        "group_distribution": base.to_dict()
    }