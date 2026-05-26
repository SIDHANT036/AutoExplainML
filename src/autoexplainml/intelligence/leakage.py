import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def detect_leakage(X, y):
    """
    Modern leakage detection (production version)
    """

    if y is None:
        return {
            "correlation_leakage": [],
            "mutual_info_leakage": []
        }

    df = X.copy()
    df = df.select_dtypes(include=[np.number])

    # correlation leakage
    corr = df.corrwith(pd.Series(y)).abs()
    corr_leakage = corr[corr > 0.9].index.tolist()

    # mutual information leakage
    try:
        mi = mutual_info_classif(df.fillna(0), y)
        threshold = np.percentile(mi, 95)
        mi_leakage = df.columns[mi >= threshold].tolist()
    except Exception:
        mi_leakage = []

    return {
        "correlation_leakage": corr_leakage,
        "mutual_info_leakage": mi_leakage
    }


# =========================================================
# BACKWARD COMPATIBILITY FIX (🔥 IMPORTANT)
# =========================================================
def leakage_check(X, y=None):
    """
    Old API support for project_mode.py
    """
    return detect_leakage(X, y)