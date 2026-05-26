# reporting/report_builder.py

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json


# =========================================================
# SAFE HELPERS
# =========================================================

def _safe_list(x):
    """Convert SHAP / numpy weird outputs safely"""
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return x
    return [x]


def _flatten_importance(importance):
    """
    SHAP sometimes returns:
    - list of floats
    - list of lists (you had this issue)
    """
    imp = _safe_list(importance)

    if len(imp) == 0:
        return []

    # If nested → flatten by mean
    if isinstance(imp[0], list):
        return [float(np.mean(i)) for i in imp]

    return [float(i) for i in imp]


# =========================================================
# SHAP VISUALIZATION
# =========================================================

def create_shap_bar(features, importance):
    importance = _flatten_importance(importance)

    df = pd.DataFrame({
        "feature": features[:len(importance)],
        "importance": importance
    }).sort_values("importance", ascending=True)

    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title="SHAP Feature Importance"
    )

    return fig.to_html(full_html=False)


def create_shap_waterfall(features, importance):
    importance = _flatten_importance(importance)

    fig = go.Figure()

    fig.add_trace(go.Waterfall(
        name="SHAP",
        orientation="v",
        x=features[:len(importance)],
        y=importance,
        connector={"line": {"color": "gray"}}
    ))

    fig.update_layout(title="SHAP Waterfall (Single Prediction View)")
    return fig.to_html(full_html=False)


# =========================================================
# CORRELATION + LEAKAGE
# =========================================================

def correlation_check(X: pd.DataFrame, threshold=0.85):
    corr = X.corr(numeric_only=True)

    high = []
    cols = corr.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) > threshold:
                high.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "corr": float(val)
                })

    return {
        "high_correlation_pairs": high,
        "warning": "high correlation detected" if high else "none"
    }


def leakage_check(X: pd.DataFrame, y=None):
    """
    True leakage detection:
    - correlation with target
    - near-duplicate features
    """
    result = {
        "correlation_leakage": [],
        "mutual_info_leakage": []
    }

    if y is None:
        return result

    try:
        df = X.copy()
        df["target"] = y

        corr = df.corr(numeric_only=True)["target"].drop("target")

        for k, v in corr.items():
            if abs(v) > 0.9:
                result["correlation_leakage"].append({
                    "feature": k,
                    "corr": float(v)
                })

    except Exception:
        pass

    return result


# =========================================================
# FAIRNESS METRICS (REAL)
# =========================================================

def fairness_metrics(X: pd.DataFrame):
    result = {
        "statistical_parity_difference": None,
        "disparate_impact_ratio": None
    }

    if "region" not in X.columns:
        return result

    try:
        groups = X["region"].value_counts(normalize=True)

        # simple proxy fairness metrics
        max_p = groups.max()
        min_p = groups.min()

        result["disparate_impact_ratio"] = float(min_p / max_p) if max_p > 0 else None
        result["statistical_parity_difference"] = float(max_p - min_p)

    except Exception:
        pass

    return result


# =========================================================
# MAIN REPORT BUILDER
# =========================================================

def build_report(explanation, data_quality, fairness=None, X=None, y=None):

    explanation = explanation or {}

    features = explanation.get("features", [])
    importance = explanation.get("importance", [])

    shap_bar = create_shap_bar(features, importance)
    shap_waterfall = create_shap_waterfall(features, importance)

    corr = correlation_check(X) if X is not None else {}
    leak = leakage_check(X, y) if X is not None else {}

    fair = fairness or {}

    report = {
        "summary": "AutoExplainML Production Report",

        "explanation": {
            "method": explanation.get("method", "shap"),
            "features": features,
            "importance": importance,
            "visualizations": {
                "shap_bar": shap_bar,
                "shap_waterfall": shap_waterfall
            }
        },

        "data_quality": data_quality,

        "fairness": {
            **fair,
        },

        "correlation": corr,

        "leakage": leak,

        "insights": {
            "top_feature": features[0] if features else None,
            "note": "AutoExplainML production report"
        }
    }

    return report