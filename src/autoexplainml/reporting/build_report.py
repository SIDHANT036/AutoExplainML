import numpy as np
import plotly.graph_objects as go


def build_report(result, data_quality, fairness, correlation=None, leakage=None):

    explanation = result.get("explanation", {})

    features = explanation.get("features", [])
    importance = explanation.get("importance", [])

    # -------------------------
    # FIX: SAFE FLATTEN
    # -------------------------
    importance = np.array(importance).flatten()

    top_k = min(10, len(features))

    top_features = features[:top_k]
    top_importance = importance[:top_k]

    # -------------------------
    # SHAP BAR CHART (FIX EMPTY PLOT)
    # -------------------------
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=top_features,
        y=top_importance
    ))
    fig_bar.update_layout(title="SHAP Feature Importance")

    # -------------------------
    # WATERFALL (ONLY 1 SAMPLE - FIX HUGE PDF)
    # -------------------------
    fig_waterfall = go.Figure(go.Waterfall(
        name="prediction",
        orientation="v",
        measure=["relative"] * len(top_features),
        x=top_features,
        y=top_importance
    ))

    fig_waterfall.update_layout(title="SHAP Waterfall (Sample)")

    return {
        "summary": "AutoExplainML Production Report",

        "explanation": {
            "features": features,
            "importance": importance.tolist()
        },

        "data_quality": data_quality,

        "fairness": fairness or {},

        "correlation": correlation or {},

        "leakage": leakage or {},

        "plots": {
            "shap_bar": fig_bar.to_json(),
            "waterfall": fig_waterfall.to_json()
        }
    }