import pandas as pd
import plotly.express as px


def shap_bar_plot(explanation):

    features = explanation.get("features", [])
    importance = explanation.get("importance", [])

    if not features or not importance:
        return "<div>No SHAP data available</div>"

    flat = []
    for x in importance:
        if isinstance(x, (list, tuple, np.ndarray)):
            flat.append(float(sum(x) / len(x)))
        else:
            flat.append(float(x))

    df = pd.DataFrame({
        "feature": features,
        "importance": flat
    }).sort_values("importance", ascending=True)

    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title="SHAP Feature Importance"
    )

    # ✅ CRITICAL FIX: embed HTML correctly
    return fig.to_html(full_html=False, include_plotlyjs="cdn")