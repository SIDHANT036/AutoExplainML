import plotly.express as px
import pandas as pd


def shap_bar_chart(features, importance):

    df = pd.DataFrame({
        "feature": features,
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