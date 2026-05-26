import shap
import pandas as pd
import matplotlib.pyplot as plt


def shap_waterfall(model, X):

    if isinstance(X, pd.DataFrame):
        X = X.iloc[[0]]

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    shap.plots.waterfall(shap_values[0], show=False)

    plt.tight_layout()
    return plt.gcf()