import numpy as np

def explain(model, X):
    import shap

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    importance = np.abs(shap_values.values).mean(axis=0)

    top_idx = np.argsort(importance)[-3:][::-1]

    result = "Top Feature Importance:\n"

    for i in top_idx:
        result += f"- {X.columns[i]}: {importance[i]:.2f}\n"

    return result, shap_values