import numpy as np

def explain(model, X):
    import shap

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    importance = np.abs(shap_values.values).mean(axis=0)

    top_idx = np.argsort(importance)[-3:][::-1]
    features = [(X.columns[i], importance[i]) for i in top_idx]

    text = "Top Features:\n"
    for f, val in features:
        text += f"- {f}: {val:.2f}\n"

    return text, shap_values