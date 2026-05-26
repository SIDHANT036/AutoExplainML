import shap
import numpy as np
import pandas as pd
from .base import BaseExplainer

class SHAPExplainer(BaseExplainer):
    def explain(self, model, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        raw = np.abs(shap_values.values)
        # Multi-class: shape is (n_samples, n_features, n_classes) — average over both
        if raw.ndim == 3:
            raw = raw.mean(axis=2)   # -> (n_samples, n_features)
        importance = np.nan_to_num(raw.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)

        return {"method": "shap", "features": list(X.columns), "importance": importance.tolist()}

def explain_shap(model, X):
    return SHAPExplainer().explain(model, X)
