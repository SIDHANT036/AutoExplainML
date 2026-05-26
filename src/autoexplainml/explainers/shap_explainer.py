import shap
import numpy as np
import pandas as pd
from .base import BaseExplainer


class SHAPExplainer(BaseExplainer):

    def explain(self, model, X):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # KEEP ONLY NUMERIC (FIX dtype('O') ERROR)
        X = X.select_dtypes(include=["number"]).copy()

        # SMALL SAMPLE (CRITICAL PERFORMANCE FIX)
        X_sample = X.sample(min(200, len(X)), random_state=42)

        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

        # FIX: flatten safely
        importance = np.abs(shap_values.values).mean(axis=0)

        return {
            "method": "shap",
            "features": list(X_sample.columns),
            "importance": importance.tolist(),
            "shap_values": shap_values.values[:50].tolist()  # LIMIT SIZE
        }


def explain_shap(model, X):
    return SHAPExplainer().explain(model, X)