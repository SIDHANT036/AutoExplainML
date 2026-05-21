# explainers/shap_explainer.py

import shap
import numpy as np
from .base import BaseExplainer

class SHAPExplainer(BaseExplainer):

    def explain(self, model, X):

        explainer = shap.Explainer(model, X)
        values = explainer(X)

        importance = np.abs(values.values).mean(axis=0)

        return {
            "method": "shap",
            "features": list(X.columns),
            "importance": importance.tolist()
        }