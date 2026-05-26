import shap
import numpy as np
import pandas as pd
from .base import BaseExplainer


class SHAPExplainer(BaseExplainer):
    """
    SHAP-based global feature importance explainer
    """

    def explain(self, model, X):
        """
        Generate SHAP explanations for a trained model.

        Parameters:
        model: trained ML model
        X: pd.DataFrame input features

        Returns:
        dict: feature importance based on SHAP values
        """

        # Ensure input is DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Create SHAP explainer
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # Compute global feature importance
        importance = np.abs(shap_values.values).mean(axis=0)

        return {
            "method": "shap",
            "features": list(X.columns),
            "importance": importance.tolist()
        }


# =========================================================
# BACKWARD COMPATIBILITY WRAPPER (IMPORTANT FIX)
# =========================================================
def explain_shap(model, X):
    """
    Functional wrapper so old engine imports still work.
    """
    return SHAPExplainer().explain(model, X)