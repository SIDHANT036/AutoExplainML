import numpy as np
import pandas as pd
from .base import BaseExplainer


class PermutationExplainer(BaseExplainer):
    """
    Permutation-based feature importance explainer.
    Measures how much model performance drops when each feature is shuffled.
    """

    def explain(self, model, X):
        """
        Generate permutation feature importance for a trained model.

        Parameters:
            model: trained sklearn-compatible model (must have predict or predict_proba)
            X: pd.DataFrame of input features

        Returns:
            dict: feature importance based on permutation
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Use predict_proba if available, else predict
        if hasattr(model, "predict_proba"):
            baseline_preds = model.predict_proba(X)
        else:
            baseline_preds = model.predict(X).reshape(-1, 1)

        importances = []
        for col in X.columns:
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)

            if hasattr(model, "predict_proba"):
                permuted_preds = model.predict_proba(X_permuted)
            else:
                permuted_preds = model.predict(X_permuted).reshape(-1, 1)

            # Importance = mean absolute change in predictions
            importance = np.abs(baseline_preds - permuted_preds).mean()
            importances.append(float(importance))

        return {
            "method": "permutation",
            "features": list(X.columns),
            "importance": importances
        }


def explain_permutation(model, X):
    """
    Functional wrapper for PermutationExplainer.
    """
    return PermutationExplainer().explain(model, X)