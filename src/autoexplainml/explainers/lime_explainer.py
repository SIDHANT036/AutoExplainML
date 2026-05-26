import numpy as np
import pandas as pd
from .base import BaseExplainer


class LimeExplainer(BaseExplainer):
    """
    LIME-based local feature importance explainer.
    Explains individual predictions using local linear approximations.
    """

    def explain(self, model, X, num_samples=5):
        """
        Generate LIME explanations for a trained model.

        Parameters:
            model: trained sklearn-compatible model
            X: pd.DataFrame of input features
            num_samples: number of rows to explain (default 5, capped at len(X))

        Returns:
            dict: per-feature importance averaged over sampled rows
        """
        try:
            import lime
            import lime.lime_tabular
        except ImportError:
            raise ImportError(
                "lime is required for LimeExplainer. "
                "Install it with: pip install autoexplainml[xai]"
            )

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        feature_names = list(X.columns)
        X_values = X.values

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_values,
            feature_names=feature_names,
            mode="classification" if hasattr(model, "predict_proba") else "regression",
            verbose=False,
        )

        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict

        n = min(num_samples, len(X))
        all_importances = np.zeros(len(feature_names))

        for i in range(n):
            exp = explainer.explain_instance(
                X_values[i],
                predict_fn,
                num_features=len(feature_names)
            )
            for feat, weight in exp.as_list():
                # feat is a string like "petal length (cm) <= 2.45"
                # match it back to the original feature name
                for j, name in enumerate(feature_names):
                    if name in feat:
                        all_importances[j] += abs(weight)
                        break

        avg_importances = (all_importances / n).tolist()

        return {
            "method": "lime",
            "features": feature_names,
            "importance": avg_importances
        }


def explain_lime(model, X, num_samples=5):
    """
    Functional wrapper for LimeExplainer.
    """
    return LimeExplainer().explain(model, X, num_samples=num_samples)