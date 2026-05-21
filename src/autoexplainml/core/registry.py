# core/registry.py

from autoexplainml.explainers.shap_explainer import SHAPExplainer
from autoexplainml.explainers.permutation_explainer import PermutationExplainer

class ExplainerRegistry:

    def __init__(self):
        self.explainers = {
            "shap": SHAPExplainer(),
            "permutation": PermutationExplainer()
        }

    def get(self, name):
        return self.explainers.get(name)