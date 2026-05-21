# explainers/base.py

from abc import ABC, abstractmethod

class BaseExplainer(ABC):

    @abstractmethod
    def explain(self, model, X):
        pass