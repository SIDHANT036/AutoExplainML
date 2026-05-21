# core/validator.py

def validate(model, X):

    if model is None:
        raise ValueError("Model is required")

    if X is None or len(X) == 0:
        raise ValueError("Dataset is empty")

    return True