def validate(model, X):
    if model is None:
        raise ValueError("Model is required")
    if isinstance(model, str):
        raise ValueError(f"Model must be a trained ML object, not a string: '{model}'")
    if X is None or len(X) == 0:
        raise ValueError("Dataset is empty")
    return True
