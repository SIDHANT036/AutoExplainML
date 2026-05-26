def detect_task(model, X):
    cls = type(model)
    name = cls.__name__.lower()
    module = (cls.__module__ or "").lower()
    if "torch" in name or "torch" in module or "keras" in name or "keras" in module:
        return "dl"
    if len(getattr(X, "shape", [])) == 4:
        return "cv"
    return "tabular"
