# core/auto_detect.py

def detect_task(model, X):

    name = str(type(model)).lower()

    if "torch" in name or "keras" in name:
        return "dl"

    if len(getattr(X, "shape", [])) == 4:
        return "cv"

    return "tabular"