from autoexplainml.core.auto_detect import detect_task
from autoexplainml.explainers.shap_explainer import explain_shap
from autoexplainml.reporting.build_report import build_report


def explain(model, X):

    task = detect_task(model, X)

    # ---------------------------
    # SAFE COPY (prevents dtype(O) crash)
    # ---------------------------
    import pandas as pd
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X = X.copy()
    X = X.select_dtypes(include=["number"])  # CRITICAL FIX

    # ---------------------------
    # SHAP (LIMIT SAMPLE → FIX FREEZE)
    # ---------------------------
    X_sample = X.sample(min(300, len(X)), random_state=42)

    if task == "tabular":
        print("🧠 Running SHAP (safe mode)...")
        result = explain_shap(model, X_sample)

    elif task == "dl":
        from autoexplainml.explainers.dl_explainer import explain_dl
        result = explain_dl(model, X_sample)

    elif task == "cv":
        from autoexplainml.explainers.cv_gradcam import explain_cv
        result = explain_cv(model, X_sample)

    else:
        raise ValueError("Unsupported model type")

    return result