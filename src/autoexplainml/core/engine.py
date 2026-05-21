# core/engine.py

from autoexplainml.core.auto_detect import detect_task
from autoexplainml.explainers.shap_explainer import explain_shap
from autoexplainml.reporting.report_builder import build_report

def explain(model, X):

    task = detect_task(model, X)

    if task == "tabular":
        result = explain_shap(model, X)

    elif task == "dl":
        from autoexplainml.explainers.dl_explainer import explain_dl
        result = explain_dl(model, X)

    elif task == "cv":
        from autoexplainml.explainers.cv_gradcam import explain_cv
        result = explain_cv(model, X)

    else:
        raise ValueError("Unsupported model type")

    return result