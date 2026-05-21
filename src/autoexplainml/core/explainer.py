# core/explainer.py

def explain(model, X, mode="auto", task="tabular"):
    
    if task == "tabular":
        from autoexplainml.classical_ml.tabular import explain_tabular
        return explain_tabular(model, X)

    if task == "deep_learning":
        from autoexplainml.deep_learning.dl_explainer import explain_dl
        return explain_dl(model, X)

    if task == "cv":
        from autoexplainml.computer_vision.gradcam import explain_image
        return explain_image(model, X)

    raise ValueError("Unsupported task type")