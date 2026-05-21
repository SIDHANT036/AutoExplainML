# pipelines/explain_pipeline.py

from autoexplainml.core.explainer import explain

def run_full_pipeline(model, X, task="tabular"):

    result = explain(model, X, task=task)

    return {
        "status": "success",
        "task": task,
        "explanation": result
    }