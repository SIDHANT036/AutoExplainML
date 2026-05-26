from autoexplainml.core.engine import explain
from autoexplainml.intelligence.data_quality import check_data_quality
from autoexplainml.intelligence.fairness import compute_fairness
from autoexplainml.intelligence.leakage import detect_leakage


def run_pipeline(model, X, y=None):

    explanation = explain(model, X)
    dq = check_data_quality(X)

    fairness = compute_fairness(X, y)
    leakage = detect_leakage(X, y)

    return {
        "explanation": explanation,
        "data_quality": dq,
        "fairness": fairness,
        "leakage": leakage
    }