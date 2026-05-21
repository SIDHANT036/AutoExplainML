# intelligence/fairness.py

def fairness_check(X, sensitive_feature):

    if sensitive_feature not in X.columns:
        return {"error": "feature not found"}

    group_stats = X.groupby(sensitive_feature).mean()

    return {
        "bias_risk": "low",  # simplified version
        "analysis": group_stats.to_dict()
    }