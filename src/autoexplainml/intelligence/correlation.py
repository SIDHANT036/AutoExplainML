import pandas as pd
import numpy as np


def correlation_check(X: pd.DataFrame):

    X = X.select_dtypes(include=[np.number])

    if X.shape[1] < 2:
        return {"status": "not_enough_numeric_features"}

    corr = X.corr().abs()

    high_corr = []

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] > 0.85:
                high_corr.append({
                    "feature_1": corr.columns[i],
                    "feature_2": corr.columns[j],
                    "correlation": float(corr.iloc[i, j])
                })

    return {
        "high_correlation_pairs": high_corr,
        "warning": "high correlation detected" if high_corr else "none"
    }