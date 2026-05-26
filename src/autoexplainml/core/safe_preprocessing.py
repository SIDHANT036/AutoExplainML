# src/autoexplainml/core/safe_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def safe_preprocess(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # 1. force dataframe
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    # 2. remove empty columns
    X = X.dropna(axis=1, how="all")

    # 3. clean infinities
    X = X.replace([np.inf, -np.inf], np.nan)

    # 4. fill missing
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("missing")
        else:
            X[col] = X[col].fillna(0)

    # 5. encode ALL object columns safely
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype(str)
            X[col] = LabelEncoder().fit_transform(X[col])

    # 6. force numeric conversion (CRITICAL FIX)
    X = X.apply(pd.to_numeric, errors="coerce")

    # 7. final cleanup
    X = X.fillna(0)

    # 8. force correct dtype
    X = X.astype(np.float64)

    return X