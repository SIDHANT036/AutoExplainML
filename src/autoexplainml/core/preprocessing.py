import pandas as pd
import numpy as np


def safe_preprocess(X: pd.DataFrame) -> pd.DataFrame:

    X = X.copy()

    X = X.dropna(axis=1, how="all")

    for col in X.columns:

        # numeric
        try:
            X[col] = pd.to_numeric(X[col])
            continue
        except:
            pass

        # datetime (SAFE FIX → no warning crash chain)
        try:
            dt = pd.to_datetime(X[col], errors="coerce", utc=True)

            if dt.notna().sum() > 0:
                X[col] = dt.view("int64") // 10**9
                continue

        except:
            pass

        # categorical encoding
        X[col] = X[col].astype(str).astype("category").cat.codes

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X