import pandas as pd


def check_data_quality(X):

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X = X.copy()

    return {
        "shape": X.shape,
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_rows": int(X.duplicated().sum()),
        "columns": list(X.columns)
    }