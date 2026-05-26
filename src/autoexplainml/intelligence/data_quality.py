# intelligence/data_quality.py
import numpy as np
import pandas as pd


def check_data_quality(X):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    report = {}

    # missing values — works for all column types
    report["missing_values"] = int(X.isnull().sum().sum())

    # shape
    report["shape"] = X.shape

    # duplicate rows
    report["duplicate_rows"] = int(X.duplicated().sum())

    # per-column missing count
    report["missing_per_column"] = X.isnull().sum().to_dict()

    # numeric columns only — basic stats
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    report["numeric_columns"] = len(numeric_cols)
    report["total_columns"] = len(X.columns)
    report["total_rows"] = len(X)

    return report
