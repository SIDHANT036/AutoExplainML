# intelligence/data_quality.py

import numpy as np

def check_data_quality(X):

    report = {}

    report["missing_values"] = int(np.isnan(X).sum())
    report["shape"] = X.shape

    return report