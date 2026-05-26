import pandas as pd


def remove_duplicate_encoded_features(X: pd.DataFrame):

    cols = list(X.columns)
    to_drop = set()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):

            if X[cols[i]].equals(X[cols[j]]):
                to_drop.add(cols[j])

    return X.drop(columns=list(to_drop)), list(to_drop)