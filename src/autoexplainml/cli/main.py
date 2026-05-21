# cli/main.py

import argparse
import joblib
import pandas as pd
from autoexplainml.core.explainer import explain

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("data")

    args = parser.parse_args()

    model = joblib.load(args.model)
    X = pd.read_csv(args.data)

    result = explain(model, X)

    print(result)