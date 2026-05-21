import argparse
import joblib
import pandas as pd
from autoexplainml.core.pipeline import run_pipeline

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("data")

    args = parser.parse_args()

    model = joblib.load(args.model)
    X = pd.read_csv(args.data)

    result = run_pipeline(model, X)

    print(result)

if __name__ == "__main__":
    main()