# api/app.py

from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from autoexplainml.core.explainer import explain

app = FastAPI(title="AutoExplainML")

@app.post("/explain")
async def explain_api(model_file: UploadFile = File(...),
                      data_file: UploadFile = File(...)):

    model = joblib.load(model_file.file)
    X = pd.read_csv(data_file.file)

    result = explain(model, X)

    return result