from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from autoexplainml.core.pipeline import run_pipeline

app = FastAPI()

@app.post("/analyze")
async def analyze(
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...)
):

    model = joblib.load(model_file.file)
    X = pd.read_csv(data_file.file)

    result = run_pipeline(model, X)

    return result