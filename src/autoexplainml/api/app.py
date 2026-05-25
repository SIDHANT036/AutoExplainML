from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib

from autoexplainml.core.pipeline import run_pipeline
from autoexplainml.product.project_mode import run_full_project

app = FastAPI(title="AutoExplainML API", version="2.1.0")


# =========================
# LOW-LEVEL ML ANALYSIS API
# =========================
@app.post("/analyze")
async def analyze(
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...)
):

    # Load model
    model = joblib.load(model_file.file)

    # Load dataset
    X = pd.read_csv(data_file.file)

    # Run core pipeline (your engine)
    result = run_pipeline(model, X)

    return {
        "status": "success",
        "mode": "analyze",
        "result": result
    }


# =========================
# HIGH-LEVEL STUDENT MODE
# =========================
@app.post("/project-mode")
async def project_mode(
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...)
):

    # Load model
    model = joblib.load(model_file.file)

    # Load dataset
    X = pd.read_csv(data_file.file)

    # Run full automated student pipeline
    result = run_full_project(model, X)

    return {
        "status": "success",
        "mode": "project-mode",
        "result": result
    }