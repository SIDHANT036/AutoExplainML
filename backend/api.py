from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from autoexplainml.explainer import explain

app = FastAPI(title="AutoExplainML API")

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/explain/")
async def explain_model(
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...)
):
    model = joblib.load(model_file.file)
    data = pd.read_csv(data_file.file)

    result, _ = explain(model, data)

    return {"result": result}