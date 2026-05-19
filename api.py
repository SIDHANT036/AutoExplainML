from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from autoexplainml.explainer import explain

app = FastAPI()

@app.get("/")
def home():
    return {"status": "AutoExplainML running"}

@app.post("/explain/")
async def explain_api(model_file: UploadFile = File(...), data_file: UploadFile = File(...)):
    model = joblib.load(model_file.file)
    data = pd.read_csv(data_file.file)

    result, _ = explain(model, data)

    return {"result": result}