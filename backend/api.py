from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from autoexplainml.core.explainer import explain

app = FastAPI(title="AutoExplainML")

@app.get("/")
def home():
    return {"status": "AutoExplainML running"}

@app.post("/explain/")
async def explain_api(
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...)
):
    try:
        model = joblib.load(model_file.file)
        data = pd.read_csv(data_file.file)

        result, _ = explain(model, data)

        return {
            "status": "success",
            "explanation": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }