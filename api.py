from fastapi import FastAPI
import pandas as pd
import joblib
from autoexplainml.explainer import explain

app = FastAPI()

@app.get("/")
def home():
    return {"status": "AutoExplainML is live"}

@app.post("/explain/")
def explain_api():
    return {"message": "API is working (connect file upload later in frontend)"}