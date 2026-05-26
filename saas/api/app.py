import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

import pandas as pd
import joblib

from saas.auth.auth import authenticate
from saas.auth.jwt_handler import create_access_token, verify_token
from saas.jobs.job_manager import create_job, get_job


# =========================================================
# APP INIT
# =========================================================
app = FastAPI(
    title="AutoExplainML SaaS",
    description="Explainable AI SaaS API (JWT + Async Jobs + Reports)",
    version="3.2.0"
)


# =========================================================
# STARTUP EVENT (ADDED - IMPORTANT FOR RENDER DEBUGGING)
# =========================================================
@app.on_event("startup")
def startup_event():
    print("AutoExplainML API started")


# =========================================================
# AUTH SETUP (JWT)
# =========================================================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


class LoginRequest(BaseModel):
    username: str
    password: str


# =========================================================
# JWT AUTH DEPENDENCY
# =========================================================
def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)

    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

    return payload


# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/")
def home():
    return {
        "service": "AutoExplainML SaaS",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================================================
# LOGIN → JWT TOKEN
# =========================================================
@app.post("/login")
def login(data: LoginRequest):

    if not authenticate(data.username, data.password):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    token = create_access_token({"sub": data.username})

    return {
        "access_token": token,
        "token_type": "bearer"
    }


# =========================================================
# INPUT VALIDATION
# =========================================================
def verify_inputs(model_file: UploadFile, data_file: UploadFile):

    if not model_file.filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Model must be .pkl file")

    if not data_file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Data must be .csv file")

    return True


# =========================================================
# RUN JOB (PROTECTED SaaS CORE)
# =========================================================
@app.post("/run-job")
async def run_job(
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...),
    user=Depends(get_current_user)
):

    # STEP 1: validate files
    verify_inputs(model_file, data_file)

    try:
        # STEP 2: read safely (IMPORTANT FIX: read bytes once)
        model_bytes = model_file.file.read()
        data_bytes = data_file.file.read()

        model = joblib.load(pd.io.common.BytesIO(model_bytes))
        X = pd.read_csv(pd.io.common.BytesIO(data_bytes))

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid model or dataset file"
        )

    if X.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    # STEP 3: create async job
    job_id = create_job(
        model=model,
        X=X,
        model_bytes=model_bytes,
        data_bytes=data_bytes
    )

    return {
        "status": "success",
        "job_id": job_id
    }


# =========================================================
# JOB STATUS (PROTECTED)
# =========================================================
@app.get("/job/{job_id}")
def job_status(
    job_id: str,
    user=Depends(get_current_user)
):
    return get_job(job_id)