from saas.queue.celery_app import celery_app
from autoexplainml.core.pipeline import run_pipeline
import joblib
import pandas as pd

@celery_app.task
def run_ml_job(model_bytes, csv_bytes):
    model = joblib.load(model_bytes)
    X = pd.read_csv(csv_bytes)

    result = run_pipeline(model, X)

    return result