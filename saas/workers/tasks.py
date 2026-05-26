from saas.workers.celery_app import celery
from autoexplainml.core.pipeline import run_pipeline

import pandas as pd
import joblib
import io


@celery.task(bind=True)
def run_analysis_task(self, model_bytes, csv_bytes):

    # load model
    model = joblib.load(io.BytesIO(model_bytes))

    # load dataset
    X = pd.read_csv(io.BytesIO(csv_bytes))

    result = run_pipeline(model, X)

    return result