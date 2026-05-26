from .celery_app import celery
from autoexplainml.core.pipeline import run_pipeline

@celery.task
def run_ml_job(model, X):
    return run_pipeline(model, X)