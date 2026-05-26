from saas.workers.celery_app import celery
from autoexplainml.core.pipeline import run_pipeline

JOBS = {}

@celery.task
def run_explainability_job(job_id, model, X):
    try:
        JOBS[job_id]["status"] = "running"

        result = run_pipeline(model, X)

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = result

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)