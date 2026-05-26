import uuid
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from autoexplainml.core.pipeline import run_pipeline

# =========================
# CONFIG
# =========================
JOBS_FILE = "jobs.json"
JOBS = {}
executor = ThreadPoolExecutor(max_workers=2)


# =========================
# LOAD JOBS ON STARTUP
# =========================
def _load_jobs():
    global JOBS
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE, "r") as f:
                JOBS = json.load(f)
        except:
            JOBS = {}
    else:
        JOBS = {}


def _save_jobs():
    with open(JOBS_FILE, "w") as f:
        json.dump(JOBS, f, indent=2)


_load_jobs()


# =========================
# WORKER
# =========================
def _run_job(job_id: str, model, X):
    try:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = datetime.utcnow().isoformat()
        _save_jobs()

        result = run_pipeline(model, X)

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = result
        JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat()
        _save_jobs()

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat()
        _save_jobs()


# =========================
# CREATE JOB
# =========================
def create_job(model, X):
    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "result": None,
        "error": None
    }

    _save_jobs()
    executor.submit(_run_job, job_id, model, X)

    return job_id


# =========================
# GET JOB
# =========================
def get_job(job_id: str):
    return JOBS.get(job_id, {
        "status": "not_found",
        "message": "Invalid job ID"
    })