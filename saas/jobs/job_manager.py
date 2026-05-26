import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from autoexplainml.core.pipeline import run_pipeline

# =========================================================
# JOB STORAGE (DEV ONLY - WILL MOVE TO REDIS / POSTGRES)
# =========================================================
JOBS: Dict[str, Dict[str, Any]] = {}

# =========================================================
# CELERY INTEGRATION (SAFE IMPORT)
# =========================================================
try:
    from saas.jobs.job_tasks import run_explainability_job
    from celery.result import AsyncResult
    CELERY_ENABLED = True
except Exception:
    CELERY_ENABLED = False


# =========================================================
# CREATE JOB (MAIN ENTRY POINT)
# =========================================================
def create_job(model, X, model_bytes: Optional[bytes] = None, data_bytes: Optional[bytes] = None):
    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
        "mode": None,
        "task_id": None
    }

    # =====================================================
    # MODE 1: CELERY (PRIMARY / PRODUCTION)
    # =====================================================
    if CELERY_ENABLED:
        try:
            task = run_explainability_job.delay(job_id, model, X)
            JOBS[job_id]["task_id"] = task.id
            JOBS[job_id]["mode"] = "celery"
            return job_id
        except Exception as e:
            # fallback if celery fails
            JOBS[job_id]["error"] = f"Celery dispatch failed: {str(e)}"

    # =====================================================
    # MODE 2: THREAD FALLBACK (DEV SAFE)
    # =====================================================
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=2)
    executor.submit(_run_job, job_id, model, X)

    JOBS[job_id]["mode"] = "thread"
    return job_id


# =========================================================
# INTERNAL WORKER (THREAD MODE ONLY)
# =========================================================
def _run_job(job_id: str, model, X):
    try:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = datetime.utcnow().isoformat()

        result = run_pipeline(model, X)

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = result
        JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat()


# =========================================================
# GET JOB STATUS
# =========================================================
def get_job(job_id: str):
    job = JOBS.get(job_id)

    if not job:
        return {
            "status": "not_found",
            "message": "Invalid job ID"
        }

    # =====================================================
    # CELERY MODE STATUS SYNC
    # =====================================================
    if job.get("mode") == "celery" and job.get("task_id"):
        try:
            result = AsyncResult(job["task_id"])

            if result.state == "PENDING":
                job["status"] = "queued"

            elif result.state == "STARTED":
                job["status"] = "running"
                if not job["started_at"]:
                    job["started_at"] = datetime.utcnow().isoformat()

            elif result.state == "SUCCESS":
                job["status"] = "completed"
                job["result"] = result.result
                job["finished_at"] = datetime.utcnow().isoformat()

            elif result.state == "FAILURE":
                job["status"] = "failed"
                job["error"] = str(result.info)
                job["finished_at"] = datetime.utcnow().isoformat()

        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)

    return job