import uuid
import json
import os
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

from autoexplainml.core.pipeline import run_pipeline

# =====================================================
# CONFIG
# =====================================================
JOBS_FILE = "jobs.json"
LOCK = threading.Lock()
executor = ThreadPoolExecutor(max_workers=2)

# =====================================================
# MEMORY STORE
# =====================================================
JOBS: Dict[str, Dict[str, Any]] = {}


# =====================================================
# SAFE LOAD (startup recovery)
# =====================================================
def _load_jobs():
    global JOBS
    if not os.path.exists(JOBS_FILE):
        JOBS = {}
        return

    try:
        with open(JOBS_FILE, "r") as f:
            JOBS = json.load(f)
    except Exception:
        JOBS = {}


# =====================================================
# SAFE SAVE (atomic write)
# =====================================================
def _save_jobs():
    tmp_file = JOBS_FILE + ".tmp"
    try:
        with open(tmp_file, "w") as f:
            json.dump(JOBS, f, default=str)

        os.replace(tmp_file, JOBS_FILE)  # atomic replace
    except Exception:
        # fail silently but DO NOT crash system
        pass


# load at import
_load_jobs()


# =====================================================
# SAFE JOB WORKER
# =====================================================
def _run_job(job_id: str, model, X):
    try:
        with LOCK:
            if job_id not in JOBS:
                return
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["started_at"] = datetime.utcnow().isoformat()
        _save_jobs()

        # ==============================
        # HARD SAFETY BOUNDARY
        # ==============================
        result = None
        try:
            result = run_pipeline(model, X)
        except Exception as e:
            result = {"pipeline_error": str(e)}

        with LOCK:
            if job_id not in JOBS:
                return
            JOBS[job_id]["status"] = "completed" if result else "failed"
            JOBS[job_id]["result"] = result
            JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat()

        _save_jobs()

    except Exception as e:
        with LOCK:
            if job_id in JOBS:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = str(e)
                JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat()
        _save_jobs()


# =====================================================
# CREATE JOB (SAFE INPUT SANITIZATION)
# =====================================================
def create_job(model, X):
    job_id = str(uuid.uuid4())

    if model is None or X is None:
        return None

    try:
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

    except Exception:
        return None


# =====================================================
# GET JOB (SAFE FALLBACK)
# =====================================================
def get_job(job_id: str) -> Dict[str, Any]:
    try:
        return JOBS.get(job_id, {
            "status": "not_found",
            "job_id": job_id,
            "message": "Invalid or expired job id"
        })
    except Exception:
        return {
            "status": "error",
            "message": "job system failure"
        }


# =====================================================
# OPTIONAL: CLEANUP OLD JOBS (PREVENT MEMORY LEAK)
# =====================================================
def cleanup_jobs(max_jobs: int = 1000):
    try:
        if len(JOBS) <= max_jobs:
            return

        sorted_jobs = sorted(
            JOBS.items(),
            key=lambda x: x[1].get("created_at", "")
        )

        excess = len(JOBS) - max_jobs

        for i in range(excess):
            JOBS.pop(sorted_jobs[i][0], None)

        _save_jobs()

    except Exception:
        pass