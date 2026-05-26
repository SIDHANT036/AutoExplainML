from celery import Celery

celery = Celery(
    "autoexplainml",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/1",
    include=["saas.jobs.job_tasks"]
)

celery.conf.update(
    task_track_started=True,
)