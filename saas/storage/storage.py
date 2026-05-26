import os
import json

BASE_DIR = "saas_storage"

os.makedirs(BASE_DIR, exist_ok=True)

def save_report(job_id, data):
    path = f"{BASE_DIR}/{job_id}.json"
    with open(path, "w") as f:
        json.dump(data, f)

    return path