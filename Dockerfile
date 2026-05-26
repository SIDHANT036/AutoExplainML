FROM python:3.11

WORKDIR /app

# Install system deps (needed for shap sometimes)
RUN apt-get update && apt-get install -y build-essential

# Copy dependency file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# ✅ Install your package WITH extras
RUN pip install --no-cache-dir ".[xai]"

CMD ["uvicorn", "saas.api.app:app", "--host", "0.0.0.0", "--port", "8000"]