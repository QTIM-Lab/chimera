# Dockerfile for CHIMERA Task 2 – Clinical only (TabPFN)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies (build-essential for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default I/O and model locations for Grand Challenge
RUN mkdir -p /input /output /model && if [ -f /app/model/model.joblib ]; then cp /app/model/model.joblib /model/model.joblib; fi
ENV CHIMERA_CLINICAL_JSON=/input/chimera-clinical-data-of-bladder-cancer-patients.json
ENV CHIMERA_OUTPUT_JSON=/output/brs-probability.json
ENV CHIMERA_MODEL=/model/model.joblib

# Inference entrypoint
ENTRYPOINT ["python", "predict.py"]
