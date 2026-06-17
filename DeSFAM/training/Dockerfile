FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "tensorflow>=2.16" \
    "scikit-learn>=1.4" \
    "numpy>=1.26" \
    "scipy>=1.12" \
    "joblib>=1.3" \
    "matplotlib>=3.8" \
    "openpyxl>=3.1" \
    "prefixspan>=0.5.2" \
    "jupyter>=1.0" \
    "notebook>=7.0"

ENV PYTHONPATH=/workspace
ENV TF_CPP_MIN_LOG_LEVEL=2
