FROM python:3.11-slim

# System deps for numpy/scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ unzip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888
