FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/           src/
COPY tetragon_grpc/ tetragon_grpc/
COPY syscalls_x86_64.py .
COPY detect.py .

ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=2

# Artifacts are mounted at runtime — not baked in
ENV ARTIFACTS_DIR=/artifacts
ENV TARGET_NAMESPACE=demo
ENV KERNEL_VER=6.17
ENV TETRAGON_ADDR=10.200.118.136:30321

RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

CMD ["python", "detect.py"]
