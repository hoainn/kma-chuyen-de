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
# Kernel syscall table — same one training used. Lets detect.py resolve every
# id (including the x32 ABI range 512–547) to its canonical name.
COPY syscall_64.tbl .
COPY detect.py .

ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=2

# Artifacts are mounted at runtime — not baked in
ENV ARTIFACTS_DIR=/artifacts

RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

CMD ["python", "detect.py"]
