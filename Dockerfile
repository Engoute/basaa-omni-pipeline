# GPU-ready base (CUDA 12.8, PyTorch 2.8, Python 3.11)
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# app deps
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# app code
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
