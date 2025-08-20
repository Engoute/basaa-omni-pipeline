# Smaller GPU base: CUDA 12.1 + cuDNN runtime (Ubuntu 22.04)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal OS + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ca-certificates curl git && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps
COPY requirements.txt /app/requirements.txt

# Install app deps (pure-Python uvicorn) + CUDA 12.1 torch
RUN set -eux; \
    python -m pip install --no-cache-dir -r /app/requirements.txt; \
    python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0; \
    python - <<'PY'\nimport uvicorn, transformers, torch\nprint('deps_ok', uvicorn.__version__, transformers.__version__, torch.__version__)\nPY

# ---- App code
COPY app /app/app

EXPOSE 8000

# Pure-Python server (asyncio + h11)
CMD ["python","-m","uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--loop","asyncio","--http","h11","--workers","1"]
