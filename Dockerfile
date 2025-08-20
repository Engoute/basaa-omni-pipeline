# Smaller GPU base: CUDA 12.1 + cuDNN runtime (Ubuntu 22.04)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System Python + tools (no build-essentials needed for these deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv ca-certificates curl git && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# App deps (single, deterministic step)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt && \
    # Prefer pure-Python uvicorn (no native loops)
    python -m pip uninstall -y uvloop httptools || true && \
    # Sanity: show key packages; fail early if missing
    python -m pip show uvicorn transformers torch || (echo "DEP_MISSING" && exit 1)

# App code
COPY app /app/app

EXPOSE 8000

# Pure-Python runtime (asyncio + h11), single worker
CMD ["python","-m","uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--loop","asyncio","--http","h11","--workers","1"]
