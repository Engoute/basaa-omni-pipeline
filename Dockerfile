# CUDA 12.1 + cuDNN runtime (Ubuntu 22.04)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1800 \
    PIP_PREFER_BINARY=1

# Minimal system deps + git (for other tools), curl, build-essential (safety for any wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ca-certificates curl git build-essential \
 && ln -sf /usr/bin/python3 /usr/local/bin/python \
 && python -m pip install --upgrade pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps (install transformers from main.zip; uvicorn w/o extras) ----
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt \
 && python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
 # sanity: make sure these 3 exist or fail the build early
 && python - <<'PY'
import uvicorn, transformers, torch
print("deps_ok", uvicorn.__version__, transformers.__version__, torch.__version__)
PY

# ---- App code ----
COPY app /app/app

EXPOSE 8000

# Pure-Python server flags (asyncio + h11). Single worker keeps GPU memory simpler.
CMD ["python","-m","uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--loop","asyncio","--http","h11","--workers","1"]
