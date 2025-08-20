# Smaller GPU base: CUDA 12.1 + cuDNN runtime (Ubuntu 22.04)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# System Python & tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ca-certificates curl git && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# App deps (includes uvicorn[standard]; we remove native bits to keep it pure-Python)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt && \
    python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 && \
    # keep pure asyncio/h11 (avoid uvloop/httptools edge-cases in some GPU bases)
    pip uninstall -y uvloop httptools || true

# App code
COPY app /app/app

EXPOSE 8000

# Pure-Python server flags (asyncio + h11)
CMD ["python","-m","uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--loop","asyncio","--http","h11","--workers","1"]
