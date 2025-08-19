# Smaller GPU base: CUDA 12.1 + cuDNN runtime (Ubuntu 22.04)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install Python + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ca-certificates curl git && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    pip3 install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# App deps first (layer cache-friendly)
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    # Install CUDA-enabled PyTorch wheels (packed with CUDA 12.1)
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# App code
COPY app /app/app

EXPOSE 8000
CMD ["python","-m","uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--loop","asyncio","--http","h11","--workers","1"]

