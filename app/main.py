from fastapi import FastAPI
from .config import APP_VERSION, HF_DATASET, M2M_ZIP, WSP_ZIP, ORP_ZIP, QWN_ZIP
from .util.download_plan import plan as make_plan
from .util.downloader import ensure_core_models

app = FastAPI(title="Basaa Omni Pipeline")

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "basaa-omni", "version": APP_VERSION}

@app.get("/configz")
def configz():
    return {
        "hf_dataset": HF_DATASET,
        "zips": {
            "m2m": M2M_ZIP,
            "whisper": WSP_ZIP,
            "orpheus": ORP_ZIP,
            "qwen": QWN_ZIP or "(pull by model id later)",
        },
    }

@app.get("/bootstrap/plan")
def bootstrap_plan():
    return make_plan()

@app.post("/bootstrap/download")
def bootstrap_download():
    """Pull zips from your HF dataset and extract into /workspace/models/* (idempotent)."""
    return ensure_core_models()
