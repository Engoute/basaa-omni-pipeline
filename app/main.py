from fastapi import FastAPI
from .config import (
    APP_VERSION, HF_DATASET, M2M_ZIP, WSP_ZIP, ORP_ZIP, QWN_ZIP, QWN_REPO
)
from .util.download_plan import plan as make_plan
from .util.downloader import ensure_core_models

# Try to import Qwen helpers, but don't crash if not present
HAVE_QWEN = True
try:
    from .util.qwen_bootstrap import plan_qwen, ensure_qwen
except Exception:
    HAVE_QWEN = False

HAVE_QWEN_PING = True
try:
    from .util.qwen_ping import ping_qwen as _ping_qwen
except Exception:
    HAVE_QWEN_PING = False

app = FastAPI(title="Basaa Omni Pipeline")

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "basaa-omni", "version": APP_VERSION, "qwen": HAVE_QWEN, "qwen_ping": HAVE_QWEN_PING}

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
        "qwen_repo": QWN_REPO,
    }

@app.get("/bootstrap/plan")
def bootstrap_plan():
    return make_plan()

@app.post("/bootstrap/download")
def bootstrap_download():
    return ensure_core_models()

# Qwen endpoints
if HAVE_QWEN:
    @app.get("/bootstrap/qwen/plan")
    def qwen_plan():
        return plan_qwen()

    @app.post("/bootstrap/qwen/download")
    def qwen_download():
        return ensure_qwen()

if HAVE_QWEN_PING:
    @app.get("/qwen/ping")
    def qwen_ping():
        return _ping_qwen()
