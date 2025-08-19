from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import (
    APP_VERSION, HF_DATASET, M2M_ZIP, WSP_ZIP, ORP_ZIP, QWN_ZIP, QWN_REPO
)
from .util.download_plan import plan as make_plan
from .util.downloader import ensure_core_models

# --- Optional helpers: Qwen bootstrap (download to /workspace) ---
HAVE_QWEN = True
try:
    from .util.qwen_bootstrap import plan_qwen, ensure_qwen
except Exception:
    HAVE_QWEN = False

# --- Optional helpers: Qwen ping (tokenizer/config checks) ---
HAVE_QWEN_PING = True
try:
    from .util.qwen_ping import ping_qwen as _ping_qwen
except Exception:
    HAVE_QWEN_PING = False

# --- Optional runtime: local Qwen text chat ---
HAVE_QWEN_CHAT = True
try:
    from .runtime.qwen_runtime import chat as _qwen_chat, ChatRequest as _ChatReq
except Exception:
    HAVE_QWEN_CHAT = False

app = FastAPI(title="Basaa Omni Pipeline")

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "service": "basaa-omni",
        "version": APP_VERSION,
        "qwen": HAVE_QWEN,
        "qwen_ping": HAVE_QWEN_PING,
        "qwen_chat": HAVE_QWEN_CHAT,
    }

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

# --- Qwen bootstrap endpoints ---
if HAVE_QWEN:
    @app.get("/bootstrap/qwen/plan")
    def qwen_plan():
        return plan_qwen()

    @app.post("/bootstrap/qwen/download")
    def qwen_download():
        return ensure_qwen()

# --- Qwen ping (never 500; returns debug JSON on error) ---
if HAVE_QWEN_PING:
    @app.get("/qwen/ping")
    def qwen_ping():
        try:
            return _ping_qwen()
        except Exception as e:
            return JSONResponse(status_code=200, content={"ok": False, "error": str(e)})

# ---------- Qwen text chat: English in → English out ----------
class QwenChatIn(BaseModel):
    text: str
    temperature: float | None = 0.3
    top_p: float | None = 0.9
    max_new_tokens: int | None = 256

@app.post("/chat/qwen")
def qwen_chat(body: QwenChatIn):
    if not HAVE_QWEN_CHAT:
        return JSONResponse(status_code=503, content={"ok": False, "error": "Qwen runtime not available"})
    try:
        req = _ChatReq(
            text=body.text,
            temperature=body.temperature or 0.3,
            top_p=body.top_p or 0.9,
            max_new_tokens=body.max_new_tokens or 256,
        )
        return _qwen_chat(req)
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
