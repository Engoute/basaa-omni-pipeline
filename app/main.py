# app/main.py
from __future__ import annotations

import os
import threading
import time
from typing import Dict, Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import (
    APP_VERSION, HF_DATASET, M2M_ZIP, WSP_ZIP, ORP_ZIP, QWN_ZIP, QWN_REPO
)
from .util.download_plan import plan as make_plan
from .util.downloader import ensure_core_models

# -------- Optional helpers: Qwen bootstrap (download to /workspace) --------
HAVE_QWEN = True
try:
    from .util.qwen_bootstrap import plan_qwen, ensure_qwen
except Exception:
    HAVE_QWEN = False

# -------- Optional helpers: Qwen ping (tokenizer/config checks) ------------
HAVE_QWEN_PING = True
try:
    from .util.qwen_ping import ping_qwen as _ping_qwen
except Exception:
    HAVE_QWEN_PING = False

# -------- Optional runtime: local Qwen text chat (our thin wrapper) --------
HAVE_QWEN_CHAT = True
try:
    from .runtime.qwen_runtime import chat as _qwen_chat, ChatRequest as _ChatReq
except Exception:
    HAVE_QWEN_CHAT = False

app = FastAPI(title="Basaa Omni Pipeline")

# ===================== Background jobs (avoid CF 524) ======================
_jobs: Dict[str, Dict[str, Any]] = {}

def _start_job(name: str, target):
    """
    Fire-and-forget background job runner.
    Stores status/result in `_jobs[name]`.
    """
    if name in _jobs and _jobs[name].get("status") in {"running", "done"}:
        return _jobs[name]

    _jobs[name] = {"status": "running", "started": time.time()}

    def _run():
        try:
            res = target()
            _jobs[name].update({"status": "done", "result": res, "finished": time.time()})
        except Exception as e:
            _jobs[name].update({"status": "error", "error": str(e), "finished": time.time()})

    threading.Thread(target=_run, daemon=True).start()
    return _jobs[name]

def _get_job(name: str) -> Dict[str, Any]:
    return _jobs.get(name, {"status": "unknown"})

# ============================= Health / Config =============================
@app.get("/healthz")
def healthz():
    # also tell the client whether the on-demand HF fallback was warmed
    return {
        "ok": True,
        "service": "basaa-omni",
        "version": APP_VERSION,
        "qwen": HAVE_QWEN,
        "qwen_ping": HAVE_QWEN_PING,
        "qwen_chat": HAVE_QWEN_CHAT,
        "qwen_fallback_cached": bool(_QWEN_PIPE is not None),
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

# =============================== Bootstrap =================================
@app.get("/bootstrap/plan")
def bootstrap_plan():
    return make_plan()

@app.post("/bootstrap/download")
def bootstrap_download():
    # Synchronous — may time out via Cloudflare on large bundles.
    return ensure_core_models()

@app.post("/bootstrap/download_async")
def bootstrap_download_async():
    """
    Start core model download/extract in a background thread and
    return immediately with a job id to poll.
    """
    _start_job("core_bootstrap", ensure_core_models)
    return {"ok": True, "job": "core_bootstrap"}

@app.get("/bootstrap/status")
def bootstrap_status():
    """Poll the status/result of the async core bootstrap."""
    return _get_job("core_bootstrap")

# --------- Qwen bootstrap endpoints (download to /workspace) ---------------
if HAVE_QWEN:
    @app.get("/bootstrap/qwen/plan")
    def qwen_plan():
        return plan_qwen()

    @app.post("/bootstrap/qwen/download")
    def qwen_download():
        return ensure_qwen()

# -------------------------------- Ping -------------------------------------
if HAVE_QWEN_PING:
    @app.get("/qwen/ping")
    def qwen_ping():
        try:
            return _ping_qwen()
        except Exception as e:
            # Never 500 here; return debuggable JSON
            return JSONResponse(status_code=200, content={"ok": False, "error": str(e)})

# ============================= Qwen text chat ==============================
# Primary goal: English in → English out (simple text mode)

class QwenChatIn(BaseModel):
    text: str
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.9
    max_new_tokens: Optional[int] = 256

# ---- Lightweight HF fallback (trust_remote_code) for brand-new arches ----
_QWEN_PIPE = None  # type: ignore

def _lazy_qwen_text_pipe():
    """
    Create (once) a HF pipeline that can run text-gen with trust_remote_code=True.
    Prefers the local extracted model folder; falls back to repo id.
    """
    global _QWEN_PIPE
    if _QWEN_PIPE is not None:
        return _QWEN_PIPE

    from transformers import pipeline, __version__ as _tfver

    local_dir = "/workspace/models/qwen2_5_omni_7b"
    src = local_dir if os.path.isdir(local_dir) else (QWN_REPO or "Qwen/Qwen2.5-Omni-7B")

    # trust_remote_code lets us run even if transformers doesn’t ship the new class yet.
    _QWEN_PIPE = pipeline(
        task="text-generation",
        model=src,
        tokenizer=src,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
    )
    # stash some debug
    setattr(_QWEN_PIPE, "_src", src)
    setattr(_QWEN_PIPE, "_tfver", _tfver)
    return _QWEN_PIPE

@app.post("/chat/qwen")
def qwen_chat(body: QwenChatIn):
    # 1) Try our local runtime wrapper (fast path if available)
    if HAVE_QWEN_CHAT:
        try:
            req = _ChatReq(
                text=body.text,
                temperature=body.temperature or 0.3,
                top_p=body.top_p or 0.9,
                max_new_tokens=body.max_new_tokens or 256,
            )
            return _qwen_chat(req)
        except Exception as e:
            # fall through to HF pipeline fallback below
            err_primary = str(e)
    else:
        err_primary = "qwen_runtime not available"

    # 2) Fallback: HF pipeline with trust_remote_code (works for new model types)
    try:
        pipe = _lazy_qwen_text_pipe()
        do_sample = True if (body.temperature and body.temperature > 0) else False
        out = pipe(
            body.text,
            max_new_tokens=body.max_new_tokens or 256,
            temperature=body.temperature or 0.3,
            top_p=body.top_p or 0.9,
            do_sample=do_sample,
            return_full_text=False,  # only the completion
        )
        text = out[0]["generated_text"] if isinstance(out, list) and out else ""
        return {
            "ok": True,
            "backend": "hf_pipeline_fallback",
            "transformers": getattr(pipe, "_tfver", "unknown"),
            "model_src": getattr(pipe, "_src", "unknown"),
            "text": text,
        }
    except Exception as e2:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": "Both primary runtime and HF fallback failed.",
                "primary": err_primary,
                "fallback": str(e2),
            },
        )

# Optional: small info endpoint to see which backend will be used
@app.get("/chat/qwen/info")
def qwen_chat_info():
    info: Dict[str, Any] = {
        "ok": True,
        "have_qwen_runtime": HAVE_QWEN_CHAT,
        "fallback_cached": bool(_QWEN_PIPE is not None),
    }
    if _QWEN_PIPE is not None:
        info.update({
            "fallback_model_src": getattr(_QWEN_PIPE, "_src", "unknown"),
            "transformers": getattr(_QWEN_PIPE, "_tfver", "unknown"),
        })
    return info
