from __future__ import annotations
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from ..config import QWN_REPO, QWN_DIR

def plan_qwen():
    return {"repo": QWN_REPO, "extract_to": str(QWN_DIR)}

def _exists_any(globs: list[str]) -> bool:
    for g in globs:
        if list(QWN_DIR.glob(g)):
            return True
    return False

def ensure_qwen():
    """
    Ensure BOTH the weights and Qwen's custom Python code (*.py) exist locally.
    This is required for qwen2_5_omni; we use trust_remote_code to load.
    """
    QWN_DIR.mkdir(parents=True, exist_ok=True)

    have_weights = (QWN_DIR / "model.safetensors.index.json").exists() or _exists_any(["model-*.safetensors"])
    have_config  = (QWN_DIR / "config.json").exists()
    have_code    = _exists_any(["**/*qwen2*_omni*.py", "*.py"])

    action = "skip_extract"

    # If anything core is missing, grab full snapshot
    if not (have_weights and have_config):
        snapshot_download(
            repo_id=QWN_REPO,
            local_dir=str(QWN_DIR),
            local_dir_use_symlinks=False,
            max_workers=4,
            token=os.getenv("HF_TOKEN") or None,
        )
        have_weights, have_config = True, True
        action = "extracted"

    # If .py code absent, pull only *.py (smaller)
    if not have_code:
        snapshot_download(
            repo_id=QWN_REPO,
            local_dir=str(QWN_DIR),
            local_dir_use_symlinks=False,
            allow_patterns=["*.py"],
            max_workers=4,
            token=os.getenv("HF_TOKEN") or None,
        )
        action = "extracted_code" if action == "skip_extract" else action

    return {
        "status": "ready",
        "action": action,
        "path": str(QWN_DIR),
        "repo": QWN_REPO,
        "have_weights": have_weights,
        "have_config": have_config,
        "have_code": _exists_any(["**/*.py"]),
    }
