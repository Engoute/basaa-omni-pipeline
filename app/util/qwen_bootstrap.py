# app/util/qwen_bootstrap.py
from __future__ import annotations
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
DEST = Path("/workspace/models/qwen2_5_omni_7b")

def plan_qwen():
    return {"repo": MODEL_ID, "extract_to": str(DEST)}

def _exists_any(globs: list[str]) -> bool:
    for g in globs:
        if list(DEST.glob(g)):
            return True
    return False

def ensure_qwen():
    """
    Make sure BOTH the weights AND Qwen's custom Python code (*.py) are present.
    If weights are already there, fetch only the code files to keep it light.
    """
    DEST.mkdir(parents=True, exist_ok=True)

    have_weights = (DEST / "model.safetensors.index.json").exists() or _exists_any(["model-*.safetensors"])
    have_config  = (DEST / "config.json").exists()
    have_code    = _exists_any([
        "modeling_qwen2_5_omni.py",
        "configuration_qwen2_5_omni.py",
        "tokenization_qwen2_5_omni.py",
        # sometimes projects nest python code under a subfolder; be broad:
        "**/*qwen2*_omni*.py",
    ])

    action = "skip_extract"

    # 1) If anything is missing, prefer to download the minimal missing set
    if not (have_weights and have_config):
        # pull everything (weights+code) one time into DEST
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(DEST),
            local_dir_use_symlinks=False,
            max_workers=4,
        )
        have_weights, have_config = True, True
        action = "extracted"

    # 2) If code is missing, fetch only .py files to keep it small
    if not have_code:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(DEST),
            local_dir_use_symlinks=False,
            allow_patterns=["*.py"],
            max_workers=4,
        )
        action = "extracted_code" if action == "skip_extract" else action

    return {
        "status": "ready",
        "action": action,
        "path": str(DEST),
        "repo": MODEL_ID,
        "have_weights": have_weights,
        "have_config": have_config,
        "have_code": _exists_any(["**/*.py"]),
    }
