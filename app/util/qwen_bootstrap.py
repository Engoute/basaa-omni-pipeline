import os
from pathlib import Path
from huggingface_hub import snapshot_download
from ..config import QWN_REPO, QWN_DIR

def plan_qwen():
    return {
        "repo": QWN_REPO,
        "extract_to": str(QWN_DIR),
    }

def ensure_qwen():
    QWN_DIR.mkdir(parents=True, exist_ok=True)
    # already present?
    if any(QWN_DIR.iterdir()):
        return {"status": "ready", "action": "skip_extract", "path": str(QWN_DIR), "repo": QWN_REPO}

    snapshot_download(
        repo_id=QWN_REPO,
        local_dir=str(QWN_DIR),
        local_dir_use_symlinks=False,
        max_workers=4,
        token=os.getenv("HF_TOKEN") or None,   # set in pod env if the repo is gated
    )
    return {"status": "ready", "action": "extracted", "path": str(QWN_DIR), "repo": QWN_REPO}
