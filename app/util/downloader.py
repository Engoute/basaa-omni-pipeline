import os, zipfile, time
from pathlib import Path
from huggingface_hub import hf_hub_download
from ..config import (
    HF_DATASET, M2M_ZIP, WSP_ZIP, ORP_ZIP, QWN_ZIP,
    PERSIST_DIR, BUNDLES_DIR, MODELS_DIR
)

def _now(): return time.strftime("%Y-%m-%d %H:%M:%S")

def _ensure_dirs():
    for d in (PERSIST_DIR, BUNDLES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

def _download_zip_from_dataset(filename: str) -> Path:
    # Downloads to /workspace/bundles/<name>.zip
    local = hf_hub_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        filename=filename,
        local_dir=BUNDLES_DIR,
        local_dir_use_symlinks=False,
        token=os.getenv("HF_TOKEN") or None,   # optional
    )
    return Path(local)

def _extract_once(zip_path: Path, dst: Path):
    if dst.exists() and any(dst.iterdir()):
        return {"action": "skip_extract", "dst": str(dst)}
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dst)
    return {"action": "extracted", "dst": str(dst)}

def _ensure_one(name: str, zip_rel: str, extract_to: Path):
    out = {"name": name, "zip_rel": zip_rel, "extract_to": str(extract_to), "ts": _now()}
    if extract_to.exists() and any(extract_to.iterdir()):
        out["status"] = "ready"
        return out

    local_zip = (BUNDLES_DIR / Path(zip_rel).name)
    if not local_zip.exists():
        z = _download_zip_from_dataset(zip_rel)
        local_zip = Path(z)

    out["zip_local"] = str(local_zip)
    res = _extract_once(local_zip, extract_to)
    out.update(res)
    out["status"] = "ready"
    return out

def ensure_core_models():
    _ensure_dirs()
    reports = []
    reports.append(_ensure_one("m2m", M2M_ZIP, MODELS_DIR / "m2m100_1p2b_basaa"))
    reports.append(_ensure_one("whisper", WSP_ZIP, MODELS_DIR / "whisper_large_v3_ct2"))
    reports.append(_ensure_one("orpheus", ORP_ZIP, MODELS_DIR / "orpheus_3b_basaa"))
    # Qwen will be handled later
    return {"ts": _now(), "dataset": HF_DATASET, "reports": reports}
