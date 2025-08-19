from pathlib import Path
import os

# Persistent base (on RunPod this is your Volume Disk mount)
PERSIST_DIR = Path(os.getenv("PERSIST_DIR", "/workspace")).resolve()

# Your HF dataset (zipped bundles for m2m, whisper, orpheus)
HF_DATASET  = os.getenv("HF_DATASET", "LeMisterIA/basaa-models")
M2M_ZIP     = os.getenv("M2M_ZIP", "bundles/m2m100_bundle_20250817_155147.zip")
WSP_ZIP     = os.getenv("WSP_ZIP", "bundles/whisper_bundle_20250817_163830.zip")
ORP_ZIP     = os.getenv("ORP_ZIP", "bundles/orpheus_bundle_20250817_155147.zip")
QWN_ZIP     = os.getenv("QWN_ZIP", "")  # unused now; we pull Qwen by repo

# Qwen repo (local load)
QWN_REPO    = os.getenv("QWN_REPO", "Qwen/Qwen2.5-Omni-7B")

# Derived paths
BUNDLES_DIR = PERSIST_DIR / "bundles"
MODELS_DIR  = PERSIST_DIR / "models"
CACHE_DIR   = PERSIST_DIR / "cache"
QWN_DIR     = MODELS_DIR / "qwen2_5_omni_7b"

APP_VERSION = "0.2.0"
