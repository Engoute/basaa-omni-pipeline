from pathlib import Path
import os

# Where persistent stuff will live later (on RunPod this will be /workspace)
PERSIST_DIR = Path(os.getenv("PERSIST_DIR", "/workspace")).resolve()

# Your HF dataset & zip filenames (from your links)
HF_DATASET = os.getenv("HF_DATASET", "LeMisterIA/basaa-models")
M2M_ZIP = os.getenv("M2M_ZIP", "bundles/m2m100_bundle_20250817_155147.zip")
WSP_ZIP = os.getenv("WSP_ZIP", "bundles/whisper_bundle_20250817_163830.zip")
ORP_ZIP = os.getenv("ORP_ZIP", "bundles/orpheus_bundle_20250817_155147.zip")
QWN_ZIP = os.getenv("QWN_ZIP", "")  # keep empty for now; we’ll pull Qwen by model id later

# Derived future paths (not used yet, just placeholders)
BUNDLES_DIR = PERSIST_DIR / "bundles"
MODELS_DIR  = PERSIST_DIR / "models"
CACHE_DIR   = PERSIST_DIR / "cache"

APP_VERSION = "0.1.0"
