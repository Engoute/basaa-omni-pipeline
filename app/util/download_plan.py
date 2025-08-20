from pathlib import Path
from ..config import (
    HF_DATASET, M2M_ZIP, WSP_ZIP, ORP_ZIP, QWN_ZIP,
    PERSIST_DIR, BUNDLES_DIR, MODELS_DIR, CACHE_DIR
)

def plan():
    m2m_dir = MODELS_DIR / "m2m100_1p2b_basaa"
    wsp_dir = MODELS_DIR / "whisper_large_v3_ct2"
    orp_dir = MODELS_DIR / "orpheus_3b_basaa"
    qwn_dir = MODELS_DIR / "qwen2_5_omni_7b"

    out = {
        "persist_dir": str(PERSIST_DIR),
        "bundles_dir": str(BUNDLES_DIR),
        "models_dir": str(MODELS_DIR),
        "cache_dir":   str(CACHE_DIR),
        "dataset": HF_DATASET,
        "artifacts": {
            "m2m": {
                "zip": M2M_ZIP,
                "zip_local": str(BUNDLES_DIR / Path(M2M_ZIP).name),
                "extract_to": str(m2m_dir),
            },
            "whisper": {
                "zip": WSP_ZIP,
                "zip_local": str(BUNDLES_DIR / Path(WSP_ZIP).name),
                "extract_to": str(wsp_dir),
            },
            "orpheus": {
                "zip": ORP_ZIP,
                "zip_local": str(BUNDLES_DIR / Path(ORP_ZIP).name),
                "extract_to": str(orp_dir),
            },
            "qwen": {
                "zip": QWN_ZIP or "(none; pull from model repo)",
                "zip_local": str(BUNDLES_DIR / Path(QWN_ZIP).name) if QWN_ZIP else "(n/a)",
                "extract_to": str(qwn_dir),
                "hf_model_fallback": "Qwen/Qwen2.5-Omni-7B",
            },
        }
    }
    return out
