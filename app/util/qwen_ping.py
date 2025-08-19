from pathlib import Path
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoConfig
from ..config import QWN_DIR

def _dir_size_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())

def ping_qwen() -> Dict[str, Any]:
    assert QWN_DIR.exists(), f"Qwen dir missing: {QWN_DIR}"

    # light-weight checks: tokenizer + config only (no model weights loaded)
    cfg = AutoConfig.from_pretrained(str(QWN_DIR), local_files_only=True, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(str(QWN_DIR), local_files_only=True, trust_remote_code=True)
    probe = tok.encode("hello", add_special_tokens=False)[:8]

    cuda = torch.cuda.is_available()
    device = (torch.cuda.get_device_name(0) if cuda else "cpu")

    shards = sorted(p.name for p in QWN_DIR.glob("model-*.safetensors"))

    return {
        "ok": True,
        "qwen_dir": str(QWN_DIR),
        "size_gb": round(_dir_size_bytes(QWN_DIR) / 1e9, 2),
        "shards": shards,
        "config_model_type": getattr(cfg, "model_type", None),
        "tokenizer_vocab_size": getattr(tok, "vocab_size", None),
        "probe_tokens": probe,
        "cuda": cuda,
        "device": device,
    }
