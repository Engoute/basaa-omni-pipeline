from pathlib import Path
from typing import Dict, Any, List
import traceback
import torch
from transformers import AutoConfig, AutoTokenizer
from ..config import QWN_DIR

def _dir_size_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())

def _list_files(root: Path, limit: int = 60) -> List[str]:
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files.append(str(p.relative_to(root)))
            if len(files) >= limit:
                break
    return files

def ping_qwen() -> Dict[str, Any]:
    info: Dict[str, Any] = {"ok": False, "qwen_dir": str(QWN_DIR)}
    if not QWN_DIR.exists():
        info["error"] = f"Qwen dir missing: {QWN_DIR}"
        return info

    info["size_gb"] = round(_dir_size_bytes(QWN_DIR) / 1e9, 2)
    info["some_files"] = _list_files(QWN_DIR, 60)
    info["shards"] = sorted(p.name for p in QWN_DIR.glob("model-*.safetensors"))

    try:
        cfg = AutoConfig.from_pretrained(str(QWN_DIR), local_files_only=True, trust_remote_code=True)
        info["config_model_type"] = getattr(cfg, "model_type", None)
    except Exception as e:
        info["config_error"] = f"{type(e).__name__}: {e}"
        info["config_trace"] = traceback.format_exc(limit=2)

    try:
        tok = AutoTokenizer.from_pretrained(str(QWN_DIR), local_files_only=True, trust_remote_code=True)
        info["tokenizer_class"] = tok.__class__.__name__
        info["tokenizer_vocab_size"] = getattr(tok, "vocab_size", None)
        info["probe_tokens"] = tok.encode("hello", add_special_tokens=False)[:8]
        info["ok"] = True
        info["cuda"] = torch.cuda.is_available()
        info["device"] = (torch.cuda.get_device_name(0) if info["cuda"] else "cpu")
        return info
    except Exception as e_tok:
        info["tokenizer_error"] = f"{type(e_tok).__name__}: {e_tok}"
        info["tokenizer_trace"] = traceback.format_exc(limit=2)

    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(str(QWN_DIR), local_files_only=True, trust_remote_code=True)
        info["processor_class"] = proc.__class__.__name__
        info["ok"] = True
        info["cuda"] = torch.cuda.is_available()
        info["device"] = (torch.cuda.get_device_name(0) if info["cuda"] else "cpu")
        return info
    except Exception as e_proc:
        info["processor_error"] = f"{type(e_proc).__name__}: {e_proc}"
        info["processor_trace"] = traceback.format_exc(limit=2)
        return info
