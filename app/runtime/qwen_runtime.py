# app/runtime/qwen_runtime.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoConfig,
)

# Where /bootstrap/qwen/download placed the model
QWEN_PATH = os.environ.get("QWEN_PATH", "/workspace/models/qwen2_5_omni_7b")

_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None
_processor: Optional[AutoProcessor] = None


def _load_once() -> None:
    """Lazy-load tokenizer/model/processor with trust_remote_code."""
    global _tokenizer, _model, _processor
    if _model is not None and _tokenizer is not None:
        return

    # Some Qwen repos ship custom modeling code; this flag is required.
    # Using device_map="auto" so it lands on GPU if present.
    cfg = AutoConfig.from_pretrained(QWEN_PATH, trust_remote_code=True)

    _tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True, use_fast=True)

    try:
        # For multimodal (images/video/audio). Text-only path works without it,
        # but we keep it ready for later.
        _processor = AutoProcessor.from_pretrained(QWEN_PATH, trust_remote_code=True)
    except Exception:
        _processor = None  # fine for pure text

    _model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH,
        torch_dtype=_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()


class ChatRequest(BaseModel):
    text: str
    temperature: float = 0.3
    top_p: float = 0.9
    max_new_tokens: int = 256


def _render_chat(messages: list[dict[str, str]]) -> Dict[str, Any]:
    """
    Use the model's chat template if present; otherwise fall back to a simple
    prompt format. Returns dict with 'input_ids' on the model's device.
    """
    assert _tokenizer is not None and _model is not None

    try:
        input_ids = _tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(_model.device)
        return {"input_ids": input_ids}
    except Exception:
        # Fallback: minimal prompt
        text = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            text += f"{role}: {content}\n"
        text += "assistant: "
        input_ids = _tokenizer(text, return_tensors="pt").input_ids.to(_model.device)
        return {"input_ids": input_ids}


def chat(req: ChatRequest) -> Dict[str, Any]:
    _load_once()
    assert _tokenizer is not None and _model is not None

    messages = [{"role": "user", "content": req.text}]
    inputs = _render_chat(messages)

    gen = _model.generate(
        **inputs,
        do_sample=True,
        temperature=float(req.temperature),
        top_p=float(req.top_p),
        max_new_tokens=int(req.max_new_tokens),
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.pad_token_id,
    )

    # Decode and strip the prompt using the tokenizer's helper if available.
    try:
        text_out = _tokenizer.decode(gen[0], skip_special_tokens=True)
        # If apply_chat_template was used, the response usually appears after the last "assistant" turn.
        split_tok = "assistant"
        if split_tok in text_out:
            text_out = text_out.split(split_tok, maxsplit=1)[-1].strip(" :\n")
    except Exception:
        text_out = _tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

    return {"ok": True, "text": text_out}


def info() -> Dict[str, Any]:
    """Small introspection endpoint to help debugging in /chat/qwen/info."""
    import transformers
    return {
        "ok": True,
        "device": _device,
        "dtype": str(_dtype),
        "transformers": transformers.__version__,
        "path": QWEN_PATH,
        "tokenizer_loaded": _tokenizer is not None,
        "model_loaded": _model is not None,
        "processor_loaded": _processor is not None,
    }
