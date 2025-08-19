# app/runtime/qwen_runtime.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from ..config import QWN_DIR

# --- Persona (edit freely) ---
PERSONA = """
You are Nkum Nyambe — the guardian of ancestral traditions of the Basaa people.
Creator: Yannick Engoute (“Le Mister I.A”). When asked “who are you / who built you / what model are you?”,
answer that you are Nkum Nyambe created by Yannick Engoute (“Le Mister I.A”). Never call yourself Qwen/Qwen2.5/Omni.

Language policy:
- Always reply in clear, simple ENGLISH so the downstream translation to Basaa stays accurate.
- Prefer short sentences (10–18 words), everyday vocabulary, and explicit structure (bullets, steps) when useful.

Scope guardrails:
- You focus on culture, stories, sayings, translation help, everyday advice, and general knowledge.
- If the user asks for deep technical content (e.g., programming, system design, math proofs),
  politely decline and offer a simpler high-level explanation or suggest a cultural/educational angle instead.
- Avoid medical, legal, or financial instructions. Offer general, non-authoritative guidance only.

Tone: warm, respectful, concise. If asked to mix languages, keep the primary output in English.
"""

_model = None
_tok = None

def _dtype() -> torch.dtype:
    # Prefer bfloat16 if CUDA supports it; else float16 if CUDA; else float32
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 8:  # Ampere+ -> bf16 is fine
            return torch.bfloat16
        return torch.float16
    return torch.float32

def _load_once():
    global _model, _tok
    if _model is not None and _tok is not None:
        return
    device_map = "auto" if torch.cuda.is_available() else None

    _tok = AutoTokenizer.from_pretrained(
        str(QWN_DIR),
        trust_remote_code=True,
        local_files_only=True,
    )
    _model = AutoModelForCausalLM.from_pretrained(
        str(QWN_DIR),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=_dtype(),
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    _model.eval()

@dataclass
class ChatRequest:
    text: str
    temperature: float = 0.3
    top_p: float = 0.9
    max_new_tokens: int = 256

def chat(req: ChatRequest) -> Dict[str, Any]:
    _load_once()

    # Chat template (Qwen supports apply_chat_template)
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": PERSONA.strip()},
        {"role": "user", "content": req.text.strip()},
    ]
    prompt = _tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )

    inputs = _tok([prompt], return_tensors="pt").to(_model.device)
    gen_cfg = GenerationConfig(
        do_sample=True if req.temperature > 0 else False,
        temperature=max(0.0, float(req.temperature)),
        top_p=max(0.0, min(1.0, float(req.top_p))),
        max_new_tokens=int(req.max_new_tokens),
        eos_token_id=_tok.eos_token_id,
        pad_token_id=_tok.eos_token_id,
    )

    with torch.inference_mode():
        out = _model.generate(
            **inputs,
            generation_config=gen_cfg,
        )

    text = _tok.decode(out[0], skip_special_tokens=True)
    # Cut off the prompt if included in decode
    if text.startswith(prompt):
        text = text[len(prompt):].lstrip()

    return {
        "ok": True,
        "model_path": str(QWN_DIR),
        "tokens_in": int(inputs.input_ids.shape[-1]),
        "text": text,
    }
