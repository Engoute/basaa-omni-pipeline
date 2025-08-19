# app/runtime/qwen_runtime.py
from __future__ import annotations
from dataclasses import dataclass
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Where /bootstrap/qwen/download placed the model
MODEL_DIR = os.getenv("QWEN_MODEL_DIR", "/workspace/models/qwen2_5_omni_7b")

_tokenizer = None
_model = None

def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            local_files_only=True,
        )
    if _model is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True,
        ).eval()
    return _tokenizer, _model

# Public request schema main.py imports
@dataclass
class ChatRequest:
    text: str
    temperature: float = 0.3
    top_p: float = 0.9
    max_new_tokens: int = 256

# Default persona (English inside pipeline; MAUI will translate to Basaa later)
_PERSONA = (
    "You are Nkum Nyambe, the guardian of ancestral traditions. "
    "You respectfully preserve, explain, and teach Basaa culture, proverbs, and customs. "
    "Be concise, warm, and factual. If a topic is extremely technical, say so and "
    "offer a simple non-technical explanation or a cultural perspective instead. "
    "Never claim you are Qwen; your creator is Yannick Engoute (Le Mister I.A). "
    "Always respond in English in this pipeline."
)

def chat(req: ChatRequest):
    tok, model = _load()

    messages = [
        {"role": "system", "content": _PERSONA},
        {"role": "user", "content": req.text},
    ]

    # Qwen 2.5 Omni exposes a chat template via tokenizer
    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    do_sample = (req.temperature or 0) > 0
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=int(req.max_new_tokens or 256),
            do_sample=do_sample,
            temperature=float(req.temperature or 0.3),
            top_p=float(req.top_p or 0.9),
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

    text = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
    return {
        "ok": True,
        "model_path": MODEL_DIR,
        "tokens_in": int(inputs.numel()),
        "text": text.strip(),
    }
