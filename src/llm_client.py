from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class LocalHFConfig:
    model_id: str
    dtype: str = "float16"          # "float16" or "bfloat16"
    device_map: str = "auto"        # "auto" is best on Runpod
    max_new_tokens: int = 256
    temperature: float = 0.0        # 0.0 => greedy
    top_p: float = 1.0
    do_sample: bool = False         # set True only if temperature>0


def _parse_dtype(s: str):
    s = (s or "").lower().strip()
    if s in ["float16", "fp16"]:
        return torch.float16
    if s in ["bfloat16", "bf16"]:
        return torch.bfloat16
    if s in ["float32", "fp32"]:
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


class LocalTransformersClient:
    """
    Drop-in replacement for your OpenAI-compatible client.
    Exposes: chat(messages) -> str
    """

    def __init__(self, cfg: LocalHFConfig):
        self.cfg = cfg
        dtype = _parse_dtype(cfg.dtype)

        # HF cache envs (optional, but helps stability)
        # If you exported these in shell, this is just a no-op.
        os.environ.setdefault("HF_HOME", "/workspace/.hf_cache")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/.hf_cache")
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/.hf_cache")

        self.tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            device_map=cfg.device_map,
        )
        self.model.eval()

    async def aclose(self) -> None:
        # no async resources to close
        return

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: [{"role": "...", "content": "..."}]
        returns: generated string
        """
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(text, return_tensors="pt")
        # move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=int(self.cfg.max_new_tokens),
            do_sample=bool(self.cfg.do_sample),
        )
        # only pass sampling args if sampling is enabled
        if self.cfg.do_sample:
            gen_kwargs["temperature"] = float(self.cfg.temperature)
            gen_kwargs["top_p"] = float(self.cfg.top_p)

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        # decode ONLY the newly generated part
        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = out[0, prompt_len:]
        return self.tok.decode(gen_tokens, skip_special_tokens=True).strip()
