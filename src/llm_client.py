from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
import httpx
from tenacity import retry, wait_exponential_jitter, stop_after_attempt

@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout_s: int

class OpenAICompatClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            timeout=httpx.Timeout(cfg.timeout_s),
            headers={"Authorization": f"Bearer {cfg.api_key}"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @retry(wait=wait_exponential_jitter(initial=1, max=20), stop=stop_after_attempt(8))
    async def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        r = await self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
