from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import httpx
from app.core.config import settings

@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

@dataclass
class LLMResponse:
    text: str
    usage: LLMUsage
    latency_ms: int

class LLMClient:
    def __init__(self) -> None:
        if not settings.llm_api_key:
            raise RuntimeError("LLM_API_KEY is missing.")
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
             temperature: Optional[float] = None, max_tokens: Optional[int] = None,
             json_mode: Optional[bool] = None, timeout_s: Optional[int] = None) -> LLMResponse:
        m = model or settings.llm_model
        t = settings.llm_temperature if temperature is None else float(temperature)
        mx = settings.llm_max_tokens if max_tokens is None else int(max_tokens)
        jm = settings.llm_use_json_mode if json_mode is None else bool(json_mode)
        to = settings.llm_timeout_s if timeout_s is None else int(timeout_s)

        payload: Dict[str, Any] = {"model": m, "messages": messages, "temperature": t, "max_tokens": mx}
        if jm:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/chat/completions"
        retries = max(0, int(settings.llm_max_retries))
        last_err: Optional[str] = None

        for attempt in range(retries + 1):
            start = time.time()
            try:
                with httpx.Client(timeout=to) as client:
                    r = client.post(url, headers=self._headers(), json=payload)
                latency_ms = int((time.time() - start) * 1000)
                if r.status_code >= 400:
                    last_err = f"LLM error {r.status_code}: {r.text[:1200]}"
                    if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                        time.sleep(min(8.0, 0.8 * (2 ** attempt)))
                        continue
                    raise RuntimeError(last_err)
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                u = LLMUsage(prompt_tokens=int(usage.get("prompt_tokens") or 0),
                             completion_tokens=int(usage.get("completion_tokens") or 0))
                return LLMResponse(text=text, usage=u, latency_ms=latency_ms)
            except Exception as e:
                last_err = str(e)
                if attempt < retries:
                    time.sleep(min(8.0, 0.8 * (2 ** attempt)))
                    continue
                raise RuntimeError(last_err)
        raise RuntimeError(last_err or "Unknown LLM failure.")
