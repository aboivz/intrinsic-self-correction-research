from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol

from .models import DecodingConfig, GenerationResult, Message, ModelIdentity, Usage


class GenerationBackend(Protocol):
    identity: ModelIdentity

    def generate(self, messages: list[Message], decoding: DecodingConfig, request_id: str) -> GenerationResult: ...


class OllamaBackend:
    def __init__(self, model_id: str, identity: ModelIdentity, base_url: str = "http://127.0.0.1:11434"):
        self.model_id, self.identity, self.base_url = model_id, identity, base_url.rstrip("/")

    def generate(self, messages: list[Message], decoding: DecodingConfig, request_id: str) -> GenerationResult:
        payload = {
            "model": self.model_id,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": 5,
            "options": decoding.model_dump(),
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "X-Request-ID": request_id},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            return GenerationResult(error=f"HTTP {exc.code}: {exc.reason}", transient=exc.code >= 500)
        except (urllib.error.URLError, TimeoutError) as exc:
            return GenerationResult(error=str(exc), transient=True)
        usage = Usage(
            input_tokens=body.get("prompt_eval_count"), output_tokens=body.get("eval_count"),
            total_duration_ns=body.get("total_duration"), load_duration_ns=body.get("load_duration"),
            eval_duration_ns=body.get("eval_duration"),
        )
        return GenerationResult(raw_output=body.get("message", {}).get("content"), usage=usage, metadata={"request": payload, "response": body})
