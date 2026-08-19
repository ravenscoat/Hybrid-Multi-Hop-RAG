from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ChatResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_eval_duration_ns: int = 0
    eval_duration_ns: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def from_ollama(cls, value: dict[str, Any]) -> "ChatResponse":
        return cls(
            content=str(value.get("message", {}).get("content", "")),
            prompt_tokens=int(value.get("prompt_eval_count") or 0),
            completion_tokens=int(value.get("eval_count") or 0),
            total_duration_ns=int(value.get("total_duration") or 0),
            load_duration_ns=int(value.get("load_duration") or 0),
            prompt_eval_duration_ns=int(value.get("prompt_eval_duration") or 0),
            eval_duration_ns=int(value.get("eval_duration") or 0),
        )


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict, timeout: int = 300) -> dict:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return self._post("/api/embed", {"model": model, "input": texts})["embeddings"]

    def chat(self, model: str, system: str, user: str, json_mode: bool = False) -> str:
        return self.chat_with_metadata(model, system, user, json_mode).content

    def chat_with_metadata(
        self, model: str, system: str, user: str, json_mode: bool = False
    ) -> ChatResponse:
        payload: dict = {
            "model": model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0},
        }
        if json_mode:
            payload["format"] = "json"
        return ChatResponse.from_ollama(self._post("/api/chat", payload))
