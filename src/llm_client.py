from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    label: str | None
    content: str
    usage: dict[str, int]
    raw_response: dict[str, Any]
    parsed_payload: dict[str, Any] | None
    parse_mode: str
    model: str


def parse_payload_from_text(text: str) -> tuple[dict[str, Any] | None, str]:
    stripped = text.strip()
    if not stripped:
        return None, "empty"

    for candidate in [stripped, *re.findall(r"\{.*?\}", stripped, flags=re.DOTALL)]:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload, "json"

    match = re.search(r"(\{.*\"label\"\s*:\s*\"[abc]\".*\})", stripped, flags=re.IGNORECASE)
    if match:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload, "json_like"

    return None, "unparsed"


def parse_label_from_text(text: str) -> tuple[str | None, str]:
    stripped = text.strip()
    payload, mode = parse_payload_from_text(text)
    if payload is not None:
        label = str(payload.get("label", "")).strip().lower()
        if label in {"a", "b", "c"}:
            return label, mode

    match = re.search(r'"label"\s*:\s*"([abc])"', stripped, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower(), "json_like"

    match = re.search(
        r"(?:final answer|answer label|answer|label)\s*[:=]\s*([abc])\b",
        stripped,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).lower(), "tagged"

    lines = [line.strip().lower() for line in stripped.splitlines() if line.strip()]
    if lines and lines[0] in {"a", "b", "c"}:
        return lines[0], "single_token"
    return None, "unparsed"


class ChatAPIClient:
    def __init__(self, backend_config: dict) -> None:
        self.provider = backend_config["provider"]
        self.api_key_env = backend_config["api_key_env"]
        self.api_key = os.environ.get(self.api_key_env)
        if not self.api_key:
            raise RuntimeError(f"Missing required API key env var: {self.api_key_env}")

        self.model = backend_config["model"]
        self.fallback_models = backend_config.get("fallback_models", [])
        self.timeout_seconds = backend_config.get("timeout_seconds", 60)
        self.max_tokens = backend_config.get("max_tokens", 200)
        self.temperature = backend_config.get("temperature")
        self.base_url = backend_config.get("base_url") or self._default_base_url()

    def _default_base_url(self) -> str:
        if self.provider == "openai":
            return "https://api.openai.com/v1"
        if self.provider == "deepseek":
            return "https://api.deepseek.com"
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _payload(self, model: str, messages: list[dict]) -> dict:
        payload = {
            "model": model,
            "messages": messages,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.provider == "openai":
            payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = self.max_tokens
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def complete(self, messages: list[dict]) -> LLMResponse:
        models = [self.model, *self.fallback_models]
        last_error: Exception | None = None
        for model in models:
            try:
                response = self._complete_once(model, messages)
                response.model = model
                return response
            except Exception as exc:
                last_error = exc
        if last_error is None:
            raise RuntimeError("No models configured for API completion.")
        raise last_error

    def _complete_once(self, model: str, messages: list[dict]) -> LLMResponse:
        request = urllib.request.Request(
            url=self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(self._payload(model, messages)).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider} HTTP {exc.code}: {body}") from exc

        content = self._extract_content(payload)
        parsed_payload, structured_mode = parse_payload_from_text(content)
        label, parse_mode = parse_label_from_text(content)
        if parse_mode == "unparsed" and structured_mode != "unparsed":
            parse_mode = structured_mode
        usage = payload.get("usage", {}) or {}
        normalized_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        return LLMResponse(
            label=label,
            content=content,
            usage=normalized_usage,
            raw_response=payload,
            parsed_payload=parsed_payload,
            parse_mode=parse_mode,
            model=model,
        )

    @staticmethod
    def _extract_content(payload: dict) -> str:
        choices = payload.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                    texts.append(item.get("text", ""))
            return "\n".join(texts)
        return str(content)
