from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Callable

import requests

from .config import AdjudicationConfig, default_adjudication_config

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """
    Abstract interface for LLM clients used in retake adjudication.
    """
    @abstractmethod
    def generate_json(self, prompt: str) -> dict[str, Any]:
        """
        Submits prompt to LLM and returns parsed JSON dict.
        """
        pass


class GeminiClient(LLMClient):
    """
    Direct HTTP client for Google Gemini API with JSON mode.
    """
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def generate_json(self, prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
            },
        }

        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout_seconds,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini API request failed with status {response.status_code}: {response.text}"
            )

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini API returned no candidates: {data}")

        part = candidates[0].get("content", {}).get("parts", [{}])[0]
        text = part.get("text", "").strip()

        # Parse JSON output
        return json.loads(text)


class OpenAIClient(LLMClient):
    """
    Direct HTTP client for OpenAI-compatible Chat Completions API with JSON mode.
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def generate_json(self, prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("OpenAI/Groq API key not found. Please set OPENAI_API_KEY or GROQ_API_KEY.")

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional video editor specializing in speech retake adjudication. Return valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI/Groq API request failed with status {response.status_code}: {response.text}"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"API returned no choices: {data}")

        content = choices[0].get("message", {}).get("content", "").strip()
        return json.loads(content)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def extract_usage(data: Any) -> dict[str, int | None]:
    """
    Token usage from a decoded API response, for OpenAI-format ("usage") and Gemini-format ("usageMetadata").
    Fields: prompt_tokens, completion_tokens (billable output INCLUDING reasoning/thinking), reasoning_tokens (informational
    subset of completion_tokens), total_tokens. A field the API did not return is None, never 0.
    """
    out: dict[str, int | None] = {"prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None, "total_tokens": None}
    if not isinstance(data, dict):
        return out
    usage = data.get("usage")
    if isinstance(usage, dict):
        out["prompt_tokens"] = _int_or_none(usage.get("prompt_tokens"))
        out["completion_tokens"] = _int_or_none(usage.get("completion_tokens"))
        details = usage.get("completion_tokens_details")
        out["reasoning_tokens"] = _int_or_none(details.get("reasoning_tokens")) if isinstance(details, dict) else None
        out["total_tokens"] = _int_or_none(usage.get("total_tokens"))
        return out
    meta = data.get("usageMetadata")
    if isinstance(meta, dict):
        out["prompt_tokens"] = _int_or_none(meta.get("promptTokenCount"))
        cand, thought = _int_or_none(meta.get("candidatesTokenCount")), _int_or_none(meta.get("thoughtsTokenCount"))
        # Gemini reports thinking tokens separately from candidates; billable output is their sum
        out["completion_tokens"] = None if cand is None and thought is None else (cand or 0) + (thought or 0)
        out["reasoning_tokens"] = thought
        out["total_tokens"] = _int_or_none(meta.get("totalTokenCount"))
    return out


def endpoint_label(url: str) -> str:
    """Name of the API endpoint a request URL actually targets (not the model name)."""
    u = str(url).lower()
    if "generativelanguage.googleapis.com" in u:
        return "gemini-generativelanguage"
    if "api.groq.com" in u:
        return "groq-openai-compat"
    if "api.openai.com" in u:
        return "openai"
    return "unknown"


class MockLLMClient(LLMClient):
    """
    Mock LLM client for deterministic testing, offline execution, and error simulation.
    """
    def __init__(
        self,
        responses: list[dict[str, Any] | Exception] | None = None,
        handler: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.responses = list(responses) if responses else []
        self.handler = handler
        self.call_count: int = 0
        self.prompts: list[str] = []

    def generate_json(self, prompt: str) -> dict[str, Any]:
        self.call_count += 1
        self.prompts.append(prompt)

        if self.handler is not None:
            return self.handler(prompt)

        if not self.responses:
            # Default fallback mock response: keep the LAST member unit of the span and remove the others (valid under the keeper-only contract)
            m = re.search(r"Member Unit IDs: (\[.*?\])", prompt)
            if m:
                try:
                    ids = [str(u) for u in json.loads(m.group(1))]
                    if ids:
                        return {"keeper_unit_id": ids[-1], "units_to_remove": ids[:-1], "confidence": "high",
                                "reasoning": "Mock fallback: keep the last unit."}
                except ValueError:
                    pass
            return {
                "is_genuine_retake_group": True,
                "keeper_unit_id": None,
                "units_to_remove": [],
                "confidence": "high",
                "reasoning": "Mock fallback adjudication response.",
            }

        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def get_llm_client(config: AdjudicationConfig | None = None) -> LLMClient:
    """
    Factory creating an appropriate LLMClient based on config and available environment keys.
    """
    cfg = config or default_adjudication_config
    provider = cfg.provider.lower()

    if provider == "mock":
        return MockLLMClient()

    if provider in {"gemini", "google"}:
        return GeminiClient(
            model=cfg.model,
            api_key=cfg.api_key,
            temperature=cfg.temperature,
            timeout_seconds=cfg.timeout_seconds,
        )

    if provider in {"openai", "groq"}:
        base_url = "https://api.groq.com/openai/v1" if provider == "groq" else "https://api.openai.com/v1"
        return OpenAIClient(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=base_url,
            temperature=cfg.temperature,
            timeout_seconds=cfg.timeout_seconds,
        )

    # Default fallback to Gemini
    return GeminiClient(
        model=cfg.model,
        api_key=cfg.api_key,
        temperature=cfg.temperature,
        timeout_seconds=cfg.timeout_seconds,
    )
