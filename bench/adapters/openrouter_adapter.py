"""OpenRouter adapter for Ko-AgentBench."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

from .base_adapter import BaseAdapter
from ..observability import observe


class OpenRouterAdapter(BaseAdapter):
    """Adapter that talks to OpenRouter's OpenAI-compatible chat completions API."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model_name: str, **config: Any) -> None:
        """Initialize OpenRouter adapter.

        Args:
            model_name: Model identifier. Accepts either the raw OpenRouter slug
                (e.g. ``"anthropic/claude-3.5-sonnet"``) or a value prefixed with
                ``"openrouter/"`` which will be stripped automatically.
            **config: Optional configuration overrides, including:
                - api_key: Explicit API key (default: ``OPENROUTER_API_KEY`` env)
                - base_url: Override API base URL
                - timeout: Request timeout in seconds (default: 60)
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum completion tokens (default: 1024)
                - referer / app_title: Optional headers recommended by OpenRouter
                - request_overrides: Dict merged into each request payload
        """
        super().__init__(model_name, **config)

        self.model_id = self._normalize_model_name(model_name)

        self.api_key = config.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is missing. "
                "Set the OPENROUTER_API_KEY environment variable or pass api_key=..."
            )

        base_url = config.get("base_url", self.DEFAULT_BASE_URL).rstrip("/")
        self.endpoint = f"{base_url}/chat/completions"

        self.timeout = config.get("timeout", 60)
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1024)

        # Optional metadata headers recommended by OpenRouter for rate limiting
        referer = (
            config.get("referer")
            or os.getenv("OPENROUTER_APP_URL")
            or os.getenv("OPENROUTER_REFERER")
        )
        app_title = config.get("app_title") or os.getenv("OPENROUTER_APP_TITLE")

        self._default_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if referer:
            self._default_headers["HTTP-Referer"] = referer
        if app_title:
            self._default_headers["X-Title"] = app_title

        # Allow callers to specify additional payload fields
        self.request_overrides: Dict[str, Any] = config.get("request_overrides", {})

        # Reuse a requests session for connection pooling
        self._session = requests.Session()

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """Normalize model name by stripping a leading 'openrouter/' prefix."""
        if model_name.lower().startswith("openrouter/"):
            return model_name.split("/", 1)[1]
        return model_name

    @observe(as_type="generation")
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a chat completion using the OpenRouter API."""
        payload = self.convert_to_provider_format(messages, tools)

        # Allow per-call overrides (e.g., n, stop) without mutating defaults
        extra_params = {k: v for k, v in kwargs.items() if v is not None}
        payload.update(extra_params)

        try:
            response = self._session.post(
                self.endpoint,
                json=payload,
                headers=self._default_headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:  # pragma: no cover - network failure path
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        if response.status_code >= 400:
            # Attempt to surface structured error details whenever possible
            detail: Any
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(
                f"OpenRouter API error {response.status_code}: {detail}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Failed to decode OpenRouter response as JSON") from exc

        if "error" in data:
            raise RuntimeError(f"OpenRouter API error: {data['error']}")

        return self.convert_from_provider_format(data)

    def convert_to_provider_format(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Convert canonical Ko-AgentBench format to OpenRouter payload."""
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = self.config.get("tool_choice", "auto")

        if self.request_overrides:
            # request_overrides last to allow callers to override defaults deliberately
            payload.update(self.request_overrides)

        return payload

    def convert_from_provider_format(self, response: Any) -> Dict[str, Any]:
        """Convert OpenRouter response JSON into the canonical adapter format."""
        if not isinstance(response, dict):
            raise TypeError("OpenRouter response must be a dict")

        choices = response.get("choices", []) or []
        usage = response.get("usage") or {}

        normalized_choices = []
        for choice in choices:
            message = choice.get("message") or {}
            role = message.get("role", "assistant")
            content = message.get("content", "")

            # OpenRouter may return content as a list of segments; flatten to string.
            if isinstance(content, list):
                flattened_segments: List[str] = []
                for segment in content:
                    if isinstance(segment, dict):
                        flattened_segments.append(segment.get("text", ""))
                    else:
                        flattened_segments.append(str(segment))
                content = "".join(flattened_segments)

            normalized_choice: Dict[str, Any] = {
                "message": {
                    "role": role,
                    "content": content or "",
                },
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                },
                "model": response.get("model") or self.model_id,
                "finish_reason": choice.get("finish_reason"),
            }

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                normalized_calls: List[Dict[str, Any]] = []
                for call in tool_calls:
                    function = call.get("function") if isinstance(call, dict) else None
                    normalized_calls.append(
                        {
                            "id": call.get("id") if isinstance(call, dict) else None,
                            "type": call.get("type") if isinstance(call, dict) else None,
                            "function": {
                                "name": (function or {}).get("name"),
                                "arguments": (function or {}).get("arguments"),
                            }
                            if isinstance(function, dict)
                            else None,
                        }
                    )
                normalized_choice["message"]["tool_calls"] = normalized_calls

            normalized_choices.append(normalized_choice)

        if len(normalized_choices) > 1:
            return {
                "choices": normalized_choices,
                "n": len(normalized_choices),
            }

        # Single-choice responses are flattened for backwards compatibility
        return normalized_choices[0] if normalized_choices else {
            "message": {"role": "assistant", "content": ""},
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "model": response.get("model") or self.model_id,
            "finish_reason": None,
        }

