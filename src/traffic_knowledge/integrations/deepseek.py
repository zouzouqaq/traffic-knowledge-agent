"""Typed client for grounded DeepSeek answer generation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class DeepSeekGeneration:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float


class DeepSeekClientError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"{code}: {message}")


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        temperature: float,
        max_output_tokens: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.transport = transport

    def generate(self, system_prompt: str, user_prompt: str) -> DeepSeekGeneration:
        started = time.perf_counter_ns()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "thinking": {"type": "disabled"},
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        response = self._post_with_one_retry(payload)
        try:
            content = response["choices"][0]["message"]["content"].strip()
            usage = response.get("usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            raise DeepSeekClientError(
                "LLM_INVALID_RESPONSE",
                "DeepSeek response schema is invalid",
            ) from error
        if not content:
            raise DeepSeekClientError(
                "LLM_INVALID_RESPONSE",
                "DeepSeek answer is empty",
            )
        return DeepSeekGeneration(
            content=content,
            model=str(response.get("model", self.model)),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=round((time.perf_counter_ns() - started) / 1_000_000, 3),
        )

    def _post_with_one_retry(self, payload: dict) -> dict:
        for attempt in range(2):
            try:
                with httpx.Client(
                    base_url=self.base_url,
                    timeout=self.timeout_seconds,
                    transport=self.transport,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                ) as client:
                    response = client.post("/chat/completions", json=payload)
            except httpx.TimeoutException as error:
                if attempt == 0:
                    continue
                raise DeepSeekClientError(
                    "LLM_TIMEOUT",
                    "DeepSeek request timed out",
                ) from error
            except httpx.HTTPError as error:
                if attempt == 0:
                    continue
                raise DeepSeekClientError(
                    "LLM_UNAVAILABLE",
                    "DeepSeek network request failed",
                ) from error

            retryable = response.status_code == 429 or response.status_code >= 500
            if retryable and attempt == 0:
                continue
            self._raise_for_error(response)
            try:
                return response.json()
            except ValueError as error:
                raise DeepSeekClientError(
                    "LLM_INVALID_RESPONSE",
                    "DeepSeek returned invalid JSON",
                ) from error
        raise DeepSeekClientError("LLM_UNAVAILABLE", "DeepSeek request failed")

    @staticmethod
    def _raise_for_error(response: httpx.Response) -> None:
        if response.status_code in {401, 403}:
            raise DeepSeekClientError(
                "LLM_AUTH_FAILED",
                "DeepSeek authentication failed",
                response.status_code,
            )
        if response.status_code == 429:
            raise DeepSeekClientError(
                "LLM_RATE_LIMITED",
                "DeepSeek rate limit reached",
                response.status_code,
            )
        if response.status_code >= 500:
            raise DeepSeekClientError(
                "LLM_UNAVAILABLE",
                "DeepSeek service unavailable",
                response.status_code,
            )
        if response.is_error:
            raise DeepSeekClientError(
                "LLM_INVALID_RESPONSE",
                "DeepSeek rejected the request",
                response.status_code,
            )
