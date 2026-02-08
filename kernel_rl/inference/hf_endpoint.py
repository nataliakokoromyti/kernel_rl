"""
Hugging Face Inference Endpoint client helpers.

Uses the Hugging Face Hub InferenceClient with the Messages API
(OpenAI-compatible chat completions) for TGI-backed endpoints.
"""

from __future__ import annotations

from typing import Any

from huggingface_hub import AsyncInferenceClient


class HFEndpointClient:
    """Async client for Hugging Face Inference Endpoints (chat-completions)."""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._client = AsyncInferenceClient(
            base_url=endpoint_url,
            api_key=api_key,
            timeout=timeout,
        )

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        response = await self._client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            extra_body=extra_body,
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""
