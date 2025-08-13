from __future__ import annotations

import time
from typing import AsyncIterator

import ollama
from ollama import AsyncClient
try:
    from ollama.exceptions import OllamaError          # type: ignore
except ImportError:
    try:
        from ollama import OllamaError             # type: ignore
    except ImportError:
        try:
            from ollama._types import OllamaError  # type: ignore
        except ImportError:
            class OllamaError(Exception):
                """Generic Ollama client error (stub)."""

from rag_agent.core.ports import LLMPort
from rag_agent.core.errors import AuthError, RetryableLLMError
from rag_agent.settings import settings
from . import register_llm


_MAX_RETRIES = 3
_RETRY_BACKOFF = 2


@register_llm("ollama")
class OllamaLLM(LLMPort):
    """Adapter that satisfies LLMPort using a local (or remote) Ollama server."""

    def __init__(self, *, base_url: str | None = None, model: str | None = None):
        # If OLLAMA_HOST env var is set, ollama-py reads it automatically.
        self.base_url = base_url
        self.model = model or settings.model_name
        self.async_client = AsyncClient(host=base_url) if base_url else AsyncClient()

    # ───────────────────────────── complete ────────────────────────────────
    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        for attempt in range(_MAX_RETRIES):
            try:
                if self.base_url:
                    response = ollama.Client(host=self.base_url).chat(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_predict": max_tokens, "temperature": 0},
                    )
                else:
                    response = ollama.chat(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_predict": max_tokens, "temperature": 0},
                    )
                return response["message"]["content"].strip()
            except OllamaError as e:
                if "context deadline" in str(e).lower() or "connection refused" in str(e).lower():
                    if attempt == _MAX_RETRIES - 1:
                        raise RetryableLLMError(str(e)) from e
                    time.sleep(_RETRY_BACKOFF * (attempt + 1))
                else:
                    raise RetryableLLMError(str(e)) from e

    # ───────────────────────────── astream ─────────────────────────────────
    async def astream(
        self, prompt: str, max_tokens: int = 512
    ) -> AsyncIterator[str]:
        try:
            stream = await self.async_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": max_tokens, "temperature": 0},
                stream=True,
            )
            async for chunk in stream:
                delta = chunk["message"]["content"]
                if delta:
                    yield delta
        except OllamaError as e:
            raise RetryableLLMError(str(e)) from e

