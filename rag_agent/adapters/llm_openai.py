from __future__ import annotations

import asyncio, os
from typing import AsyncIterator

import openai
from openai import OpenAI
from rag_agent.core.ports import LLMPort
from rag_agent.core.errors import AuthError, RetryableLLMError
from rag_agent.settings import settings
from . import register_llm
import time


_MAX_RETRIES = 3
_RETRY_BACKOFF = 2  # seconds


@register_llm("openai")
class OpenAILLM(LLMPort):
    """Concrete adapter that satisfies LLMPort using OpenAI's chat API."""

    def __init__(self, *, api_key: str | None = None, model: str | None = None):
        api_key = api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise AuthError("OPENAI_API_KEY not provided")

        self.client = OpenAI(api_key=api_key)
        self.model = model or settings.model_name

    # ───────────────────────────────── complete ──────────────────────────────────
    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        for attempt in range(_MAX_RETRIES):
            try:
                res = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0,
                )
                return res.choices[0].message.content.strip()
            except openai.AuthenticationError as e:
                raise AuthError(str(e)) from e
            except (openai.RateLimitError, openai.APITimeoutError) as e:
                if attempt == _MAX_RETRIES - 1:
                    raise RetryableLLMError(str(e)) from e
                time.sleep(_RETRY_BACKOFF * (attempt + 1))

        # will never reach here
        raise RuntimeError("unreachable")

    # ───────────────────────────────── astream ──────────────────────────────────
    async def astream(
        self, prompt: str, max_tokens: int = 512
    ) -> AsyncIterator[str]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stream=True,
                temperature=0,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:          # skip empty keep-alive deltas
                    yield delta
        except openai.AuthenticationError as e:
            raise AuthError(str(e)) from e
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            raise RetryableLLMError(str(e)) from e

