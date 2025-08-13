from typing import AsyncIterator
from rag_agent.core.ports import LLMPort

class FakeLLM(LLMPort):
    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        return "FAKE ANSWER"

    async def astream(self, prompt: str, max_tokens: int = 512) -> AsyncIterator[str]:
        yield "FAKE "
        yield "STREAM"
