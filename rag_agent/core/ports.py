from __future__ import annotations
from typing import Protocol, AsyncIterator, List, Tuple
from rag_agent.core.dto import DocumentChunk, Citation, Claim

class LLMPort(Protocol):
    def complete(self, prompt: str, max_tokens: int = 512) -> str: ...
    async def astream(self, prompt: str, max_tokens: int = 512) -> AsyncIterator[str]: ...

class VectorPort(Protocol):
    def add_documents(self, docs: List[DocumentChunk]) -> None: ...
    def similarity_search(self, query: str, k: int) -> List[DocumentChunk]: ...

class EmbeddingPort(Protocol):
    def info(self) -> dict: ...
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...

class LoaderPort(Protocol):
    def load(self, file_path: str) -> list[DocumentChunk]:
        """Parse & chunk a file, return DocumentChunks."""

class RerankerPort(Protocol):
    def rerank(
        self,
        query: str,
        candidates: List[DocumentChunk],
        top_k: int | None = None,
    ) -> List[Tuple[DocumentChunk, float]]: ...
    def top_k(
        self,
        query: str,
        chunks: List[DocumentChunk],
        k: int = 1
    ) -> List[DocumentChunk]: ...

class VerifierPort(Protocol):
    def extract_claims(self, answer_text: str, citations: list[Citation]) -> list[Claim]:
        """Return structured claims that reference citation_ids from `citations`."""

    def score(self, claim: str, evidence: str) -> float:
        """Return support score in [0, 1] for (claim, evidence)."""
