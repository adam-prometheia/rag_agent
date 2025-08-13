from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

import chromadb
from chromadb.errors import ChromaError

from rag_agent.core.dto import DocumentChunk
from rag_agent.core.errors import VectorError
from rag_agent.core.ports import VectorPort
from rag_agent.settings import settings
from . import register_vector


@register_vector("chroma")
class ChromaVectorStore(VectorPort):
    """Adapter that satisfies VectorPort using a local Chroma DB."""

    def __init__(self, *, index_path: str | None = None):
        # Determine index path
        index_path = index_path or settings.index_dir or "./storage/chroma"
        try:
            # 1) Clear folder if requested
            if settings.clear_index_on_startup:
                db_dir = Path(index_path)
                if db_dir.exists():
                    shutil.rmtree(db_dir)

            # 2) Create (or reuse) the persistent client
            client = chromadb.PersistentClient(path=index_path)
            self.collection = client.get_or_create_collection("rag_index")
        except ChromaError as e:
            raise VectorError(f"Failed to open Chroma index at {index_path}: {e}") from e

    # ───────────────────────── add_documents ─────────────────────────
    def add_documents(self, docs: List[DocumentChunk]) -> None:
        try:
            self.collection.add(
                ids=[d.id for d in docs],
                documents=[d.text for d in docs],
                metadatas=[d.metadata for d in docs],
            )
        except ChromaError as e:
            raise VectorError(f"Failed to add docs: {e}") from e

    # ──────────────────────── similarity_search ──────────────────────
    def similarity_search(self, query: str, k: int) -> List[DocumentChunk]:
        try:
            res = self.collection.query(query_texts=[query], n_results=k)
            return [
                DocumentChunk(
                    id=doc_id,
                    text=doc_text,
                    metadata=meta or {},
                )
                for doc_id, doc_text, meta in zip(
                    res["ids"][0], res["documents"][0], res["metadatas"][0]
                )
            ]
        except ChromaError as e:
            raise VectorError(f"Query failed: {e}") from e
