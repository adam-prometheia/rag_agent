# rag_agent/adapters/vector_chroma.py
from __future__ import annotations
import shutil
from pathlib import Path
import json
from typing import List
import chromadb
from chromadb.errors import ChromaError

from rag_agent.core.dto import DocumentChunk
from rag_agent.core.errors import VectorError
from rag_agent.core.ports import VectorPort, EmbeddingPort
from rag_agent.settings import settings
from . import register_vector

@register_vector("chroma")
class ChromaVectorStore(VectorPort):
    """Adapter that satisfies VectorPort using a local Chroma DB with precomputed embeddings."""

    def __init__(self, *, index_path: str | None = None, embedder: EmbeddingPort | None = None, metric: str | None = None):
        if embedder is None:
            raise VectorError("ChromaVectorStore requires an embedder instance")
        self.embedder = embedder
        self.metric = metric or settings.embedding_metric or "cosine"

        index_path = index_path or settings.index_dir or "./storage/chroma"
        try:
            if settings.clear_index_on_startup:
                db_dir = Path(index_path)
                if db_dir.exists():
                    shutil.rmtree(db_dir)

            client = chromadb.PersistentClient(path=index_path)
            fp = {
                "embedding_provider": settings.embedding_provider,
                "embedding_model": settings.embedding_model,
                "normalise": settings.embedding_normalise,
                "metric": self.metric,
                "chunk_size": settings.chunk_size,
            }
            self.collection = client.get_or_create_collection(
                name="rag_index",
                metadata={"fingerprint": json.dumps(fp)}
            )
            meta = self.collection.metadata or {}
            old_fp = meta.get("fingerprint")
            try:
                old_fp = json.loads(old_raw) if isinstance(old_raw, str) else (old_raw or {})
            except Exception:
                old_fp = {}
            if old_fp and old_fp != fp:
                from rag_agent.core.logging_setup import get_logger
                get_logger(__name__).warning(
                    "vector:fingerprint:mismatch",
                    extra={"old": old_fp, "new": fp}
                )
            from rag_agent.core.logging_setup import get_logger
            get_logger(__name__).info(
                "vector:open",
                extra={"path": str(index_path), "docs": self.collection.count()}
            )
        except ChromaError as e:
            raise VectorError(f"Failed to open Chroma index at {index_path}: {e}") from e

    def add_documents(self, docs: List[DocumentChunk]) -> None:
        try:
            texts = [d.text for d in docs]
            vecs  = self.embedder.embed_documents(texts)
            self.collection.add(
                ids=[d.id for d in docs],
                embeddings=vecs,
                metadatas=[d.metadata for d in docs],
                documents=[d.text for d in docs],
            )
        except ChromaError as e:
            raise VectorError(f"Failed to add docs: {e}") from e

    def similarity_search(self, query: str, k: int) -> List[DocumentChunk]:
        try:
            qvec = self.embedder.embed_query(query)
            res = self.collection.query(query_embeddings=[qvec], n_results=k)
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
