from __future__ import annotations
from typing import List, Tuple
import os
import time
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from rag_agent.adapters import register_embedder

@register_embedder("ollama")
class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", base_url: str | None = None, concurrency: int = 4):
        self.model = model
        # prefer env in Docker, fall back to localhost for local dev
        self.base_url = (base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        # allow override via env; factory will also pass from settings
        self.concurrency = int(os.getenv("EMBEDDING_CONCURRENCY", concurrency))
        self._dim = None  # filled on first call

    def info(self) -> dict:
        return {"provider": "ollama", "model": self.model, "dim": self._dim or -1, "normalize": True}

    def _embed_one(self, idx_text: Tuple[int, str]) -> Tuple[int, List[float]]:
        idx, t = idx_text
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                with httpx.Client(timeout=30) as client:
                    r = client.post(f"{self.base_url}/api/embeddings",
                                    json={"model": self.model, "prompt": t})
                    if r.status_code == 404:
                        raise RuntimeError(
                            f"Ollama model '{self.model}' not found. "
                            f"Run: `docker compose exec ollama ollama pull {self.model}`"
                        )
                    r.raise_for_status()
                    vec = r.json()["embedding"]
                    if self._dim is None:
                        self._dim = len(vec)
                    return idx, vec
            except (httpx.ConnectError, httpx.ReadTimeout):
                if attempt == max_attempts:
                    raise
                time.sleep(0.8 * attempt)  # tiny backoff

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Bound concurrency (just in case)
        workers = max(1, min(self.concurrency, 16))
        results: List[List[float]] = [None] * len(texts)  # type: ignore
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self._embed_one, (i, t)) for i, t in enumerate(texts)]
            for fut in as_completed(futures):
                i, vec = fut.result()
                results[i] = vec
        return _l2_normalize(results)  # type: ignore

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

def _l2_normalize(vecs: List[List[float]]) -> List[List[float]]:
    import math
    out = []
    for v in vecs:
        n = math.sqrt(sum(x*x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out
