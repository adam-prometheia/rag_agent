from __future__ import annotations
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from rag_agent.core.dto import DocumentChunk
from rag_agent.core.ports import RerankerPort
from rag_agent.adapters import register_reranker
from rag_agent.settings import settings

from rag_agent.core.logging_setup import get_logger
logger = get_logger(__name__)

@register_reranker("hf-cross-encoder")
class HFReRanker(RerankerPort):
    """
    Cross-encoder reranker using a Hugging Face sequence classification model.
    - rerank(query, candidates, top_k): returns [(chunk, score)] sorted desc
    - top_k(query, chunks, k): returns [chunk, ...] (adapter convenience for Service)
    """

    def __init__(self):
        model_name = getattr(settings, "reranker_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.max_length = getattr(settings, "reranker_max_length", 256)
        self.batch_size = getattr(settings, "reranker_batch_size", 16)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # speed & determinism
        logger.info("reranker:init", extra={"model": model_name,
                                            "device": self.device.type,
                                            "max_len": self.max_length})

    def _scores(self, query: str, texts: List[str]) -> List[float]:
        """
        Return raw scores (higher is better) for query vs each text.
        Uses pair encoding and no_grad for speed.
        """
        scores: List[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                # Pair encoding: (query, passage) is the standard for cross-encoders
                enc = self.tokenizer(
                    [query] * len(batch_texts),
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)
                logits = self.model(**enc).logits  # [B, num_labels]
                # Some models are binary (1 logit), some have 2 classes; use last dim safely
                # If 1-dim, squeeze; if 2-dim, take score for "relevant" (index 1)
                if logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1).float().tolist()
                else:
                    batch_scores = logits[:, -1].float().tolist()
                scores.extend(batch_scores)
        return scores

    def rerank(
        self,
        query: str,
        candidates: List[DocumentChunk],
        top_k: int | None = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        if not candidates:
            return []
        texts = [c.text for c in candidates]
        raw_scores = self._scores(query, texts)
        ranked = sorted(zip(candidates, raw_scores), key=lambda x: x[1], reverse=True)
        return ranked if top_k is None else ranked[:top_k]

    def top_k(self,
              query: str,
              chunks: List[DocumentChunk],
              k: int = 1) -> List[DocumentChunk]:
        """Adapter convenience for the Service: returns chunk objects only."""
        return [ch for ch, _ in self.rerank(query=query, candidates=chunks, top_k=k)]
