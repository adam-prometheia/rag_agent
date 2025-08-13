from __future__ import annotations
from typing import List
from pathlib import Path
from rag_agent.core.dto    import DocumentChunk
from rag_agent.core.errors import LoaderError
from rag_agent.settings    import settings
from . import register_loader

@register_loader("txt")
class TxtLoader:
    def load(self, file_path: str) -> List[DocumentChunk]:
        try:
            text = Path(file_path).read_text(encoding="utf-8")
        except Exception as e:
            raise LoaderError(f"TXT load failed: {e}") from e

        words, chunks, idx = [], [], 0
        for word in text.split():
            words.append(word)
            if len(words) >= settings.chunk_size:
                chunks.append(" ".join(words))
                idx += 1
                words = []
        if words:
            chunks.append(" ".join(words))

        return [
            DocumentChunk(
                id=f"{file_path}-chunk-{i}",
                text=chunk,
                metadata={"source": file_path},
            )
            for i, chunk in enumerate(chunks)
        ]
