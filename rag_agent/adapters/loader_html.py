# adapters/loader_html.py
from __future__ import annotations
from typing import List
from unstructured.partition.html import partition_html
from rag_agent.core.dto    import DocumentChunk
from rag_agent.core.errors import LoaderError
from rag_agent.settings    import settings
from . import register_loader

@register_loader("html")
class HtmlLoader:
    def load(self, file_path: str) -> List[DocumentChunk]:
        try:
            elements = partition_html(filename=file_path)
        except Exception as e:
            raise LoaderError(f"HTML parse failed: {e}") from e

        text = "\n".join(el.text for el in elements if el.text.strip())
        # reuse TxtLoader chunking logic
        words, chunks, idx = text.split(), [], 0
        buf = []
        for w in words:
            buf.append(w)
            if len(buf) >= settings.chunk_size:
                chunks.append(" ".join(buf))
                idx += 1
                buf = []
        if buf:
            chunks.append(" ".join(buf))

        return [
            DocumentChunk(
                id=f"{file_path}-chunk-{i}",
                text=chunk,
                metadata={"source": file_path},
            )
            for i, chunk in enumerate(chunks)
        ]
