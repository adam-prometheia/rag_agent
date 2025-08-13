from __future__ import annotations
from typing import List
from docx import Document
from rag_agent.core.dto    import DocumentChunk
from rag_agent.core.errors import LoaderError
from rag_agent.settings    import settings
from . import register_loader

@register_loader("docx")
class DocxLoader:
    def load(self, file_path: str) -> List[DocumentChunk]:
        try:
            doc = Document(file_path)
        except Exception as e:
            raise LoaderError(f"DOCX parse failed: {e}") from e

        words, chunks, idx = [], [], 0
        for para in doc.paragraphs:
            words.extend(para.text.split())
            if len(words) >= settings.chunk_size:
                text = " ".join(words)
                chunks.append(DocumentChunk(
                    id=f"{file_path}-chunk-{idx}",
                    text=text,
                    metadata={"source": file_path},
                ))
                idx += 1
                words = []
        if words:
            chunks.append(DocumentChunk(
                id=f"{file_path}-chunk-{idx}",
                text=" ".join(words),
                metadata={"source": file_path},
            ))
        return chunks
