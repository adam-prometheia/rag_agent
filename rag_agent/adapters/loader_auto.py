from __future__ import annotations
from pathlib import Path
from typing import List
from rag_agent.core.dto    import DocumentChunk
from rag_agent.core.errors import LoaderError
from rag_agent.adapters    import LOADER_REG, register_loader

@register_loader("auto")
class AutoLoader:
    def load(self, file_path: str) -> List[DocumentChunk]:
        ext = Path(file_path).suffix.lower().lstrip(".")
        loader_cls = LOADER_REG.get(ext)
        if not loader_cls:
            raise LoaderError(f"No loader for '.{ext}' files")
        return loader_cls().load(file_path)
