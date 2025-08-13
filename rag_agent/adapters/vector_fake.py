from typing import List
from rag_agent.core.dto import DocumentChunk
from rag_agent.core.ports import VectorPort
from rag_agent.adapters import register_vector
from typing import List, Dict

@register_vector("fake")
class FakeVector(VectorPort):
    def __init__(self):
        self._store: Dict[str, DocumentChunk] = {}

    def add_documents(self, docs: List[DocumentChunk]) -> None:
        for d in docs:
            self._store[d.id] = d

    def similarity_search(self, query: str, k: int) -> List[DocumentChunk]:
        # naive: return first k docs (deterministic for tests)
        return list(self._store.values())[:k]
