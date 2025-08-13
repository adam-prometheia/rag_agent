import pytest
from rag_agent.core.service import RAGService
from rag_agent.core.dto import DocumentChunk, QuestionDTO
from rag_agent.core.ports import LoaderPort, RerankerPort, VerifierPort
from rag_agent.settings import settings
from rag_agent.adapters.llm_fake import FakeLLM
from rag_agent.adapters.vector_fake import FakeVector

# --- tiny stubs for the missing ports ---
class NoopLoader(LoaderPort):
    def load(self, file_path: str) -> list[DocumentChunk]:
        return []

class PassthroughReranker(RerankerPort):
    def rerank(self, query, candidates, top_k=None):
        ranked = [(c, 1.0) for c in candidates]  # pretend every chunk scores 1.0
        return ranked if top_k is None else ranked[:top_k]
    def top_k(self, query, chunks, k=1):
        return chunks[:k]

class NoopVerifier(VerifierPort):
    def extract_claims(self, answer_text, citations):
        return []  # keep verification out of this test
    def score(self, claim, evidence) -> float:
        return 1.0

def make_service(k=2) -> RAGService:
    settings.retriever_k = k
    settings.reranker_top_k = k
    return RAGService(
        llm=FakeLLM(),
        vector=FakeVector(),
        loader=NoopLoader(),
        reranker=PassthroughReranker(),
        verifier=NoopVerifier(),
    )

def seed(vector: FakeVector, n=3):
    docs = [DocumentChunk(id=f"d{i}", text=f"text {i}", metadata={"page": i}) for i in range(n)]
    vector.add_documents(docs)
    return docs

def test_ask_returns_fake_answer_and_citations(caplog):
    svc = make_service(k=2)
    seed(svc.vector, n=5)
    caplog.set_level("INFO", logger="rag_agent")  # capture your breadcrumbs
    ans = svc.ask(QuestionDTO(text="hello"))
    assert ans.text == "FAKE ANSWER"             # from FakeLLM
    assert len(ans.citations) == 2               # k=2 → two citations from FakeVector
    msgs = [r.getMessage() for r in caplog.records if r.name.startswith("rag_agent")]
    assert any("ask:start" in m for m in msgs)
    assert any("ask:done" in m for m in msgs)

@pytest.mark.asyncio
async def test_astream_yields_tokens_and_terminator():
    svc = make_service(k=1)
    seed(svc.vector, n=1)
    tokens = []
    async for tok in svc.astream(QuestionDTO(text="hi")):
        tokens.append(tok)
    assert tokens[-1] == ""                      # service sends "" as the end marker
    assert "".join(tokens[:-1]) == "FAKE STREAM" # from FakeLLM.astream
