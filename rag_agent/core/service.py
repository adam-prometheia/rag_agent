from __future__ import annotations
from typing import List, Dict, AsyncIterator

from rag_agent.core.dto import QuestionDTO, AnswerDTO, Citation, DocumentChunk
from rag_agent.core.ports import LLMPort, VectorPort, LoaderPort, RerankerPort, VerifierPort
from rag_agent.core.errors import LLMError, VectorError, LoaderError, RerankerError, VerifierError
from rag_agent.settings import settings
from rag_agent.core.logging_setup import get_logger, request_ctx  
import time, uuid

logger = get_logger(__name__)

class RAGService:
    """Facade the UI calls. Hides LLM/vector details behind DTOs."""

    def __init__(
        self,
        *,
        llm: LLMPort,
        vector: VectorPort,
        loader: LoaderPort,
        reranker: RerankerPort,
        verifier: VerifierPort,
    ):
        self.llm = llm
        self.vector = vector
        self.loader = loader
        self.reranker = reranker
        self.verifier = verifier
        self.k = settings.retriever_k
        self.reranker_top_k = settings.reranker_top_k

        # Fallback knobs (claim-centric expansion)
        self.expand_enabled = getattr(settings, "verifier_expand_on_low_conf", True)
        self.expand_threshold = getattr(
            settings, "verifier_expand_threshold", getattr(settings, "verification_threshold", 0.6)
        )
        self.expand_k = getattr(settings, "verifier_expand_k", max(8, self.k))

    # ─────────────────────── sync answer (blocking) ────────────────────────
    def ask(self, q: QuestionDTO, history: List[Dict[str, str]] | None = None) -> AnswerDTO:
        history = history or []

        trace = uuid.uuid4().hex[:8]

        request_ctx.set({"trace": trace, "user_id": q.user_id or "-"})
        
        t0 = time.perf_counter()
        logger.info("ask:start", extra={"trace": trace, "question_len": len(q.text)})

        try:
            logger.debug(f"[{trace}] ask: retrieving", extra={"question_len": len(q.text)})
            chunks = self._retrieve(q.text)
        except Exception:
            logger.exception("ask:retrieval:error", extra={"trace": trace})
            raise  # re-raise so the caller still sees the failure
        
        prompt = self._build_prompt(q.text, chunks, history)

        # 1) Generate the answer
        try:
            answer_txt = self.llm.complete(prompt, max_tokens=512)
        except Exception:
            logger.exception("ask:llm:error", extra={"trace": trace})
            raise

        # 2) Build citations
        citations = self._chunks_to_citations(chunks)

        # 3) Extract claims
        claims = self.verifier.extract_claims(answer_txt, citations)

        # 4) Score claims (with optional claim-centric expansion)
        self._score_claims_with_fallback(claims, chunks)
        total = time.perf_counter() - t0
        logger.info("ask:done", extra={"trace": trace, "elapsed_s": round(total, 3)})

        return AnswerDTO(text=answer_txt, citations=citations, claims=claims)

    # ─────────────────────── async streaming answer ───────────────────────
    async def astream(
        self,
        q: QuestionDTO,
        history: List[Dict[str, str]] | None = None
        ) -> AsyncIterator[str]:
        
        """Yield answer tokens incrementally, then a final empty chunk."""
        
        history = history or []
        trace = uuid.uuid4().hex[:8]
        request_ctx.set({"trace": trace, "user_id": q.user_id or "-"})
        t0 = time.perf_counter()
        logger.info("astream:start", extra={"trace": trace, "question_len": len(q.text)})

        try:
            # retrieval + prompt build
            chunks = self._retrieve(q.text)
            prompt = self._build_prompt(q.text, chunks, history)

            # stream tokens
            async for token in self.llm.astream(prompt, max_tokens=512):
                yield token

            # normal completion
            total = time.perf_counter() - t0
            logger.info("astream:done", extra={"trace": trace, "elapsed_s": round(total, 3)})
        except Exception:
            # any failure (retrieval, prompt build, or streaming)
            logger.exception("astream:error", extra={"trace": trace})
            raise

        # end-of-stream marker (kept for your UI)
        yield ""
    # ────────────────────────── helpers ────────────────────────────────────
    def ingest_files(self, paths: list[str]) -> None:
        for p in paths:
            logger.info("ingest:file:start", extra={"path": p})
            try:
                docs = self.loader.load(p)
                self.vector.add_documents(docs)
                logger.info("ingest:file:done", extra={"path": p, "docs": len(docs)})
            except Exception:
                logger.exception("ingest:file:error", extra={"path": p})
                # optional: continue to next file instead of raising
                raise

    def _retrieve(self, query: str) -> List[DocumentChunk]:
        candidates = self.vector.similarity_search(query, k=self.k)

        pool_cap = getattr(settings, "reranker_max_candidates", 8)
        pool = candidates[:pool_cap]

        if self.reranker:
            return self.reranker.top_k(
                query=query,
                chunks=pool,
                k=getattr(settings, "reranker_top_k", 3),
            )
        return pool

    @staticmethod
    def _build_prompt(
        query: str, chunks: List[DocumentChunk], history: List[Dict[str, str]] | None = None
    ) -> str:
        history = history or []
        chat_lines = []
        for turn in history:
            role = turn["role"].capitalize()
            chat_lines.append(f"{role}: {turn['content']}")
        chat_str = "\n".join(chat_lines)

        context = "\n\n".join(c.text for c in chunks)
        return (
            "You are a helpful assistant.\n\n"
            f"Context:\n{context}\n\n"
            f"Chat History:\n{chat_str}\n\n"
            f"Question: {query}\nAnswer in no more than 6 bullet points.\n"
            "Keep messages spartan: flush with relevant information, but nothing more."
        )

    @staticmethod
    def _chunks_to_citations(chunks: List[DocumentChunk]) -> List[Citation]:
        return [
            Citation(chunk_id=c.id, page=c.metadata.get("page"), snippet=c.text[:120]) for c in chunks
        ]

    def verify(self, question: QuestionDTO, history: List[dict], answer_text: str) -> AnswerDTO:
        """Verify a provided answer text against retrieved context for the question."""
        history = history or []

        trace = uuid.uuid4().hex[:8]
        t0 = time.perf_counter()
        logger.info("verify:start", extra={"trace": trace, "answer_len": len(answer_text)})
        
        # Retrieve using the same strategy as ask()
        try:
            chunks = self._retrieve(question.text)
        except Exception:
            logger.exception("verify:retrieval:error", extra={"trace": trace})
            raise
        citations = self._chunks_to_citations(chunks)

        # Extract claims and score with fallback
        try:
            claims = self.verifier.extract_claims(answer_text, citations)
        except Exception:
            logger.exception("verify:extract:error", extra={"trace": trace})
            raise

        try:
            self._score_claims_with_fallback(claims, chunks)
        except Exception:
            logger.exception("verify:score:error", extra={"trace": trace})
            raise

        total = time.perf_counter() - t0
        verified_n = sum(1 for c in claims if getattr(c, "verified", False))
        logger.info(
            "verify:done",
            extra={"trace": trace, "claims": len(claims), "verified": verified_n, "elapsed_s": round(total, 3)},
        )

        return AnswerDTO(text=answer_text, citations=citations, claims=claims)

    # ───────────────────── claim scoring + fallback ───────────────────────
    def _score_claims_with_fallback(self, claims, base_chunks: List[DocumentChunk]) -> None:
        """Score each claim on the initial pool; if confidence is low, expand the pool using the claim text
        and rescore (claim-centric fallback)."""
        limit = getattr(settings, "verifier_evidence_per_claim", 1)
        by_chunk_id = {c.id: c for c in base_chunks}

        for claim in claims:
            # 1) Ensure we have citation candidates
            if not getattr(claim, "citation_ids", None):
                if self.reranker:
                    picked = self.reranker.top_k(query=claim.text, chunks=base_chunks, k=limit)
                    claim.citation_ids = [p.id for p in picked]
                elif base_chunks:
                    claim.citation_ids = [base_chunks[0].id]

            # 2) Score on the current pool
            best_score = None
            best_cid = None
            for cid in (claim.citation_ids or [])[: limit or None]:
                ch = by_chunk_id.get(cid)
                if not ch:
                    continue
                try:
                    s = self.verifier.score(claim.text, ch.text)
                    if best_score is None or s > best_score:
                        best_score, best_cid = s, cid
                except VerifierError:
                    continue

            claim.confidence = best_score
            claim.verified = (best_score is not None) and (
                best_score >= getattr(settings, "verification_threshold", 0.6)
            )
            claim.support_citation_id = best_cid

            # 3) Claim-centric expansion if under-supported
            need_expand = self.expand_enabled and (
                (best_score is None) or (best_score < self.expand_threshold)
            )
            if not need_expand:
                continue

            # Expand retrieval using the claim text itself
            try:
                extra = self.vector.similarity_search(claim.text, k=self.expand_k)
            except Exception:
                extra = []

            # Merge base + extra without duplicates
            merged: dict[str, DocumentChunk] = {c.id: c for c in base_chunks}
            for ch in extra:
                if ch.id not in merged:
                    merged[ch.id] = ch
            expanded = list(merged.values())

            # Pick new evidence against the expanded set
            if self.reranker and expanded:
                picked = self.reranker.top_k(query=claim.text, chunks=expanded, k=limit)
            else:
                picked = expanded[:limit]

            claim.citation_ids = [p.id for p in picked]

            # Rescore using the expanded evidence
            best_score2 = None
            best_cid2 = None
            for cid in claim.citation_ids:
                ch = merged.get(cid)
                if not ch:
                    continue
                try:
                    s = self.verifier.score(claim.text, ch.text)
                    if best_score2 is None or s > best_score2:
                        best_score2, best_cid2 = s, cid
                except VerifierError:
                    continue

            # If improvement, apply
            if best_score2 is not None and (best_score is None or best_score2 > best_score):
                claim.confidence = best_score2
                claim.verified = best_score2 >= getattr(settings, "verification_threshold", 0.6)
                claim.support_citation_id = best_cid2
