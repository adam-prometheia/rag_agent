from rag_agent.adapters import LLM_REG, LOADER_REG, VECTOR_REG, RERANKER_REG, VERIFIER_REG
from rag_agent.settings import settings
from rag_agent.core.service import RAGService
from rag_agent.core.errors import LLMError, LoaderError, VectorError, RerankerError, VerifierError

from rag_agent.core.logging_setup import init_logging, get_logger
logger = get_logger(__name__)

def build_service() -> RAGService:
    init_logging(getattr(settings, "debug", False))
    logger.info("factory:config", extra={
        "provider": settings.model_provider, "model": getattr(settings, "model_name", "-"),
        "vector": settings.vector_backend or "chroma",
        "reranker": settings.reranker_backend, "verifier": settings.verifier_backend,
        "retriever_k": settings.retriever_k, "reranker_top_k": settings.reranker_top_k,
        "index_dir": settings.index_dir,
    })
    try:
        llm_cls = LLM_REG[settings.model_provider]
    except KeyError:
        raise LLMError(
            f"Unknown LLM backend '{settings.model_provider} : {settings.model_name}'. "
            f"Valid options are: {list(LLM_REG.keys())}"
        )
    try:
        loader_cls = LOADER_REG[settings.loader_backend]
    except KeyError:
        raise LoaderError(
            f"Unknown loader backend '{settings.loader_backend}'. "
            f"Valid options are: {list(LOADER_REG.keys())}"
        )
    try:
        vector_cls = VECTOR_REG[settings.vector_backend or "chroma"]
    except KeyError:
        raise VectorError(
            f"Unknown vector backend '{settings.vector_backend}'. "
            f"Valid options are: {list(VECTOR_REG.keys())}"
        )
    try:
        reranker_cls = RERANKER_REG[settings.reranker_backend]
    except KeyError:
        raise RerankerError(
            f"Unknown reranker backend '{settings.reranker_backend}'. "
            f"Valid options are: {list(RERANKER_REG.keys())}"
        )
    try:
        verifier_cls = VERIFIER_REG[settings.verifier_backend]
    except KeyError:
        raise VerifierError(
            f"Unknown verifier backend '{settings.verifier_backend}'. "
            f"Valid options are: {list(VERIFIER_REG.keys())}"
        )
    llm = llm_cls()
    loader = loader_cls()
    vector = vector_cls(index_path=settings.index_dir)
    reranker = reranker_cls()
    verifier = verifier_cls()
    return RAGService(llm=llm,
                      vector=vector,
                      loader=loader,
                      reranker=reranker,
                      verifier=verifier)
