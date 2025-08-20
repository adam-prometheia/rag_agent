from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    # LLM
    model_provider: str = "ollama"
    model_name: str = "phi3:mini"
    openai_api_key: str | None = None

    # Vector store
    vector_backend: str = "chroma"
    index_dir: str = "./storage/chroma"
    clear_index_on_startup: bool = True

    # Embeddings
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"   # tiny, fast default
    embedding_normalise: bool = True
    embedding_metric: str = "cosine"            # for the vector store
    embedding_concurrency: int = 4  # safe default on CPU
    
    # Retrieval/summariser knobs
    retriever_k: int = 6
    quick_chars: int = 12_000

    # Loader
    loader_backend: str = "auto"
    chunk_size: int = 500

    # PDF OCR fallback settings
    pdf_ocr_threshold: int = 1000
    ocr_dpi: int = 300
    poppler_path: str | None = None
    tesseract_cmd: str | None = None

    # Which reranker to use (must match register name)
    reranker_backend: str = "hf-cross-encoder"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_max_length: int = 256
    reranker_batch_size: int = 16
    reranker_max_candidates: int = 8
    reranker_top_k: int = 3 

    # Which verifier to use (pretty much only one for now)
    verifier_backend: str = "llm-check"
    verifier_max_claims: int = 5            
    verifier_evidence_per_claim: int = 1
    verifier_temperature: float = 0.0
    verifier_top_p: float = 1.0
    verifier_max_tokens_score: int = 16
    verification_threshold: float = 0.6

    # Flags
    debug: bool = True
    
    # Minimalist mode override
    if os.getenv("MINIMAL_MODE", "false").lower() == "true":
        RERANKER_PROVIDER = "none"
        VERIFIER_PROVIDER = "none"
        EMBEDDING_CONCURRENCY = int(os.getenv("EMBEDDING_CONCURRENCY", "6"))

    model_config = SettingsConfigDict(env_file=".env", env_prefix="")

settings = Settings()
