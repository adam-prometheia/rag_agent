"""
Package initialisation for rag_agent.adapters.

1.  Define the LLM registry + decorator.
2.  Import built-in adapters so they register themselves.
   (imports come *after* the registry to avoid circular-import errors)
"""

from __future__ import annotations
from typing import Dict, Type

LLM_REG: Dict[str, Type] = {}

def register_llm(name: str):
    """Decorator to add an adapter class to the global registry."""
    def _wrap(cls):
        LLM_REG[name] = cls
        return cls
    return _wrap

VECTOR_REG: dict[str, type] = {}

def register_vector(name: str):
    def _wrap(cls):
        VECTOR_REG[name] = cls
        return cls
    return _wrap

LOADER_REG: dict[str, type] = {}

def register_loader(name: str):
    def _wrap(cls):
        LOADER_REG[name] = cls
        return cls
    return _wrap

RERANKER_REG: dict[str, type] = {}

def register_reranker(name: str):
    def _wrap(cls):
        RERANKER_REG[name] = cls
        return cls
    return _wrap

VERIFIER_REG: dict[str, type] = {}

def register_verifier(name: str):
    def _wrap(cls):
        VERIFIER_REG[name] = cls
        return cls
    return _wrap

EMBEDDER_REG: dict[str, type] = {}

def register_embedder(name: str):
    def _wrap(cls):
        EMBEDDER_REG[name] = cls
        return cls
    return _wrap

from .llm_openai import OpenAILLM   # noqa: F401,E402
from .llm_ollama import OllamaLLM   # noqa: F401,E402
from .vector_chroma import ChromaVectorStore   # noqa: F401,E402
from rag_agent.adapters.loader_pdf   import PDFLoader   # noqa: F401,E402
from rag_agent.adapters.loader_docx  import DocxLoader  # noqa: F401,E402
from rag_agent.adapters.loader_txt   import TxtLoader   # noqa: F401,E402
from rag_agent.adapters.loader_html  import HtmlLoader  # noqa: F401,E402
from rag_agent.adapters.loader_auto  import AutoLoader  # noqa: F401,E402
from rag_agent.adapters.reranker_hf import HFReRanker   # noqa: F401,E402
from rag_agent.adapters.verifier_llm import LLVerify   # noqa: F401,E402
from rag_agent.adapters.embedder_ollama import OllamaEmbedder  # noqa: F401,E402

__all__ = [
    "LLM_REG", "register_llm",
    "VECTOR_REG", "register_vector",
    "LOADER_REG", "register_loader",
    "RERANKER_REG", "register_reranker",
    "VERIFIER_REG", "register_verifier",
    "EMBEDDER_REG", "register_embedder"
]
