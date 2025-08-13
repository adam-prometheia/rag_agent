"""
Domain-level exceptions raised by port adapters.
Core and UI layers import these to decide user-facing behaviour.
"""
class LLMError(Exception):
    """Base class for LLM-related failures."""


class AuthError(LLMError):
    """Wrong or missing credentials."""


class RetryableLLMError(LLMError):
    """Temporary outage, time-out, or rate-limit.

    Adapters should retry internally a few times before raising this.
    """

class VectorError(Exception):
    """Base class for vector store failures (connection, bad index, etc.)."""

class LoaderError(Exception):
    """Raised when a document can’t be parsed or chunked."""

class RerankerError(Exception):
    """Raised when the reranking step fails (e.g. model loading, scoring)."""

class VerifierError(Exception):
    """Raised when the verifier adapter fails (e.g., model error)."""
