# ---------- builder ----------
FROM python:3.10-slim AS builder
WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY . .

# Upgrade pip, install CPU-only torch first, then your package with extras
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.0,<3" && \
    pip install --no-cache-dir ".[ui,vector,reranker,llm,loader]"

# ---------- runtime ----------
FROM python:3.10-slim
WORKDIR /app

# curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

# Copy from the *builder* stage (the alias above must exist!)
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app ./

# Create runtime user and ensure ownership so CLEAR_INDEX_ON_STARTUP works
RUN useradd -m -u 10001 appuser \
  && mkdir -p /app/storage/chroma \
  && chown -R appuser:appuser /app

USER appuser

# Defaults (override via env/compose)
ENV PORT=7860 \
    DEBUG=false \
    LOG_JSON=0 \
    MODEL_PROVIDER=ollama \
    MODEL_NAME=phi3:mini \
    EMBEDDING_PROVIDER=ollama \
    EMBEDDING_MODEL=nomic-embed-text \
    EMBEDDING_METRIC=cosine

EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=15s --timeout=3s --retries=20 \
  CMD curl -fsS http://127.0.0.1:7860/ || exit 1

# v2 entrypoint
CMD ["python", "-m", "rag_agent.ui.gradio_app"]
