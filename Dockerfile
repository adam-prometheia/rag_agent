# ---------- base ----------
FROM python:3.11-slim

# Keep Python snappy & quiet
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps needed by your loaders (OCR/PDF/magic)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 poppler-utils tesseract-ocr curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata first to maximize layer caching
COPY pyproject.toml README.md ./
# Then the package
COPY rag_agent ./rag_agent

# ---------- Python deps ----------
# 1) Upgrade pip
# 2) Install **CPU-only** PyTorch to avoid huge CUDA wheels
# 3) Install your package in editable mode with the selected extras
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch>=2.0,<3.0' \
 && pip install --no-cache-dir -e '.[ui,vector,reranker,llm,loader]'

# ---------- runtime ----------
ENV OLLAMA_HOST=http://ollama:11434 \
    MODEL_PROVIDER=ollama \
    MODEL_NAME=phi3:mini \
    PORT=7860

EXPOSE 7860
CMD ["python", "-m", "rag_agent.ui.gradio_app"]
