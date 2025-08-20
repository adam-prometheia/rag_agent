# rag\_agent

Modular, port–adapter **RAG agent** with a Gradio UI. Backends (LLM, vector, reranker, loader, verifier) are swappable via settings. Structured logging includes per-request `trace` and an optional JSON format for containers.

---

## Features

* Clean layering: **ports → adapters → factory → service → UI**
* LLM via **Ollama** by default (e.g. `phi3:mini`), easily switchable
* Vector store via **Chroma**
* Optional Hugging Face **cross-encoder** reranker
* Claim extraction + verification hook
* Structured logging (INFO breadcrumbs, `logger.exception`, JSON toggle)

---

## Requirements

* Python **3.10** (recommended)
* (Local OCR/PDF) poppler + tesseract (Docker image already includes these)

---

## Quickstart (local)

```bash
python -m venv .venv
. .venv/Scripts/activate    # PowerShell: .venv\\Scripts\\Activate.ps1
pip install -U pip
pip install -e .\[ui,vector,reranker,llm,loader]

# Run the UI
python -m rag\_agent.ui.gradio\_app
# open http://localhost:7860
```

### Settings

Pydantic settings (env vars override defaults in `settings.py`). Common knobs:

| Env var                  | Meaning                    | Example                                                           |
| ------------------------ | -------------------------- | ----------------------------------------------------------------- |
| `MODEL\_PROVIDER`         | LLM backend                | `ollama`                                                          |
| `MODEL\_NAME`             | Model id/name              | `phi3:mini`                                                       |
| `OLLAMA\_HOST`            | Ollama base URL            | `http://127.0.0.1:11434` (local) / `http://ollama:11434` (Docker) |
| `RETRIEVER\_K`            | Retriever top-k            | `6`                                                               |
| `RERANKER\_TOP\_K`         | Reranker top-k             | `3`                                                               |
| `INDEX\_DIR`              | Chroma path                | `./storage/chroma` (default)                                      |
| `CLEAR\_INDEX\_ON\_STARTUP` | Wipe/rebuild index on boot | `0` or `1`                                                        |
| `DEBUG`                  | Verbose logs               | `0` or `1`                                                        |
| `LOG\_JSON`               | JSON log output            | `0` or `1`                                                        |

> Tip: add a local `.env` (not committed) to set these conveniently.

---

## Logging

* **INFO** breadcrumbs at start/end of major ops (`ask`, `astream`, `verify`, ingestion)
* **DEBUG** for counts/timings/decisions (only when `DEBUG=1`)
* **ERROR** with `logger.exception(...)` includes stack traces
* **JSON logs** when `LOG\_JSON=1`; includes `trace`, `user`, and any extra fields you add

Example JSON line:

```json
{"ts":"2025-08-12T15:28:01Z","level":"INFO","logger":"rag\_agent.core.service","msg":"astream:start","trace":"53a765da","user":"-","question\_len":42}
```

---

## Tests

```bash
pip install -e .\[dev]
pytest -q
```

* Uses fakes for LLM and Vector for fast, offline tests.

---

## Docker (with Compose)

This repo includes a `Dockerfile` and `docker-compose.yml` that run:

* **ollama** (model server)
* **app** (Gradio UI)

### Run

```bash
docker compose build
docker compose up
# open http://localhost:7860
```

* The app talks to Ollama at `http://ollama:11434` (service name inside the compose network).
* On first request, Ollama auto-downloads `phi3:mini` into its own volume.

### Index persistence (optional)

Current compose keeps things simple: **no Chroma persistence** for the app. To persist later:

1. Set `INDEX\_DIR=/data/index` in the app env.
2. Add a volume mount `chroma-data:/data/index`.
3. Optionally set `CLEAR\_INDEX\_ON\_STARTUP=0`.

---

## Project structure

```
rag\_agent/
  core/            # DTOs, ports, service, logging
  adapters/        # LLM/vector/reranker/verifier/loader implementations
  ui/              # gradio\_app.py entry point
  tests/           # unit tests with fakes
Dockerfile
docker-compose.yml
pyproject.toml
README.md
```

---

## License

MIT — see the LICENSE file below. Third‑party packages and model runtimes retain their own licenses; model weights (e.g., via Ollama) are pulled at runtime and are subject to each model’s license, the same goes for API's.

---


