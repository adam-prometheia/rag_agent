import gradio as gr
import os
from typing import List, Dict, Any, AsyncGenerator, Tuple
from rag_agent.adapters.factory import build_service
from rag_agent.core.dto import QuestionDTO, AnswerDTO
from rag_agent.core.errors import LLMError, VectorError, LoaderError, RerankerError, VerifierError
from rag_agent.settings import settings

# Build the service once on startup
service = build_service()

# -------------------- helpers --------------------

def _to_paths(files: List[Any] | None) -> List[str]:
    paths: List[str] = []
    for f in files or []:
        # Gradio can return a string path or a dict with a 'path' key
        if isinstance(f, str):
            paths.append(f)
        elif isinstance(f, dict) and "path" in f:
            paths.append(str(f["path"]))
        elif hasattr(f, "name") and hasattr(f, "read") and hasattr(f, "seek"):
            # Temporary file object
            try:
                paths.append(getattr(f, "name"))
            except Exception:
                pass
    return paths

def _shorten(text: str, max_chars: int = 160) -> str:
    """Collapse whitespace and clamp to max_chars with an ellipsis."""
    s = " ".join((text or "").split())
    return s if len(s) <= max_chars else s[: max_chars - 1].rstrip() + "…"

def _claims_to_markdown(dto: AnswerDTO) -> str:
    """Compact, readable summary of claim checks for the UI."""
    if not dto.claims:
        return "_No claims extracted._"

    total = len(dto.claims)
    verified = sum(1 for c in dto.claims if c.verified)
    lines = [f"**Claim checks:** {verified}/{total} verified\n"]

    for c in dto.claims:
        status = "✅" if c.verified is True else "❌" if c.verified is False else "—"
        conf = f"{round(float(c.confidence) * 100)}%" if c.confidence is not None else "n/a"
        lines.append(f"- {status} **{conf}** — {_shorten(c.text, 160)}")

    return "\n".join(lines)


# -------------------- gradio callbacks --------------------

async def on_run(question: str,
           stream: bool,
           verify: bool,
           uploads: List[Any] | None,
           history: List[Dict[str, str]]) -> AsyncGenerator[Tuple[list[dict], str], None]:
    """Main callback. Yields (history, claim_rows) while streaming.

    Now verifies the exact streamed text using `service.verify(...)` so we never re-generate the answer just to score claims.
    """
    history = history or []
    q = (question or "").strip()
    if not q:
        yield history, ""
        return

    # Ingest any uploads before answering
    try:
        file_paths = _to_paths(uploads)
        if file_paths:
            service.ingest_files(file_paths)
    except LoaderError as e:
        # Surface loader issues in the chat but keep the UI responsive
        history.append({"role": "assistant", "content": f"Could not load files: {e}"})
        yield history, ""
        return

    q_dto = QuestionDTO(text=q)

    # Streaming path
    if stream:
        # push user message
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": ""})
        try:
            final_text_parts: List[str] = []
            async for token in service.astream(q_dto, history):
                if token:
                    final_text_parts.append(token)
                    history[-1]["content"] += token
                    yield history, ""  # stream answer; claims come after

            # Verify the *exact* streamed text without re-asking the model
            if verify:
                final_text = "".join(final_text_parts)
                dto: AnswerDTO = service.verify(q_dto, history, final_text)
                rows = _claims_to_markdown(dto)
                yield history, rows
            else:
                yield history, ""

        except (LLMError, VectorError, RerankerError, VerifierError) as e:
            # Append a terse error line; don't intermingle with the answer tokens
            history.append({"role": "assistant", "content": f"Error: {e}"})
            yield history, ""
        return

    # Non-streaming path (blocking)
    try:
        dto: AnswerDTO = service.ask(q_dto, history)
        # Reflect the produced answer into the chat
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": dto.text})
        rows = (_claims_to_markdown(dto) if verify else "")
        yield history, rows
    except (LLMError, VectorError, RerankerError, VerifierError) as e:
        history.append({"role": "assistant", "content": f"Error: {e}"})
        yield history, ""


# -------------------- UI --------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG Agent with Verifier", fill_height=True) as demo:
        gr.Markdown("# RAG Agent · Answer + Claim Verifier")
        with gr.Row():
            question = gr.Textbox(label="Ask a question", placeholder="Ask me about your documents…", lines=3)
        with gr.Row():
            stream = gr.Checkbox(value=True, label="Stream answer")
            verify = gr.Checkbox(value=True, label="Verify claims")
            uploads = gr.File(label="Upload files", file_count="multiple", type="filepath")
        run = gr.Button("Run", variant="primary")

        history_state = gr.State([])  # per-session chat history

        with gr.Row():
            answer = gr.Chatbot(label="Chat", height=360, show_label=True, type="messages")
            claims_md = gr.Markdown(label="Claim checks")

        run.click(
            on_run,
            inputs=[question, stream, verify, uploads, history_state],
            outputs=[answer, claims_md],
        ).then(
            lambda h: h,  # keep latest history in state
            inputs=[answer],
            outputs=[history_state],
        )

    return demo


if __name__ == "__main__":
    import os
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_api=False,  # optional
    )
