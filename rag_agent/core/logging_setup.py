# rag_agent/core/logging_setup.py
import logging, sys, os, json
from datetime import datetime
from contextvars import ContextVar

request_ctx: ContextVar[dict] = ContextVar("request_ctx", default={})

class CtxAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        ctx = request_ctx.get()
        extra = kwargs.get("extra", {})
        extra.update(ctx)
        kwargs["extra"] = extra
        return msg, kwargs

def get_logger(name: str) -> logging.Logger:
    return CtxAdapter(logging.getLogger(name), {})

# --- helpers for safe formatting ---
class _EnsureKeys(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # guarantee these keys exist so formatters don't crash
        if not hasattr(record, "trace"):
            record.trace = "-"
        if not hasattr(record, "user_id"):
            record.user_id = "-"
        return True

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace": getattr(record, "trace", "-"),
            "user": getattr(record, "user_id", "-"),
        }
        # include any 'extra' fields passed to logger.* (e.g., elapsed_s, claims, verified, question_len)
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in {
                "name","msg","args","levelname","levelno","pathname","filename","module",
                "exc_info","exc_text","stack_info","lineno","funcName","created","msecs",
                "relativeCreated","thread","threadName","processName","process","asctime",
                "trace","user_id"
            }
        }
        if extras:
            payload.update(extras)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

def init_logging(debug: bool = False) -> None:
    """Configure root logging once. Human text locally; JSON if LOG_JSON=1."""
    if getattr(init_logging, "_configured", False):
        return

    use_json = os.getenv("LOG_JSON", "").strip().lower() in {"1", "true", "yes", "on"}
    level = logging.DEBUG if debug else logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_EnsureKeys())
    if use_json:
        handler.setFormatter(_JsonFormatter())
    else:
        fmt = "%(asctime)s %(levelname)s %(name)s [trace=%(trace)s user=%(user_id)s] - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))

    # root at INFO to avoid super noisy libs; we'll set our package below
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    # our package level honors the debug flag
    logging.getLogger("rag_agent").setLevel(level)

    # quiet common chatty libraries
    for noisy in ["urllib3", "httpx", "httpcore", "asyncio", "chromadb", "transformers", "gradio", "uvicorn"]:
        logging.getLogger(noisy).setLevel(logging.WARNING if debug else logging.ERROR)

    init_logging._configured = True
