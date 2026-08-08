"""
Small persisted JSON store for state that needs to survive restarts but
doesn't warrant a real database: which models are active, whether the
vector index is in sync with the documents on disk, and where each
ingested document came from (upload / YouTube / web page).

Not designed for concurrent writers — this is a single-worker local/dev
service, so a simple read-modify-write is sufficient.
"""
import json
import logging
import threading
from pathlib import Path

from app.config import settings

logger = logging.getLogger("docmind.state_store")

_lock = threading.Lock()
_STATE_PATH = Path(settings.data_dir) / "app_state.json"

_DEFAULT_STATE = {
    "active_llm": settings.llm_model_name,
    "active_embedding": settings.embed_model_name,
    # index_status: "empty" (never built) | "stale" (docs/model changed since last build) | "synced"
    "index_status": "empty",
    "last_built_at": None,
    "embedded_doc_count": 0,
    "embed_model_at_build": None,
    "llm_model_at_build": None,
    "doc_sources": {},  # filename -> {"source_type": "upload"|"youtube"|"web", "source_url": str|None, "ingested_at": iso str}
}


def _read() -> dict:
    if not _STATE_PATH.exists():
        return dict(_DEFAULT_STATE)
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        merged = dict(_DEFAULT_STATE)
        merged.update(data)
        return merged
    except Exception as e:
        logger.warning("Could not read state file, using defaults: %s", e)
        return dict(_DEFAULT_STATE)


def _write(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_state() -> dict:
    with _lock:
        return _read()


def update_state(**kwargs) -> dict:
    with _lock:
        state = _read()
        state.update(kwargs)
        _write(state)
        return state


def mark_documents_changed() -> dict:
    """Call whenever a document is ingested or deleted — invalidates a 'synced' index."""
    with _lock:
        state = _read()
        if state["index_status"] == "synced":
            state["index_status"] = "stale"
            _write(state)
        return state


def refresh_status_for_embed_model(embed_id: str, vector_count: int) -> dict:
    """Call after switching the active embedding model.

    Since each embedding model gets its own Chroma collection, we can't
    know whether that collection's contents still match what's on disk —
    so any existing vectors are treated as 'stale' (safe default: prompts
    a rebuild) and an empty collection is marked 'empty'.
    """
    with _lock:
        state = _read()
        state["index_status"] = "stale" if vector_count > 0 else "empty"
        state["embed_model_at_build"] = embed_id if vector_count > 0 else state.get("embed_model_at_build")
        _write(state)
        return state


def register_doc_source(filename: str, source_type: str, source_url: str | None = None) -> None:
    from datetime import datetime, timezone

    with _lock:
        state = _read()
        state["doc_sources"][filename] = {
            "source_type": source_type,
            "source_url": source_url,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        _write(state)


def remove_doc_source(filename: str) -> None:
    with _lock:
        state = _read()
        state["doc_sources"].pop(filename, None)
        _write(state)


def get_doc_source(filename: str) -> dict | None:
    return get_state()["doc_sources"].get(filename)
