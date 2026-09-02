"""
Curated catalog of lightweight models the user can pick from the frontend's
"Model Gallery" panel.

The actual data lives in `model_catalog.json` (same directory) rather than
in this file — editing which models are offered is a JSON edit, not a
code change. This module just loads it, validates the shape loosely, and
exposes backend-aware accessors.

Two backends, two catalogs, one file:
  - HuggingFace: raw fp16/fp32 weights loaded in-process via
    `transformers`. Simple, but memory-hungry — the source of the OOM
    issue seen on CPU inference with larger models.
  - Ollama: pre-quantized GGUF models served by a separate `ollama`
    container over HTTP. Much lower memory footprint for the same model
    family.

`get_active_llm_catalog()` / `get_active_embedding_catalog()` pick the
right one based on `settings.model_backend`, so routers and the frontend
don't need to know which backend is active — `GET /models` just returns
whichever catalog applies.

The catalog is re-read from disk on every call rather than cached, so
editing `model_catalog.json` takes effect on next request with no
container restart needed — it's a tiny file, the I/O cost is negligible.
"""
import json
import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger("docmind.model_catalog")

_CATALOG_PATH = Path(__file__).resolve().parent / "model_catalog.json"

# Used only if model_catalog.json is missing or fails to parse, so the app
# still starts (with an empty gallery) instead of crashing outright.
_EMPTY_CATALOG = {"huggingface": {"llms": [], "embeddings": []}, "ollama": {"llms": [], "embeddings": []}}


def _load_catalog() -> dict:
    try:
        with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error("model_catalog.json not found at %s — model gallery will be empty.", _CATALOG_PATH)
        return _EMPTY_CATALOG
    except json.JSONDecodeError as e:
        logger.error("model_catalog.json is not valid JSON (%s) — model gallery will be empty.", e)
        return _EMPTY_CATALOG

    for backend in ("huggingface", "ollama"):
        data.setdefault(backend, {})
        data[backend].setdefault("llms", [])
        data[backend].setdefault("embeddings", [])

    return data


def _active_backend_key() -> str:
    return "ollama" if settings.model_backend == "ollama" else "huggingface"


def get_active_llm_catalog() -> list[dict]:
    return _load_catalog()[_active_backend_key()]["llms"]


def get_active_embedding_catalog() -> list[dict]:
    return _load_catalog()[_active_backend_key()]["embeddings"]


def get_llm_entry(model_id: str) -> dict | None:
    return next((m for m in get_active_llm_catalog() if m["id"] == model_id), None)


def get_embedding_entry(model_id: str) -> dict | None:
    return next((m for m in get_active_embedding_catalog() if m["id"] == model_id), None)
