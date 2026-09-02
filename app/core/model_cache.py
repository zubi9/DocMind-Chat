"""
Model weight caching helpers.

HuggingFace backend: HuggingFace's own download layer (`huggingface_hub`)
already caches every file it downloads under `HF_HOME` (see `.env` / the
`./data` Docker volume), so weights are NOT re-downloaded on every
container restart as long as that volume persists. This module adds an
explicit preflight check on top of that so we can (a) log clearly whether
a model load will hit the network, and (b) expose cache status to the
frontend so the Model Gallery can show "Cached" vs "Downloads on first
use" before the person picks a model that might take a while.

Ollama backend: the equivalent check is "has this tag already been
pulled?", answered by the Ollama server itself via `GET /api/tags` (see
`app/core/ollama_client.py`). Ollama manages its own on-disk model store
independently of `HF_HOME`.
"""
import logging

from app.config import settings

logger = logging.getLogger("docmind.model_cache")


def _is_cached_huggingface(model_id: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError
    except ImportError:
        logger.debug("huggingface_hub not installed (expected for the Ollama-only image variant).")
        return False

    try:
        snapshot_download(repo_id=model_id, local_files_only=True)
        return True
    except LocalEntryNotFoundError:
        return False
    except Exception as e:
        # Any other error (bad repo id, gated repo needing auth, etc.) --
        # treat as "not cached" rather than crash the caller. The real
        # error (if any) will surface when we actually try to load it.
        logger.debug("Cache check failed for '%s': %s", model_id, e)
        return False


def _is_cached_ollama(model_id: str) -> bool:
    from app.core.ollama_client import is_pulled

    return is_pulled(model_id)


def is_model_cached(model_id: str) -> bool:
    """True if `model_id` is already available locally for the active backend."""
    if settings.model_backend == "ollama":
        return _is_cached_ollama(model_id)
    return _is_cached_huggingface(model_id)


def log_cache_status(model_id: str, kind: str) -> bool:
    """Logs + returns whether `model_id` is already cached locally."""
    cached = is_model_cached(model_id)
    source = "Ollama's local model store" if settings.model_backend == "ollama" else "the local HuggingFace cache"
    if cached:
        logger.info("%s '%s' found in %s — no download needed.", kind, model_id, source)
    else:
        logger.info("%s '%s' not cached yet — this load will pull/download it.", kind, model_id)
    return cached
