"""
Model weight caching helpers.

HuggingFace's own download layer (`huggingface_hub`) already caches every
file it downloads under `HF_HOME` (see `.env` / the `./data` Docker volume),
so weights are NOT re-downloaded on every container restart as long as that
volume persists. This module adds an explicit preflight check on top of
that so we can (a) log clearly whether a model load will hit the network,
and (b) expose cache status to the frontend so the Model Gallery can show
"Cached" vs "Downloads on first use" before the person picks a model that
might take a while.
"""
import logging

from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

logger = logging.getLogger("docmind.model_cache")


def is_model_cached(model_id: str) -> bool:
    """True if every file for `model_id` is already present in the local HF cache."""
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


def log_cache_status(model_id: str, kind: str) -> bool:
    """Logs + returns whether `model_id` is already cached locally."""
    cached = is_model_cached(model_id)
    if cached:
        logger.info("%s '%s' found in local cache — loading offline, no download needed.", kind, model_id)
    else:
        logger.info("%s '%s' not cached yet — this load will download it from HuggingFace Hub.", kind, model_id)
    return cached
