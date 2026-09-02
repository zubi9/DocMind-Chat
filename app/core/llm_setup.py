"""
Configures and switches the global LlamaIndex `Settings` object
(embedding model + LLM).

Supports two backends, selected via `MODEL_BACKEND` in `.env`:

  - "huggingface" (default): raw model weights loaded in-process via
    `transformers`. Simple, single-container, but memory-hungry — CPU
    generation with even a "small" model can transiently need several GB,
    which is what caused the OOM kills seen earlier.
  - "ollama": inference is delegated over HTTP to a separate `ollama`
    container serving pre-quantized GGUF models via llama.cpp. Much lower
    memory footprint for an equivalent model, at the cost of running a
    second container (see docker-compose.ollama.yml).

Model imports for each backend are lazy (inside the loader functions) so
that a slim, HuggingFace-free image (see Dockerfile.ollama /
requirements-ollama.txt) doesn't need `torch`/`transformers` installed at
all, and vice versa.

Either way, switching models at runtime, unloading the previous one, and
persisting the active choice to disk works identically from the caller's
perspective (`switch_llm`, `switch_embedding_model`, `get_active_models`).
"""
import gc
import logging

from llama_index.core import Settings

from app.config import settings
from app.core.model_cache import log_cache_status
from app.core.model_catalog import get_embedding_entry, get_llm_entry
from app.core.state_store import get_state, update_state

logger = logging.getLogger("docmind.llm_setup")

_current_llm_id: str | None = None
_current_embed_id: str | None = None
_models_configured = False


class ModelSwitchError(Exception):
    """Raised when a requested model_id isn't in the curated catalog, or fails to load."""


def _resolve_device_map() -> str:
    if settings.llm_device_map != "auto":
        return settings.llm_device_map
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA is available. LLM will attempt to use GPU.")
            return "auto"
    except ImportError:
        pass
    logger.info("CUDA not available (or torch not importable). Falling back to CPU.")
    return "cpu"


def _free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --------------------------------------------------------------------- #
# HuggingFace backend
# --------------------------------------------------------------------- #

def _load_embed_model_huggingface(embed_id: str) -> None:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    entry = get_embedding_entry(embed_id)
    kwargs = {"model_name": embed_id}
    if entry:
        if entry.get("query_instruction"):
            kwargs["query_instruction"] = entry["query_instruction"]
        if entry.get("text_instruction"):
            kwargs["text_instruction"] = entry["text_instruction"]

    logger.info("Loading embedding model (HuggingFace): %s", embed_id)
    try:
        Settings.embed_model = HuggingFaceEmbedding(**kwargs)
    except TypeError:
        # Older llama-index versions may not support instruction kwargs.
        logger.warning("Embedding model does not accept instruction kwargs, retrying without them.")
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_id)


def _load_llm_huggingface(llm_id: str) -> None:
    from llama_index.llms.huggingface import HuggingFaceLLM

    entry = get_llm_entry(llm_id)
    context_window = entry["context_window"] if entry else settings.llm_context_window
    device_map = _resolve_device_map()

    logger.info(
        "Loading LLM (HuggingFace): %s (device_map=%s, context_window=%s, max_new_tokens=%s)",
        llm_id, device_map, context_window, settings.llm_max_new_tokens,
    )
    Settings.llm = HuggingFaceLLM(
        model_name=llm_id,
        context_window=context_window,
        device_map=device_map,
        max_new_tokens=settings.llm_max_new_tokens,
    )


# --------------------------------------------------------------------- #
# Ollama backend
# --------------------------------------------------------------------- #

def _load_embed_model_ollama(embed_id: str) -> None:
    from llama_index.embeddings.ollama import OllamaEmbedding

    from app.core.ollama_client import ensure_pulled

    ensure_pulled(embed_id)
    logger.info("Loading embedding model (Ollama): %s @ %s", embed_id, settings.ollama_base_url)
    Settings.embed_model = OllamaEmbedding(
        model_name=embed_id,
        base_url=settings.ollama_base_url,
    )


def _load_llm_ollama(llm_id: str) -> None:
    from llama_index.llms.ollama import Ollama

    from app.core.ollama_client import ensure_pulled

    ensure_pulled(llm_id)
    entry = get_llm_entry(llm_id)
    context_window = entry["context_window"] if entry else settings.llm_context_window

    logger.info(
        "Loading LLM (Ollama): %s @ %s (context_window=%s, num_predict=%s)",
        llm_id, settings.ollama_base_url, context_window, settings.llm_max_new_tokens,
    )
    Settings.llm = Ollama(
        model=llm_id,
        base_url=settings.ollama_base_url,
        request_timeout=settings.ollama_request_timeout_s,
        context_window=context_window,
        # Without this, Ollama has no output-length cap at all (unlike the
        # HuggingFace path's max_new_tokens above) — the model can ramble,
        # repeat itself, or hallucinate follow-up Q&A turns indefinitely.
        # Reusing llm_max_new_tokens keeps one shared "answer length" knob
        # across both backends instead of two separately-tuned settings.
        additional_kwargs={"num_predict": settings.llm_max_new_tokens},
    )


# --------------------------------------------------------------------- #
# Backend-agnostic dispatch
# --------------------------------------------------------------------- #

def _load_embed_model(embed_id: str) -> None:
    log_cache_status(embed_id, "Embedding model")
    if settings.model_backend == "ollama":
        _load_embed_model_ollama(embed_id)
    else:
        _load_embed_model_huggingface(embed_id)


def _load_llm(llm_id: str) -> None:
    log_cache_status(llm_id, "LLM")
    if settings.model_backend == "ollama":
        _load_llm_ollama(llm_id)
    else:
        _load_llm_huggingface(llm_id)


def configure_models(force: bool = False) -> None:
    """Loads the persisted (or default) active models. Called once at startup."""
    global _models_configured, _current_llm_id, _current_embed_id
    if _models_configured and not force:
        return

    state = get_state()
    if settings.model_backend == "ollama":
        default_llm, default_embed = settings.ollama_llm_model, settings.ollama_embed_model
    else:
        default_llm, default_embed = settings.llm_model_name, settings.embed_model_name

    llm_id = state.get("active_llm") or default_llm
    embed_id = state.get("active_embedding") or default_embed

    # A saved active_llm/active_embedding from a previous run of the OTHER
    # backend won't exist in this backend's catalog — fall back to this
    # backend's default rather than trying (and failing) to load it.
    if get_llm_entry(llm_id) is None:
        logger.info("Persisted LLM '%s' isn't in the active backend's catalog, using default '%s'.", llm_id, default_llm)
        llm_id = default_llm
    if get_embedding_entry(embed_id) is None:
        logger.info("Persisted embedding '%s' isn't in the active backend's catalog, using default '%s'.", embed_id, default_embed)
        embed_id = default_embed

    _load_embed_model(embed_id)
    _current_embed_id = embed_id

    _load_llm(llm_id)
    _current_llm_id = llm_id

    _models_configured = True
    update_state(active_llm=llm_id, active_embedding=embed_id)
    logger.info(
        "Models loaded successfully (backend=%s, llm=%s, embedding=%s).",
        settings.model_backend, llm_id, embed_id,
    )


def switch_llm(model_id: str) -> None:
    """Loads a new LLM from the active backend's catalog and unloads the previous one."""
    global _current_llm_id
    entry = get_llm_entry(model_id)
    if entry is None:
        raise ModelSwitchError(f"'{model_id}' is not in the model catalog.")

    if model_id == _current_llm_id:
        return

    old_llm = Settings.llm
    try:
        _load_llm(model_id)
    except Exception as e:
        raise ModelSwitchError(f"Failed to load '{model_id}': {e}") from e

    del old_llm
    _free_memory()
    _current_llm_id = model_id
    update_state(active_llm=model_id)


def switch_embedding_model(model_id: str) -> None:
    """Loads a new embedding model from the active backend's catalog and unloads the previous one.

    Note: this does NOT touch the vector index. Each embedding model has
    its own Chroma collection (see app/core/indexing.py); the caller is
    responsible for refreshing index status via
    `indexing.refresh_state_after_embed_switch()`.
    """
    global _current_embed_id
    entry = get_embedding_entry(model_id)
    if entry is None:
        raise ModelSwitchError(f"'{model_id}' is not in the model catalog.")

    if model_id == _current_embed_id:
        return

    old_embed = Settings.embed_model
    try:
        _load_embed_model(model_id)
    except Exception as e:
        raise ModelSwitchError(f"Failed to load '{model_id}': {e}") from e

    del old_embed
    _free_memory()
    _current_embed_id = model_id
    update_state(active_embedding=model_id)


def get_active_models() -> dict:
    return {"llm": _current_llm_id, "embedding": _current_embed_id, "backend": settings.model_backend}


def models_ready() -> bool:
    return _models_configured
