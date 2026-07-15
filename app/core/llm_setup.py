"""
Configures and switches the global LlamaIndex `Settings` object
(embedding model + LLM).

Unlike the original single-shot setup, this now supports switching either
model at runtime (triggered from the frontend's Model Gallery), unloading
the previous model to free memory, and persisting the active choice to
disk so it survives a container restart.
"""
import gc
import logging

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

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


def _load_embed_model(embed_id: str) -> None:
    log_cache_status(embed_id, "Embedding model")

    entry = get_embedding_entry(embed_id)
    kwargs = {"model_name": embed_id}
    if entry:
        if entry.get("query_instruction"):
            kwargs["query_instruction"] = entry["query_instruction"]
        if entry.get("text_instruction"):
            kwargs["text_instruction"] = entry["text_instruction"]

    logger.info("Loading embedding model: %s", embed_id)
    try:
        Settings.embed_model = HuggingFaceEmbedding(**kwargs)
    except TypeError:
        # Older llama-index versions may not support instruction kwargs.
        logger.warning("Embedding model does not accept instruction kwargs, retrying without them.")
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_id)


def _load_llm(llm_id: str) -> None:
    log_cache_status(llm_id, "LLM")

    entry = get_llm_entry(llm_id)
    context_window = entry["context_window"] if entry else settings.llm_context_window
    device_map = _resolve_device_map()

    logger.info(
        "Loading LLM: %s (device_map=%s, context_window=%s, max_new_tokens=%s)",
        llm_id, device_map, context_window, settings.llm_max_new_tokens,
    )
    Settings.llm = HuggingFaceLLM(
        model_name=llm_id,
        context_window=context_window,
        device_map=device_map,
        max_new_tokens=settings.llm_max_new_tokens,
    )


def configure_models(force: bool = False) -> None:
    """Loads the persisted (or default) active models. Called once at startup."""
    global _models_configured, _current_llm_id, _current_embed_id
    if _models_configured and not force:
        return

    state = get_state()
    llm_id = state.get("active_llm") or settings.llm_model_name
    embed_id = state.get("active_embedding") or settings.embed_model_name

    _load_embed_model(embed_id)
    _current_embed_id = embed_id

    _load_llm(llm_id)
    _current_llm_id = llm_id

    _models_configured = True
    logger.info("Models loaded successfully (llm=%s, embedding=%s).", llm_id, embed_id)


def switch_llm(model_id: str) -> None:
    """Loads a new LLM from the catalog and unloads the previous one."""
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
    """Loads a new embedding model from the catalog and unloads the previous one.

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
    return {"llm": _current_llm_id, "embedding": _current_embed_id}


def models_ready() -> bool:
    return _models_configured
