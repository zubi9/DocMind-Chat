"""
Configures the global LlamaIndex `Settings` object (embedding model + LLM).

This is the FastAPI-service equivalent of "PART 2: CONFIGURE LIGHTWEIGHT
LOCAL MODELS" in the original Colab sandbox script. It is called exactly
once, on application startup, so the (potentially large) models are only
loaded into memory a single time and reused across all requests.
"""
import logging

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

from app.config import settings

logger = logging.getLogger("docmind.llm_setup")

_models_configured = False


def _resolve_device_map() -> str:
    """Mirrors the sandbox's CUDA auto-detection, but only if the user asked for 'auto'."""
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


def configure_models(force: bool = False) -> None:
    """Load and register the embedding model + LLM on `Settings`.

    Idempotent by default: safe to call multiple times, the expensive
    model loads only happen once unless `force=True`.
    """
    global _models_configured
    if _models_configured and not force:
        return

    logger.info("Loading embedding model: %s", settings.embed_model_name)
    Settings.embed_model = HuggingFaceEmbedding(model_name=settings.embed_model_name)

    device_map = _resolve_device_map()
    logger.info(
        "Loading LLM: %s (device_map=%s, context_window=%s, max_new_tokens=%s)",
        settings.llm_model_name,
        device_map,
        settings.llm_context_window,
        settings.llm_max_new_tokens,
    )
    Settings.llm = HuggingFaceLLM(
        model_name=settings.llm_model_name,
        context_window=settings.llm_context_window,
        device_map=device_map,
        max_new_tokens=settings.llm_max_new_tokens,
    )

    _models_configured = True
    logger.info("Models loaded successfully.")


def models_ready() -> bool:
    return _models_configured
