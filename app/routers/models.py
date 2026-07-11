import logging

from fastapi import APIRouter, HTTPException

from app.core.indexing import refresh_state_after_embed_switch
from app.core.llm_setup import ModelSwitchError, get_active_models, switch_embedding_model, switch_llm
from app.core.model_catalog import EMBEDDING_CATALOG, LLM_CATALOG
from app.models import ModelsResponse, ModelSwitchRequest, ModelSwitchResponse

logger = logging.getLogger("docmind.routers.models")

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    active = get_active_models()
    return ModelsResponse(
        llms=LLM_CATALOG,
        embeddings=EMBEDDING_CATALOG,
        active_llm=active["llm"],
        active_embedding=active["embedding"],
    )


@router.post("/llm", response_model=ModelSwitchResponse)
def set_llm(request: ModelSwitchRequest) -> ModelSwitchResponse:
    """Switches the active generation model. Loading a new model can take a while
    (download on first use, then load into memory) — this call blocks until it's ready.
    """
    try:
        switch_llm(request.model_id)
    except ModelSwitchError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    active = get_active_models()
    return ModelSwitchResponse(
        status="success",
        detail=f"Generation model switched to '{request.model_id}'.",
        active_llm=active["llm"],
        active_embedding=active["embedding"],
    )


@router.post("/embedding", response_model=ModelSwitchResponse)
def set_embedding(request: ModelSwitchRequest) -> ModelSwitchResponse:
    """Switches the active embedding model. This does not re-embed your documents —
    it marks the index stale/empty for the new model so you can rebuild it explicitly.
    """
    try:
        switch_embedding_model(request.model_id)
    except ModelSwitchError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    state = refresh_state_after_embed_switch(request.model_id)
    active = get_active_models()
    return ModelSwitchResponse(
        status="success",
        detail=(
            f"Embedding model switched to '{request.model_id}'. "
            "Click 'Create Embeddings' on the Documents page to (re)build your knowledge base with it."
        ),
        active_llm=active["llm"],
        active_embedding=active["embedding"],
        index_status=state["index_status"],
    )
