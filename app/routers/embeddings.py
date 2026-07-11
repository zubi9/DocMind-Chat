import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.indexing import build_embeddings
from app.core.llm_setup import get_active_models
from app.core.state_store import get_state
from app.models import IndexStatusResponse, IngestResponse

logger = logging.getLogger("docmind.routers.embeddings")

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


def _doc_count_on_disk() -> int:
    folder = Path(settings.user_docs_dir)
    if not folder.exists():
        return 0
    return sum(
        1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in settings.supported_extensions
    )


@router.get("/status", response_model=IndexStatusResponse)
def embeddings_status() -> IndexStatusResponse:
    state = get_state()
    active = get_active_models()
    return IndexStatusResponse(
        index_status=state["index_status"],
        active_llm=active["llm"],
        active_embedding=active["embedding"],
        last_built_at=state.get("last_built_at"),
        embedded_doc_count=state.get("embedded_doc_count", 0),
        document_count_on_disk=_doc_count_on_disk(),
    )


@router.post("/build", response_model=IngestResponse)
def build() -> IngestResponse:
    """(Re)builds the vector index from everything currently in user_docs/.

    This is a full rebuild: any previous embeddings for the active
    embedding model are discarded and regenerated from scratch. Use this
    after uploading/ingesting new sources, deleting documents, or
    switching the embedding model.
    """
    try:
        state = build_embeddings()
    except Exception as e:
        logger.exception("Embedding build failed")
        raise HTTPException(status_code=500, detail=f"Embedding build failed: {e}") from e

    count = state.get("embedded_doc_count", 0)
    if count == 0:
        detail = "No documents found in user_docs/. Ingest something first, then build embeddings."
    else:
        detail = f"Embeddings built for {count} document(s). Your knowledge base is up to date."

    return IngestResponse(status="success", detail=detail, chunks_added=count)
