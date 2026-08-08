import datetime
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.state_store import get_doc_source, mark_documents_changed, remove_doc_source
from app.models import DeleteResponse, DocumentInfo, DocumentListResponse

logger = logging.getLogger("docmind.routers.documents")

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """Lists every document currently sitting in the user_docs folder."""
    folder = Path(settings.user_docs_dir)
    docs = []
    for file in sorted(folder.iterdir()):
        if file.is_file() and file.suffix.lower() in settings.supported_extensions:
            stat = file.stat()
            source = get_doc_source(file.name) or {}
            docs.append(
                DocumentInfo(
                    filename=file.name,
                    size_bytes=stat.st_size,
                    modified_at=datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    source_type=source.get("source_type", "upload"),
                    source_url=source.get("source_url"),
                )
            )
    return DocumentListResponse(documents=docs, count=len(docs))


@router.delete("/{filename}", response_model=DeleteResponse)
def delete_document(filename: str) -> DeleteResponse:
    """Deletes a document from disk. The vector index is left untouched until the
    next 'Create Embeddings' build — the index is marked 'stale' so the UI can
    prompt the user to rebuild it.
    """
    file_path = Path(settings.user_docs_dir) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

    file_path.unlink()
    remove_doc_source(filename)
    state = mark_documents_changed()

    return DeleteResponse(
        status="success",
        detail=(
            f"Document '{filename}' deleted. Your embeddings are now out of date — "
            "click 'Create Embeddings' on the Documents page to update your knowledge base."
        ),
        index_status=state["index_status"],
    )
