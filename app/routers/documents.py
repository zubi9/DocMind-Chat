import datetime
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.indexing import delete_document_vectors
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
            docs.append(
                DocumentInfo(
                    filename=file.name,
                    size_bytes=stat.st_size,
                    modified_at=datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                )
            )
    return DocumentListResponse(documents=docs, count=len(docs))


@router.delete("/{filename}", response_model=DeleteResponse)
def delete_document(filename: str) -> DeleteResponse:
    """Deletes a document from disk and best-effort removes its vectors from the index.

    For a guaranteed-clean vector store after deletes, call
    POST /ingest/reindex afterwards.
    """
    file_path = Path(settings.user_docs_dir) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

    file_path.unlink()
    delete_document_vectors(filename)

    return DeleteResponse(status="success", detail=f"Document '{filename}' deleted.")
