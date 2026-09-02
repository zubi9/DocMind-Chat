import logging

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.core.ingestion import (
    IngestionError,
    fetch_web_page,
    fetch_youtube_transcript,
    load_single_document,
    save_uploaded_file,
)
from app.core.state_store import mark_documents_changed
from app.models import IngestResponse, WebIngestRequest, YoutubeIngestRequest

logger = logging.getLogger("docmind.routers.ingest")

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_NEEDS_EMBEDDING_NOTE = "Click 'Create Embeddings' on the Documents page to add it to the searchable index."


@router.post("/document", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """Uploads a PDF/DOCX/TXT/MD file and saves it. Does NOT embed it — see /embeddings/build."""
    content = await file.read()

    try:
        saved_path = save_uploaded_file(file.filename, content)
    except IngestionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Validate it's actually parseable before accepting it, even though we don't embed yet.
    doc = load_single_document(saved_path)
    if doc is None:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"'{file.filename}' could not be parsed into readable text (empty, corrupt, or unsupported).",
        )

    mark_documents_changed()

    return IngestResponse(
        status="success",
        detail=f"Document '{file.filename}' saved. {_NEEDS_EMBEDDING_NOTE}",
        source=file.filename,
    )


@router.post("/youtube", response_model=IngestResponse)
def ingest_youtube(request: YoutubeIngestRequest) -> IngestResponse:
    """Fetches a YouTube transcript and saves it. Does NOT embed it — see /embeddings/build."""
    try:
        transcript_path = fetch_youtube_transcript(request.url)
    except IngestionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    doc = load_single_document(transcript_path)
    if doc is None:
        transcript_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Transcript could not be parsed into readable text.")

    mark_documents_changed()

    return IngestResponse(
        status="success",
        detail=f"Transcript for '{request.url}' saved. {_NEEDS_EMBEDDING_NOTE}",
        source=transcript_path.name,
    )


@router.post("/web", response_model=IngestResponse)
def ingest_web(request: WebIngestRequest) -> IngestResponse:
    """Fetches a web page, extracts its main text, and saves it. Does NOT embed it — see /embeddings/build."""
    try:
        page_path = fetch_web_page(request.url)
    except IngestionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    mark_documents_changed()

    return IngestResponse(
        status="success",
        detail=f"Page '{request.url}' saved. {_NEEDS_EMBEDDING_NOTE}",
        source=page_path.name,
    )
