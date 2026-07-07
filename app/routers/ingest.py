import logging

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.core.indexing import add_documents, rebuild_index_from_scratch
from app.core.ingestion import (
    IngestionError,
    fetch_youtube_transcript,
    load_single_document,
    save_uploaded_file,
)
from app.models import IngestResponse, YoutubeIngestRequest

logger = logging.getLogger("docmind.routers.ingest")

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/document", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """Uploads a PDF/DOCX/TXT/MD file, parses it, and adds it to the live index."""
    content = await file.read()

    try:
        saved_path = save_uploaded_file(file.filename, content)
    except IngestionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    doc = load_single_document(saved_path)
    if doc is None:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"'{file.filename}' could not be parsed into readable text (empty, corrupt, or unsupported).",
        )

    chunks_added = add_documents([doc])

    return IngestResponse(
        status="success",
        detail=f"Document '{file.filename}' ingested and added to the index.",
        source=file.filename,
        chunks_added=chunks_added,
    )


@router.post("/youtube", response_model=IngestResponse)
def ingest_youtube(request: YoutubeIngestRequest) -> IngestResponse:
    """Fetches a YouTube transcript and adds it to the live index."""
    try:
        transcript_path = fetch_youtube_transcript(request.url)
    except IngestionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    doc = load_single_document(transcript_path)
    if doc is None:
        raise HTTPException(status_code=422, detail="Transcript could not be parsed into readable text.")

    chunks_added = add_documents([doc])

    return IngestResponse(
        status="success",
        detail=f"YouTube transcript for '{request.url}' ingested and added to the index.",
        source=transcript_path.name,
        chunks_added=chunks_added,
    )


@router.post("/reindex", response_model=IngestResponse)
def reindex() -> IngestResponse:
    """Wipes the vector store and rebuilds the entire index from everything in user_docs/.

    Use this after manually dropping files into the data volume, or to
    recover from any drift between the vector store and the documents on
    disk (e.g. after several individual deletes).
    """
    doc_count = rebuild_index_from_scratch()
    return IngestResponse(
        status="success",
        detail=f"Index rebuilt from {doc_count} document(s) in user_docs/.",
        chunks_added=doc_count,
    )
