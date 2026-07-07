"""Pydantic request/response schemas used across the API routers."""
from typing import Optional

from pydantic import BaseModel, Field


class YoutubeIngestRequest(BaseModel):
    url: str = Field(..., description="Full YouTube video URL to transcribe and ingest.")


class IngestResponse(BaseModel):
    status: str
    detail: str
    source: Optional[str] = None
    chunks_added: Optional[int] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question about your documents.")
    top_k: Optional[int] = Field(None, description="Override the number of chunks retrieved.")
    response_mode: Optional[str] = Field(
        None, description="Override llama-index response mode (compact, tree_summarize, refine)."
    )


class SourceChunk(BaseModel):
    source: Optional[str] = None
    score: float
    preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int
    modified_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    count: int


class DeleteResponse(BaseModel):
    status: str
    detail: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
    app_version: str
    index_ready: bool
