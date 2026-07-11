"""Pydantic request/response schemas used across the API routers."""
from typing import Optional

from pydantic import BaseModel, Field


class YoutubeIngestRequest(BaseModel):
    url: str = Field(..., description="Full YouTube video URL to transcribe and ingest.")


class WebIngestRequest(BaseModel):
    url: str = Field(..., description="URL of a web page to extract and ingest.")


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
    index_stale: bool = False


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int
    modified_at: str
    source_type: str = "upload"
    source_url: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    count: int


class DeleteResponse(BaseModel):
    status: str
    detail: str
    index_status: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
    app_version: str
    index_ready: bool


class ModelCatalogEntry(BaseModel):
    id: str
    name: str
    publisher: str
    params: str
    notes: str
    gated: Optional[bool] = None
    context_window: Optional[int] = None
    dims: Optional[int] = None


class ModelsResponse(BaseModel):
    llms: list[ModelCatalogEntry]
    embeddings: list[ModelCatalogEntry]
    active_llm: Optional[str] = None
    active_embedding: Optional[str] = None


class ModelSwitchRequest(BaseModel):
    model_id: str


class ModelSwitchResponse(BaseModel):
    status: str
    detail: str
    active_llm: Optional[str] = None
    active_embedding: Optional[str] = None
    index_status: Optional[str] = None


class IndexStatusResponse(BaseModel):
    index_status: str
    active_llm: Optional[str] = None
    active_embedding: Optional[str] = None
    last_built_at: Optional[str] = None
    embedded_doc_count: int = 0
    document_count_on_disk: int = 0
