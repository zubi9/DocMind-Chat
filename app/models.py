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
    model_backend: str = "huggingface"


class ModelCatalogEntry(BaseModel):
    id: str
    name: str
    publisher: str
    params: str
    notes: str
    gated: Optional[bool] = None
    context_window: Optional[int] = None
    dims: Optional[int] = None
    max_input_tokens: Optional[int] = None
    cached: Optional[bool] = None
    default: Optional[bool] = None


class ModelsResponse(BaseModel):
    llms: list[ModelCatalogEntry]
    embeddings: list[ModelCatalogEntry]
    active_llm: Optional[str] = None
    active_embedding: Optional[str] = None
    backend: str = "huggingface"


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


class EvalDatasetItem(BaseModel):
    id: Optional[str] = None
    question: str
    reference_answer: Optional[str] = None
    expected_sources: Optional[list[str]] = None
    must_include_keywords: Optional[list[str]] = None


class EvalDatasetResponse(BaseModel):
    readme: Optional[str] = None
    items: list[EvalDatasetItem]


class EvalItemResult(BaseModel):
    id: Optional[str] = None
    question: str
    answer: Optional[str] = None
    sources: Optional[list[str]] = None
    latency_s: Optional[float] = None
    index_stale: Optional[bool] = None
    retrieval_hit: Optional[bool] = None
    retrieval_mrr: Optional[float] = None
    keyword_coverage: Optional[float] = None
    semantic_similarity: Optional[float] = None
    judge_faithfulness: Optional[float] = None
    judge_relevance: Optional[float] = None
    error: Optional[str] = None


class EvalSummary(BaseModel):
    total_questions: int
    errors: int
    avg_latency_s: Optional[float] = None
    retrieval_hit_rate: Optional[float] = None
    avg_retrieval_mrr: Optional[float] = None
    avg_keyword_coverage: Optional[float] = None
    avg_semantic_similarity: Optional[float] = None
    avg_judge_faithfulness: Optional[float] = None
    avg_judge_relevance: Optional[float] = None


class EvalReport(BaseModel):
    generated_at: str
    active_models: dict
    summary: EvalSummary
    results: list[EvalItemResult]
