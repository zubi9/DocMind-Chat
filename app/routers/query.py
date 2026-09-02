import logging

from fastapi import APIRouter, HTTPException

from app.core.query_engine import QueryNotReadyError, ask_question
from app.models import QueryRequest, QueryResponse

logger = logging.getLogger("docmind.routers.query")

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Asks a question against the ingested knowledge base and returns the answer + source chunks."""
    try:
        return ask_question(
            question=request.question,
            top_k=request.top_k,
            response_mode=request.response_mode,
        )
    except QueryNotReadyError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}") from e
