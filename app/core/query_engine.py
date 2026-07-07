"""
Runs a question through the query engine and shapes the response for the API.

This mirrors `ask_question()` from PART 5 of the sandbox script, but
returns structured data instead of printing to stdout.
"""
import logging

from app.core.indexing import get_query_engine
from app.models import QueryResponse, SourceChunk

logger = logging.getLogger("docmind.query_engine")


def ask_question(question: str, top_k: int | None = None, response_mode: str | None = None) -> QueryResponse:
    logger.info("Question: %s", question)

    query_engine = get_query_engine(top_k=top_k, response_mode=response_mode)
    response = query_engine.query(question)

    sources = [
        SourceChunk(
            source=node.metadata.get("source") if node.metadata else None,
            score=float(node.score) if node.score is not None else 0.0,
            preview=node.text[:200].replace("\n", " "),
        )
        for node in response.source_nodes
    ]

    return QueryResponse(question=question, answer=str(response.response), sources=sources)
