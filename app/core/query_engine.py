"""
Runs a question through the query engine and shapes the response for the API.

This mirrors `ask_question()` from PART 5 of the sandbox script, but
returns structured data instead of printing to stdout.
"""
import logging
import time

from app.core.indexing import can_query, get_query_engine
from app.core.state_store import get_state
from app.models import QueryResponse, SourceChunk

logger = logging.getLogger("docmind.query_engine")


class QueryNotReadyError(Exception):
    """Raised when there is no built index to query yet."""


def ask_question(question: str, top_k: int | None = None, response_mode: str | None = None) -> QueryResponse:
    ready, reason = can_query()
    if not ready:
        raise QueryNotReadyError(reason)

    logger.info("Question received: %s", question)

    query_engine = get_query_engine(top_k=top_k, response_mode=response_mode)

    # Bracket the actual generation call with explicit start/elapsed logs.
    # If the process gets killed mid-generation (e.g. an OOM kill — the
    # most common cause on CPU inference with limited container memory),
    # you'll see "Generating answer..." with no matching "Generated answer"
    # line, which pinpoints exactly where it died instead of the process
    # just going silent.
    start = time.monotonic()
    logger.info("Generating answer... (this can take a while on CPU)")
    try:
        response = query_engine.query(question)
    except Exception:
        elapsed = time.monotonic() - start
        logger.exception("Generation failed after %.1fs", elapsed)
        raise
    elapsed = time.monotonic() - start
    logger.info("Generated answer in %.1fs", elapsed)

    sources = [
        SourceChunk(
            source=node.metadata.get("source") if node.metadata else None,
            score=float(node.score) if node.score is not None else 0.0,
            preview=node.text[:200].replace("\n", " "),
        )
        for node in response.source_nodes
    ]

    index_stale = get_state().get("index_status") == "stale"

    return QueryResponse(
        question=question,
        answer=str(response.response),
        sources=sources,
        index_stale=index_stale,
    )
