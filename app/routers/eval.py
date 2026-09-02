import logging

from fastapi import APIRouter, HTTPException, Query

from app.core.evaluation import load_dataset, run_evaluation
from app.models import EvalDatasetResponse, EvalReport

logger = logging.getLogger("docmind.routers.eval")

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/dataset", response_model=EvalDatasetResponse)
def get_dataset() -> EvalDatasetResponse:
    """Returns the current evaluation dataset (auto-created with a starter template on first call)."""
    data = load_dataset()
    return EvalDatasetResponse(readme=data.get("_readme"), items=data.get("items", []))


@router.post("/run", response_model=EvalReport)
def run(
    llm_judge: bool = Query(
        False,
        description="Also score each answer with the active LLM as a judge (1-5 faithfulness/relevance). "
        "Adds one extra LLM call per question, so this is noticeably slower.",
    )
) -> EvalReport:
    """Runs every question in the evaluation dataset through the live RAG pipeline and scores the results.

    Requires embeddings to already be built — questions will report a
    per-item error (not a hard failure of the whole run) if the knowledge
    base is empty when this is called.
    """
    try:
        return run_evaluation(use_llm_judge=llm_judge)
    except ValueError as e:
        # e.g. eval_dataset.json was hand-edited into invalid JSON
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Evaluation run failed")
        raise HTTPException(status_code=500, detail=f"Evaluation run failed: {e}") from e
