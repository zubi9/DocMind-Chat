"""
Lightweight, self-hosted RAG evaluation harness.

Deliberately not built on `ragas` or `deepeval` — both pull in a large
dependency tree (langchain, datasets, etc.) and typically expect an
OpenAI-compatible judge model by default, which conflicts with this
project's whole point: fully local, no external API keys, and (for the
Ollama backend) a torch-free image. Instead this reuses infrastructure
the app already has loaded in-process — the active embedding model for
semantic similarity, and optionally the active LLM as a judge — with zero
new heavy dependencies.

Metrics computed per question (each is skipped, not zeroed, when the
corresponding optional dataset field isn't provided):

  - retrieval_hit / retrieval_mrr: whether an expected source document
    appears among the retrieved chunks, and how highly ranked it is.
    Needs `expected_sources` on the dataset item.
  - keyword_coverage: fraction of `must_include_keywords` present in the
    generated answer (case-insensitive substring match). A crude but
    cheap and interpretable groundedness/completeness proxy.
  - semantic_similarity: cosine similarity between the generated answer's
    and a `reference_answer`'s embeddings, computed with whichever
    embedding model is currently active — free, local, no judge needed.
  - judge_faithfulness / judge_relevance: optional 1-5 ratings from the
    currently active LLM via a structured grading prompt (--llm-judge).
    Quality of this metric is bounded by the local judge model's own
    capability — treat it as a rough signal, not ground truth.

The dataset lives at `{DATA_DIR}/eval_dataset.json`, inside the same
persisted volume as ingested documents, so it survives restarts and can
be hand-edited without a rebuild. A small starter template is written
automatically on first use if the file doesn't exist yet.
"""
import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.core.query_engine import QueryNotReadyError, ask_question

logger = logging.getLogger("docmind.evaluation")

_DEFAULT_DATASET = {
    "_readme": (
        "Edit this file to match documents you've actually ingested — the starter "
        "item below is just a template. Fields per item: 'question' (required); "
        "'reference_answer' (optional — used for semantic similarity and the LLM "
        "judge); 'expected_sources' (optional — exact filenames as shown on the "
        "Documents page, used for retrieval hit-rate/MRR); 'must_include_keywords' "
        "(optional — terms the answer should mention). This file lives in the "
        "persisted data/ volume, so edits here take effect immediately, no rebuild "
        "needed — just re-run the evaluation."
    ),
    "items": [
        {
            "id": "example-1",
            "question": "What is this document about?",
            "reference_answer": "Replace with a short reference answer based on a document you've ingested.",
            "expected_sources": ["replace-with-actual-filename.pdf"],
            "must_include_keywords": ["replace", "these", "keywords"],
        }
    ],
}


def _dataset_path() -> Path:
    return Path(settings.data_dir) / "eval_dataset.json"


def ensure_dataset_exists() -> None:
    path = _dataset_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_DEFAULT_DATASET, indent=2), encoding="utf-8")
        logger.info("Created starter evaluation dataset at %s", path)


def load_dataset() -> dict:
    ensure_dataset_exists()
    try:
        data = json.loads(_dataset_path().read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"eval_dataset.json is not valid JSON: {e}") from e
    data.setdefault("items", [])
    return data


# --------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------- #

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _semantic_similarity(generated: str, reference: str | None) -> float | None:
    if not reference:
        return None
    from llama_index.core import Settings as LISettings

    if LISettings.embed_model is None:
        return None
    try:
        emb_gen = LISettings.embed_model.get_text_embedding(generated)
        emb_ref = LISettings.embed_model.get_text_embedding(reference)
        return round(_cosine_similarity(emb_gen, emb_ref), 4)
    except Exception as e:
        logger.warning("Semantic similarity computation failed: %s", e)
        return None


def _keyword_coverage(answer: str, keywords: list[str] | None) -> float | None:
    if not keywords:
        return None
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 4)


def _retrieval_metrics(source_names: list[str], expected_sources: list[str] | None) -> dict:
    if not expected_sources:
        return {
            "retrieval_hit": None,
            "retrieval_mrr": None,
            "retrieval_recall": None,
            "retrieval_precision": None,
            "retrieval_ndcg": None,
        }
    expected_set = {e.casefold() for e in expected_sources}
    retrieved_set = {name.casefold() for name in source_names}
    relevant_count = len(retrieved_set & expected_set)
    hit = relevant_count > 0
    mrr = 0.0
    for rank, name in enumerate(source_names, start=1):
        if name.casefold() in expected_set:
            mrr = round(1.0 / rank, 4)
            break

    recall = relevant_count / len(expected_set)
    precision = relevant_count / len(retrieved_set) if retrieved_set else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, name in enumerate(source_names, start=1) if name.casefold() in expected_set)
    ideal_count = min(len(expected_set), len(source_names))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return {
        "retrieval_hit": hit,
        "retrieval_mrr": round(mrr, 4),
        "retrieval_recall": round(recall, 4),
        "retrieval_precision": round(precision, 4),
        "retrieval_ndcg": round(ndcg, 4),
    }


_JUDGE_PROMPT = """You are grading an AI assistant's answer for a document Q&A system.

Question: {question}
Reference answer (a guide, may be incomplete): {reference}
Assistant's answer: {answer}

Rate the assistant's answer from 1 (worst) to 5 (best) on:
- faithfulness: consistent with the reference and the question, no fabricated specifics
- relevance: actually addresses the question asked

Respond with ONLY a JSON object and nothing else: {{"faithfulness": <1-5>, "relevance": <1-5>}}
"""


def _llm_judge(question: str, answer: str, reference: str | None) -> dict | None:
    from llama_index.core import Settings as LISettings

    if LISettings.llm is None:
        return None

    prompt = _JUDGE_PROMPT.format(question=question, reference=reference or "(none provided)", answer=answer)
    try:
        raw = str(LISettings.llm.complete(prompt))
    except Exception as e:
        logger.warning("LLM judge call failed: %s", e)
        return None

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        logger.warning("LLM judge did not return parseable JSON: %r", raw[:200])
        return None
    try:
        parsed = json.loads(match.group(0))
        return {
            "judge_faithfulness": parsed.get("faithfulness"),
            "judge_relevance": parsed.get("relevance"),
        }
    except json.JSONDecodeError:
        logger.warning("LLM judge JSON parse failed: %r", raw[:200])
        return None


# --------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------- #

def evaluate_item(item: dict, use_llm_judge: bool = False) -> dict:
    question = item["question"]
    reference = item.get("reference_answer")

    start = time.monotonic()
    try:
        response = ask_question(question)
    except QueryNotReadyError as e:
        return {"id": item.get("id"), "question": question, "error": str(e)}
    except Exception as e:
        logger.exception("Evaluation query failed for %r", question)
        return {"id": item.get("id"), "question": question, "error": f"{type(e).__name__}: {e}"}
    latency_s = round(time.monotonic() - start, 2)

    source_names = [s.source for s in response.sources if s.source]

    result = {
        "id": item.get("id"),
        "question": question,
        "answer": response.answer,
        "sources": source_names,
        "latency_s": latency_s,
        "index_stale": response.index_stale,
    }
    result.update(_retrieval_metrics(source_names, item.get("expected_passages") or item.get("expected_sources")))
    result["keyword_coverage"] = _keyword_coverage(response.answer, item.get("must_include_keywords"))
    result["semantic_similarity"] = _semantic_similarity(response.answer, reference)

    if use_llm_judge:
        judge = _llm_judge(question, response.answer, reference)
        if judge:
            result.update(judge)

    return result


def _mean(results: list[dict], key: str) -> float | None:
    values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
    return round(sum(values) / len(values), 4) if values else None


def run_evaluation(use_llm_judge: bool = False) -> dict:
    dataset = load_dataset()
    items = dataset.get("items", [])

    if use_llm_judge:
        logger.info("Running evaluation over %d question(s) with LLM-judge enabled (slower).", len(items))
    else:
        logger.info("Running evaluation over %d question(s).", len(items))

    results = [evaluate_item(item, use_llm_judge=use_llm_judge) for item in items]

    summary = {
        "total_questions": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "avg_latency_s": _mean(results, "latency_s"),
        "retrieval_hit_rate": _mean(results, "retrieval_hit"),
        "avg_retrieval_mrr": _mean(results, "retrieval_mrr"),
        "avg_retrieval_recall": _mean(results, "retrieval_recall"),
        "avg_retrieval_precision": _mean(results, "retrieval_precision"),
        "avg_retrieval_ndcg": _mean(results, "retrieval_ndcg"),
        "avg_keyword_coverage": _mean(results, "keyword_coverage"),
        "avg_semantic_similarity": _mean(results, "semantic_similarity"),
    }
    if use_llm_judge:
        summary["avg_judge_faithfulness"] = _mean(results, "judge_faithfulness")
        summary["avg_judge_relevance"] = _mean(results, "judge_relevance")

    from app.core.llm_setup import get_active_models

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active_models": get_active_models(),
        "summary": summary,
        "results": results,
    }
